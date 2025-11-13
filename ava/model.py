import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict, Any, Union
from dataclasses import dataclass
from torch import Tensor
from einops import rearrange, repeat

from .config import ModelConfig, ModelArchitecture, OptimizerType, SchedulerType, PrecisionType

# Helper Functions
def rotate_half(x):
    """Rotate half the hidden dims of the input."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None):
    """Apply rotary positional embeddings to query and key tensors."""
    # q, k: [batch_size, seq_len, num_heads, head_dim]
    # cos, sin: [seq_len, head_dim // 2]
    
    # Reshape for rotary embeddings
    q_ = q.float().reshape(*q.shape[:-1], -1, 2)
    k_ = k.float().reshape(*k.shape[:-1], -1, 2)
    
    # Apply rotation
    q_rot = torch.stack([-q_[..., 1::2], q_[..., ::2]], dim=-1)
    q_rot = q_rot.reshape(q.shape)
    q = q * cos.unsqueeze(1) + q_rot * sin.unsqueeze(1)
    
    k_rot = torch.stack([-k_[..., 1::2], k_[..., ::2]], dim=-1)
    k_rot = k_rot.reshape(k.shape)
    k = k * cos.unsqueeze(1) + k_rot * sin.unsqueeze(1)
    
    return q.type_as(k), k.type_as(k)

# Core Components
class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

class SwiGLU(nn.Module):
    """SwiGLU activation function."""
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return F.silu(gate) * x

class RotaryEmbedding(nn.Module):
    """Rotary positional embeddings (RoPE)."""
    def __init__(self, dim: int, max_seq_len: int = 2048, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._set_cos_sin_cache(max_seq_len, device=self.inv_freq.device)

    def _set_cos_sin_cache(self, seq_len: int, device):
        self.max_seq_len = seq_len
        t = torch.arange(seq_len, device=device, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(torch.float32), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(torch.float32), persistent=False)

    def forward(self, x: torch.Tensor, seq_len: Optional[int] = None):
        if seq_len > self.max_seq_len:
            self._set_cos_sin_cache(seq_len, device=x.device)
        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype, device=x.device),
            self.sin_cached[:seq_len].to(dtype=x.dtype, device=x.device),
        )

class GroupedQueryAttention(nn.Module):
    """Grouped Query Attention with support for multi-head, multi-query, and grouped-query attention."""
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        
        self.rotary_emb = RotaryEmbedding(
            self.head_dim,
            max_seq_len=self.max_position_embeddings,
            base=self.rope_theta,
        )
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        batch_size, seq_len, _ = hidden_states.size()
        
        # Project queries, keys, and values
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        
        # Reshape for attention computation
        query_states = query_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        
        # Apply rotary embeddings
        cos, sin = self.rotary_emb(value_states, seq_len=seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
        
        # Repeat k/v heads if using GQA
        if self.num_key_value_groups > 1:
            key_states = repeat_kv(key_states, self.num_key_value_groups)
            value_states = repeat_kv(value_states, self.num_key_value_groups)
        
        # Cache key and value states for inference
        if use_cache:
            if past_key_value is not None:
                key_states = torch.cat([past_key_value[0], key_states], dim=2)
                value_states = torch.cat([past_key_value[1], value_states], dim=2)
            past_key_value = (key_states, value_states) if use_cache else None
        
        # Scaled dot-product attention
        attn_weights = torch.matmul(
            query_states, key_states.transpose(2, 3)
        ) / math.sqrt(self.head_dim)
        
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
            
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)
        
        # Reshape and project back
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(batch_size, seq_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)
        
        return attn_output, attn_weights, past_key_value

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Repeat key and value tensors for grouped query attention."""
    batch, num_key_value_heads, seq_len, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, seq_len, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, seq_len, head_dim)

class MLP(nn.Module):
    """Multi-Layer Perceptron with SwiGLU activation."""
    def __init__(self, config: ModelConfig):
        super().__init__()
        hidden_size = config.hidden_size
        intermediate_size = config.intermediate_size
        
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act_fn = SwiGLU()
        
    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

class MoE(nn.Module):
    """Mixture of Experts layer."""
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.num_experts = config.moe_num_experts
        self.num_experts_per_tok = config.moe_num_experts_per_tok
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.moe_intermediate_size or config.intermediate_size
        
        # Experts
        self.experts = nn.ModuleList([
            MLP(config) for _ in range(self.num_experts)
        ])
        
        # Router
        self.router = nn.Linear(self.hidden_size, self.num_experts, bias=False)
        self.router_aux_loss_coef = config.moe_router_aux_loss_coef
        
    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        
        # Get router logits and probs
        router_logits = self.router(hidden_states)
        router_probs = F.softmax(router_logits, dim=1, dtype=torch.float32)
        
        # Get top-k experts
        topk_probs, topk_indices = torch.topk(router_probs, self.num_experts_per_tok, dim=1)
        topk_weights = topk_probs / topk_probs.sum(dim=1, keepdim=True)
        
        # Initialize output tensor
        final_hidden_states = torch.zeros_like(hidden_states)
        
        # Auxiliary loss for router
        aux_loss = None
        if self.training:
            router_probs = router_probs.float()
            router_probs_sum = router_probs.sum(0)
            router_probs_sum = router_probs_sum + 1e-9  # Add small epsilon
            router_frac = router_probs_sum / router_probs_sum.sum()
            aux_loss = self.router_aux_loss_coef * (router_frac * torch.log(router_frac * self.num_experts + 1e-9)).sum()
        
        # Dispatch to experts and combine
        for expert_idx, expert in enumerate(self.experts):
            # Get batch indices and weights for this expert
            expert_mask = (topk_indices == expert_idx).any(dim=1)
            if not expert_mask.any():
                continue
                
            # Get the weights for this expert
            expert_weights = torch.where(
                topk_indices == expert_idx,
                topk_weights,
                torch.zeros_like(topk_weights)
            ).sum(dim=1)
            
            # Process with expert
            expert_input = hidden_states[expert_mask]
            expert_output = expert(expert_input)
            
            # Scale by expert weights and add to final output
            final_hidden_states[expert_mask] += expert_output * expert_weights[expert_mask].view(-1, 1)
        
        # Reshape back to original dimensions
        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        
        return final_hidden_states, aux_loss

class TransformerBlock(nn.Module):
    """A transformer block with support for MoE."""
    def __init__(self, config: ModelConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = GroupedQueryAttention(config)
        
        # MoE or dense MLP
        self.is_moe_layer = (
            config.architecture == ModelArchitecture.MIXTRAL and 
            layer_idx % config.moe_layers_interval == 0
        )
        
        if self.is_moe_layer:
            self.mlp = MoE(config)
        else:
            self.mlp = MLP(config)
        
        # Normalization
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        # Self Attention
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        
        # Self-attention with RoPE
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        hidden_states = residual + hidden_states
        
        # MLP or MoE
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        
        if self.is_moe_layer:
            hidden_states, router_logits = self.mlp(hidden_states)
            hidden_states = residual + hidden_states
            return hidden_states, self_attn_weights, present_key_value, router_logits
        else:
            hidden_states = self.mlp(hidden_states)
            hidden_states = residual + hidden_states
            return hidden_states, self_attn_weights, present_key_value

class AvaModel(nn.Module):
    """The core Ava model with support for multiple architectures."""
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        
        # Token embeddings
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerBlock(config, layer_idx=i) 
            for i in range(config.num_hidden_layers)
        ])
        
        # Normalization
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """Initialize weights for different layer types."""
        if isinstance(module, nn.Linear):
            std = self.config.initializer_range
            if hasattr(self.config, 'initializer_factor'):
                std = self.config.initializer_range * self.config.initializer_factor
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
    
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        # Get input embeddings
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
            inputs_embeds = self.embed_tokens(input_ids)
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")
        
        # Prepare attention mask
        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_length), 
                dtype=torch.bool, 
                device=inputs_embeds.device
            )
        
        # Prepare position ids if not provided
        if position_ids is None:
            position_ids = torch.arange(
                0, seq_length, 
                dtype=torch.long, 
                device=inputs_embeds.device
            )
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        
        # Prepare past key values
        if past_key_values is None:
            past_key_values = tuple([None] * len(self.layers))
        
        # Initialize output tensors
        hidden_states = inputs_embeds
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_router_logits = () if self.config.architecture == ModelArchitecture.MIXTRAL else None
        next_decoder_cache = () if use_cache else None
        
        # Forward pass through all layers
        for layer_idx, (block, layer_past) in enumerate(zip(self.layers, past_key_values)):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
                
            # Forward through layer
            layer_outputs = block(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=layer_past,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )
            
            # Handle MoE outputs
            if len(layer_outputs) == 4:  # MoE layer with router logits
                hidden_states, layer_self_attn, present_key_value, router_logits = layer_outputs
                if all_router_logits is not None:
                    all_router_logits = all_router_logits + (router_logits,)
            else:
                hidden_states, layer_self_attn, present_key_value = layer_outputs
            
            if use_cache:
                next_decoder_cache += (present_key_value,)
                
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_self_attn,)
        
        # Apply final normalization
        hidden_states = self.norm(hidden_states)
        
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        
        if not return_dict:
            return tuple(
                v for v in [
                    hidden_states,
                    next_decoder_cache,
                    all_hidden_states,
                    all_self_attentions,
                    all_router_logits,
                ] if v is not None
            )
        
        return {
            "last_hidden_state": hidden_states,
            "past_key_values": next_decoder_cache,
            "hidden_states": all_hidden_states,
            "attentions": all_self_attentions,
            "router_logits": all_router_logits,
        }

class AvaForCausalLM(nn.Module):
    """Ava model for causal language modeling."""
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.model = AvaModel(config)
        
        # Language modeling head
        self.lm_head = nn.Linear(
            config.hidden_size, 
            config.vocab_size, 
            bias=False
        )
        
        # Share weights between input embeddings and output embeddings
        self.lm_head.weight = self.model.embed_tokens.weight
        
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        # Forward pass through the model
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        # Get logits from the language modeling head
        hidden_states = outputs["last_hidden_state"]
        logits = self.lm_head(hidden_states)
        
        # Calculate loss if labels are provided
        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
            
            # Add MoE auxiliary loss if present
            if "router_logits" in outputs and outputs["router_logits"] is not None:
                aux_loss = sum(outputs["router_logits"]) / len(outputs["router_logits"])
                loss += self.config.moe_router_aux_loss_coef * aux_loss
        
        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output
        
        return {
            "loss": loss,
            "logits": logits,
            "past_key_values": outputs["past_key_values"],
            "hidden_states": outputs["hidden_states"],
            "attentions": outputs["attentions"],
            "router_logits": outputs.get("router_logits"),
        }
