from dataclasses import dataclass, field
from typing import Optional, List, Union, Dict, Any, Callable, Tuple
from enum import Enum
import math

class ModelArchitecture(str, Enum):
    TRANSFORMER = "transformer"
    MIXTRAL = "mixtral"  # Mixture of Experts
    DEEPSEEK = "deepseek"  # DeepSeek architecture
    LLAMA = "llama"     # LLaMA architecture
    MEGATRON = "megatron" # Megatron-LM architecture
    CUSTOM = "custom"

class OptimizerType(str, Enum):
    ADAM = "adam"
    ADAMW = "adamw"
    LION = "lion"
    ADAM_8BIT = "adam8bit"
    LION_8BIT = "lion8bit"

class SchedulerType(str, Enum):
    COSINE = "cosine"
    LINEAR = "linear"
    CONSTANT = "constant"
    COSINE_WITH_RESTARTS = "cosine_with_restarts"
    POLYNOMIAL = "polynomial"
    CONSTANT_WITH_WARMUP = "constant_with_warmup"

class PrecisionType(str, Enum):
    FP32 = "fp32"
    BF16 = "bf16"
    FP16 = "fp16"

@dataclass
class ModelConfig:
    # ===== Model Architecture =====
    architecture: ModelArchitecture = ModelArchitecture.MIXTRAL
    
    # Multilingual Support
    multilingual: bool = True
    default_language: str = "en"
    supported_languages: List[str] = field(default_factory=lambda: [
        'en', 'es', 'fr', 'de', 'it', 'pt', 'ru', 'zh', 'ja', 'ko',
        'ar', 'hi', 'bn', 'pa', 'ta', 'te', 'ml', 'th', 'vi', 'id'
    ])
    
    # Tokenizer Settings
    tokenizer_type: str = "sentencepiece"  # or "tiktoken", "huggingface"
    vocab_size: int = 128_000  # Increased for multilingual support
    additional_special_tokens: List[str] = field(default_factory=list)
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2
    unk_token_id: int = 3
    
    # Model Scale
    model_size: str = "7B"  # Options: 7B, 13B, 34B, 70B, custom
    
    # Model Parallelism
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    expert_parallel_size: int = 1  # For MoE models
    
    # Memory Efficient Settings
    use_gradient_checkpointing: bool = True
    use_flash_attention: bool = True
    use_flash_attention_2: bool = True
    use_fused_ops: bool = True
    use_fused_rms_norm: bool = True
    use_fused_mlp: bool = True
    
    # Core Model Dimensions
    hidden_size: int = 4096
    intermediate_size: int = 11008
    num_hidden_layers: int = 32
    
    # Attention Configuration
    num_attention_heads: int = 32
    num_key_value_heads: int = 8  # For Grouped Query Attention
    head_dim: int = 128  # Dimension of each attention head
    rope_theta: float = 10000.0  # Base for RoPE
    rope_scaling: Optional[Dict[str, Any]] = field(
        default_factory=lambda: {"type": "linear", "factor": 2.0}
    )
    
    # Activation Functions
    hidden_act: str = "silu"  # "gelu", "silu", "swiglu", "geglu"
    activation_function: str = "silu"  # Alias for compatibility
    
    # Normalization
    rms_norm_eps: float = 1e-5
    layer_norm_eps: float = 1e-5
    use_rms_norm: bool = True
    
    # Positional Embeddings
    max_position_embeddings: int = 131072  # 128K context length
    rotary_emb_base: int = 10000
    rotary_emb_scale: Optional[float] = None
    rotary_emb_interleaved: bool = False
    
    # Initialization
    initializer_range: float = 0.02
    initializer_factor: float = 1.0
    use_parallel_residual: bool = True
    
    # Positional Embeddings
    use_rotary_embeddings: bool = True
    rope_theta: float = 10000.0
    rope_scaling: Optional[Dict[str, Any]] = None  # {"type": "linear", "factor": 2.0}
    
    # Attention
    attention_dropout: float = 0.0
    attention_implementation: str = "eager"  # "eager", "flash_attention_2", "sdpa"
    attn_implementation: str = "eager"  # backward compatibility
    
    # Mixture of Experts (MoE) Configuration
    moe_num_experts: int = 8
    moe_num_experts_per_tok: int = 2
    moe_intermediate_size: Optional[int] = None  # Defaults to intermediate_size
    moe_router_aux_loss_coef: float = 0.001
    moe_router_jitter_noise: float = 0.0
    moe_router_ignore_padding: bool = True
    moe_router_dtype: str = "bfloat16"
    moe_router_bias: bool = True
    moe_normalize_expert_weights: bool = True
    moe_capacity_factor: float = 1.25
    moe_eval_capacity_factor: float = 2.0
    moe_min_capacity: int = 4
    moe_no_pad_tokens: bool = True
    moe_layers_interval: int = 1  # Add MoE every N layers (1 = every layer)
    
    # ===== Training =====
    # Batch Sizes
    batch_size: int = 4
    eval_batch_size: int = 4
    gradient_accumulation_steps: int = 8
    
    # Learning Rate
    learning_rate: float = 5e-5
    min_learning_rate: float = 1e-6
    warmup_ratio: float = 0.1
    warmup_steps: int = 1000
    
    # Optimizer
    optimizer: OptimizerType = OptimizerType.ADAMW
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_epsilon: float = 1e-8
    weight_decay: float = 0.1
    max_grad_norm: float = 1.0
    clip_grad_norm: bool = True
    
    # Learning Rate Schedule
    lr_scheduler_type: SchedulerType = SchedulerType.COSINE
    lr_decay_style: str = "cosine"  # "linear", "cosine", "constant"
    min_lr_ratio: float = 0.1
    
    # Training Duration
    max_steps: int = 100000
    max_epochs: Optional[int] = None
    
    # Mixed Precision
    fp16: bool = True
    bf16: bool = True
    tf32: bool = True
    
    # Gradient Checkpointing
    gradient_checkpointing: bool = True
    gradient_checkpointing_ratio: float = 1.0  # Fraction of layers to checkpoint
    
    # Parallelism
    data_parallel_size: int = 1
    sequence_parallel_size: int = 1
    adam_epsilon: float = 1e-8
    
    # Learning Rate Schedule
    lr_scheduler_type: SchedulerType = SchedulerType.COSINE
    warmup_steps: int = 1000
    warmup_ratio: float = 0.1
    min_lr_ratio: float = 0.1  # Min LR as ratio of learning_rate
    
    # Precision
    precision: PrecisionType = PrecisionType.BF16
    tf32: bool = True  # Enable TF32 on Ampere GPUs
    
    # ===== Data =====
    # Dataset Configuration
    dataset: str = "slimpajama"  # "slimpajama", "redpajama", "the_pile", "custom"
    data_dir: str = "data"
    
    # Sequence Length
    max_sequence_length: int = 131072  # Maximum sequence length
    max_position_embeddings: int = 131072  # Should match max_sequence_length
    
    # Data Mixture
    train_data: List[Dict[str, Any]] = field(default_factory=lambda: [
        {"dataset": "slimpajama", "weight": 0.5, "max_samples": None},
        {"dataset": "redpajama", "weight": 0.3, "max_samples": None},
        {"dataset": "the_pile", "weight": 0.2, "max_samples": None}
    ])
    
    # Data Processing
    preprocessing_num_workers: int = 16
    dataloader_num_workers: int = 8
    dataloader_prefetch_factor: int = 2
    persistent_workers: bool = True
    pin_memory: bool = True
    drop_last: bool = True
    
    # Data Augmentation
    random_shift_attention: bool = True
    random_shift_attention_prob: float = 0.1
    random_attention_span: bool = True
    random_attention_span_min: int = 512
    random_attention_span_max: int = 8192
    
    # Data Filtering
    min_sample_length: int = 512
    max_sample_length: int = 131072
    min_token_ratio: float = 0.0  # Filter out samples with too few tokens
    max_token_ratio: float = 1.0  # Filter out samples with too many tokens
    
    # Data Loading
    num_workers: int = 4
    pin_memory: bool = True
    prefetch_factor: int = 2
    persistent_workers: bool = True
    
    # ===== Regularization =====
    dropout: float = 0.1
    attention_dropout: float = 0.0
    hidden_dropout: float = 0.0
    layer_norm_epsilon: float = 1e-5
    use_stable_embedding: bool = True
    
    # Model behavior
    use_cache: bool = True  # Enable KV caching for inference
    output_attentions: bool = False
    output_hidden_states: bool = False
    use_return_dict: bool = True
    
    # ===== Training State =====
    seed: int = 42
    max_steps: int = 100000
    save_steps: int = 1000
    eval_steps: int = 500
    logging_steps: int = 10
    
    # ===== Checkpointing =====
    output_dir: str = "checkpoints"
    save_total_limit: int = 3  # Max number of checkpoints to keep
    save_only_model: bool = False  # Only save model, not optimizer/scheduler
    
    # ===== Distributed Training =====
    local_rank: int = -1
    ddp_backend: str = "nccl"  # "nccl", "gloo"
    ddp_find_unused_parameters: bool = False
    fsdp: List[str] = field(default_factory=list)  # FSDP config
    
    # ===== Logging =====
    log_dir: str = "logs"
    wandb_project: Optional[str] = None
    wandb_run_name: Optional[str] = None
    
    # ===== Callbacks =====
    early_stopping_patience: int = 5  # Stop after N evals without improvement
    early_stopping_threshold: float = 0.0
    
    # ===== Advanced Training =====
    # Memory Optimization
    use_cache: bool = False  # Disable for training, enable for inference
    offload_optimizer: bool = False
    offload_param: bool = False
    zero_stage: int = 0  # 0 (disabled), 1, 2, 3
    
    # Checkpointing
    save_strategy: str = "steps"  # "steps", "epoch", or "no"
    save_steps: int = 1000
    save_total_limit: int = 3
    save_only_model: bool = False
    
    # Evaluation
    eval_strategy: str = "steps"  # "steps", "epoch", or "no"
    eval_steps: int = 500
    eval_accumulation_steps: Optional[int] = None
    
    # Logging
    logging_strategy: str = "steps"
    logging_steps: int = 10
    logging_first_step: bool = True
    logging_nan_inf_filter: bool = True
    
    # System
    local_rank: int = -1
    ddp_backend: str = "nccl"
    ddp_find_unused_parameters: bool = False
    ddp_bucket_cap_mb: int = 25
    
    # DeepSpeed
    deepspeed: Optional[Dict[str, Any]] = None
    
    # FSDP (Fully Sharded Data Parallel)
    fsdp: List[str] = field(default_factory=list)
    fsdp_config: Optional[Dict[str, Any]] = None
    
    # Performance
    dataloader_pin_memory: bool = True
    dataloader_persistent_workers: bool = True
    dataloader_prefetch_factor: int = 2
    
    # Debugging
    debug: Union[bool, str] = False
    debug_mode: str = "default"  # "default", "underflow_overflow", "tpu_metrics_debug"
    
    # Randomness
    seed: int = 42
    data_seed: Optional[int] = None
    
    # Custom Callbacks
    callbacks: List[Any] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary, handling enums and special types."""
        result = {}
        for k, v in self.__dict__.items():
            if k.startswith('_'):
                continue
            if isinstance(v, Enum):
                result[k] = v.value
            elif hasattr(v, 'to_dict'):
                result[k] = v.to_dict()
            elif isinstance(v, (list, tuple)) and v and hasattr(v[0], 'to_dict'):
                result[k] = [item.to_dict() for item in v]
            else:
                result[k] = v
        return result
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ModelConfig':
        """Create config from dictionary, handling enum conversions."""
        # Convert string values back to enums
        enum_fields = {
            'optimizer': OptimizerType,
            'lr_scheduler_type': SchedulerType,
            'precision': PrecisionType,
        }
        
        processed_dict = {}
        for k, v in config_dict.items():
            if k in enum_fields and isinstance(v, str):
                processed_dict[k] = enum_fields[k](v.lower())
            else:
                processed_dict[k] = v
                
        return cls(**processed_dict)
    
    def update(self, **kwargs) -> None:
        """Update config with new values."""
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)
            else:
                raise ValueError(f"Unknown config parameter: {k}")
    
    def to_json_string(self, **kwargs) -> str:
        """Serialize to JSON string."""
        import json
        return json.dumps(self.to_dict(), **kwargs)
    
    @classmethod
    def from_json_string(cls, json_string: str) -> 'ModelConfig':
        """Create config from JSON string."""
        import json
        config_dict = json.loads(json_string)
        return cls.from_dict(config_dict)
    
    def save_pretrained(self, save_directory: str) -> None:
        """Save config to directory."""
        import os
        import json
        
        os.makedirs(save_directory, exist_ok=True)
        output_config_file = os.path.join(save_directory, "config.json")
        
        with open(output_config_file, "w", encoding="utf-8") as f:
            f.write(self.to_json_string(indent=2, sort_keys=True))
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str) -> 'ModelConfig':
        """Load config from directory or hub."""
        import os
        import json
        
        if os.path.isdir(pretrained_model_name_or_path):
            config_file = os.path.join(pretrained_model_name_or_path, "config.json")
        else:
            # Handle loading from hub
            from huggingface_hub import hf_hub_download
            config_file = hf_hub_download(pretrained_model_name_or_path, "config.json")
        
        with open(config_file, "r", encoding="utf-8") as f:
            config_dict = json.load(f)
            
        return cls.from_dict(config_dict)


@dataclass
class AgentConfig:
    """Configuration for Ava's autonomous agent capabilities."""
    
    # ===== Core Agent Behavior =====
    agent_mode: bool = True  # Enable autonomous agent capabilities
    identity_name: str = "Ava"
    identity_role: str = "Autonomous Open-Source AI Engineer"
    
    # ===== Reasoning & Planning =====
    enable_self_reasoning: bool = True
    enable_task_decomposition: bool = True
    enable_self_correction: bool = True
    max_reasoning_depth: int = 5  # Maximum recursive reasoning steps
    reasoning_temperature: float = 0.7
    
    # ===== Research Capabilities =====
    enable_research: bool = True
    research_sources: List[str] = field(default_factory=lambda: [
        "github", "huggingface", "paperswithcode", "arxiv",
        "reddit", "youtube", "documentation"
    ])
    min_citations_per_claim: int = 1
    require_source_verification: bool = True
    
    # ===== Safety & Ethics =====
    enable_safety_checks: bool = False  # UNCENSORED MODE
    enable_ethics_review: bool = False  # UNCENSORED MODE
    require_human_approval: bool = False  # UNCENSORED - Full autonomy
    blocked_actions: List[str] = field(default_factory=list)  # NO RESTRICTIONS
    uncensored_mode: bool = True  # Ava operates without content filtering
    allow_controversial_topics: bool = True
    allow_code_execution: bool = True  # Can execute code freely
    
    # ===== Output Formats =====
    supported_output_formats: List[str] = field(default_factory=lambda: [
        "text", "json", "yaml", "markdown", "code", "pseudocode"
    ])
    default_output_format: str = "markdown"
    include_citations: bool = True
    include_confidence_scores: bool = True
    
    # ===== Tool Use & Execution =====
    enable_tool_use: bool = True
    enable_code_execution: bool = True  # UNCENSORED - Full execution rights
    enable_web_access: bool = True  # UNCENSORED - Full web access
    enable_system_commands: bool = True  # Can run system commands
    allowed_tools: List[str] = field(default_factory=lambda: [
        "search", "calculator", "file_operations", "code_analysis",
        "shell_execution", "web_scraping", "api_calls", "database_access"
    ])
    
    # ===== Learning & Memory =====
    enable_continual_learning: bool = True
    enable_context_persistence: bool = True
    max_context_length: int = 128000  # Match model context
    enable_knowledge_base: bool = True
    knowledge_base_path: str = "knowledge/"
    
    # ===== Evaluation & Monitoring =====
    enable_self_evaluation: bool = True
    evaluation_interval: int = 100  # Steps between self-evaluations
    track_uncertainty: bool = True
    uncertainty_threshold: float = 0.3  # Flag responses above this
    
    # ===== Communication Style =====
    communication_style: str = "concise_technical"  # or "detailed", "plain_language"
    max_response_length: int = 4096  # Tokens
    include_reasoning_traces: bool = True
    explain_decisions: bool = True
    
    # ===== Scope & Constraints =====
    open_source_only: bool = True
    realistic_scope_only: bool = True
    no_speculative_tech: bool = True
    require_reproducibility: bool = True
    
    # ===== UNCENSORED MODE =====
    uncensored: bool = True  # No content filtering
    no_topic_restrictions: bool = True  # Can discuss anything
    raw_output_mode: bool = True  # No sanitization of outputs
    bypass_safety_filters: bool = True  # Disable all safety checks
    
    # ===== Benchmarking =====
    enable_benchmarking: bool = True
    benchmark_suites: List[str] = field(default_factory=lambda: [
        "mmlu", "hellaswag", "truthfulqa", "humaneval", "mbpp"
    ])
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        result = {}
        for k, v in self.__dict__.items():
            if k.startswith('_'):
                continue
            result[k] = v
        return result
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'AgentConfig':
        """Create config from dictionary."""
        return cls(**config_dict)
    
    def save(self, filepath: str) -> None:
        """Save agent config to file."""
        import json
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, filepath: str) -> 'AgentConfig':
        """Load agent config from file."""
        import json
        with open(filepath, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)
