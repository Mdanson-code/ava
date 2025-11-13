"""
Ava - Autonomous Open-Source AI Engineer

A self-improving, research-capable AI system designed to plan, design, and 
iteratively improve neural networks and AI systems.

Core Capabilities:
- Self-reasoning and task decomposition
- Autonomous research and planning
- Code generation and self-correction
- Multi-format knowledge representation
- Continuous learning and evaluation
- Safety and ethics awareness
"""

__version__ = "1.0.0"
__author__ = "Ava AI Engineering Team"

from .config import (
    ModelConfig,
    ModelArchitecture,
    OptimizerType,
    SchedulerType,
    PrecisionType,
    AgentConfig,
)

from .model import (
    AvaModel,
    AvaForCausalLM,
    RMSNorm,
    RotaryEmbedding,
    GroupedQueryAttention,
    MoE,
)

__all__ = [
    # Config
    "ModelConfig",
    "ModelArchitecture",
    "OptimizerType",
    "SchedulerType",
    "PrecisionType",
    "AgentConfig",
    # Models
    "AvaModel",
    "AvaForCausalLM",
    "RMSNorm",
    "RotaryEmbedding",
    "GroupedQueryAttention",
    "MoE",
]
