"""
Poetry Generation Configuration and Utilities

This module contains configuration classes and utilities for poetry generation
with advanced sampling strategies and style conditioning.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class PoetryGenerationConfig:
    """Configuration for poetry generation with advanced sampling strategies."""
    
    temperature: float = 0.8
    top_k: Optional[int] = 50
    top_p: float = 0.9
    repetition_penalty: float = 1.1
    length_penalty: float = 1.0
    use_attention_guidance: bool = True
    preserve_meter: bool = False
    style_conditioning: Optional[str] = None