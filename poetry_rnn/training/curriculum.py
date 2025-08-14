"""
Curriculum Learning Scheduler for RNN Training

Implements curriculum learning strategies to improve RNN training on
variable-length sequences, starting with shorter sequences and gradually
increasing complexity.
"""

import torch
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass


@dataclass
class CurriculumPhase:
    """
    Represents a single phase in curriculum learning.
    
    Attributes:
        max_length: Maximum sequence length for this phase
        num_epochs: Number of epochs to train in this phase
        description: Human-readable description of the phase
        learning_rate_scale: Optional LR scaling factor for this phase
    """
    max_length: int
    num_epochs: int
    description: str
    learning_rate_scale: float = 1.0


class CurriculumScheduler:
    """
    Manages curriculum learning schedule for RNN training.
    
    Curriculum learning helps RNNs learn more effectively by starting with
    simpler (shorter) sequences and gradually increasing difficulty. This
    addresses gradient flow issues and helps the model learn basic patterns
    before tackling complex long-range dependencies.
    
    Theory Foundation:
        Based on analysis showing RNN gradient norm decays exponentially
        with sequence length. Starting with shorter sequences where gradients
        are stronger helps establish good initial representations.
    
    Args:
        phases: List of curriculum phases or None for default schedule
        adaptive: Whether to adapt schedule based on performance
        patience: Epochs to wait before moving to next phase if adaptive
    """
    
    def __init__(
        self,
        phases: Optional[List[CurriculumPhase]] = None,
        adaptive: bool = False,
        patience: int = 3
    ):
        # Default curriculum for poetry (max length 50)
        if phases is None:
            phases = [
                CurriculumPhase(
                    max_length=20,
                    num_epochs=10,
                    description="Short sequences (≤20 tokens)",
                    learning_rate_scale=1.0
                ),
                CurriculumPhase(
                    max_length=35,
                    num_epochs=15,
                    description="Medium sequences (≤35 tokens)",
                    learning_rate_scale=0.8
                ),
                CurriculumPhase(
                    max_length=50,
                    num_epochs=25,
                    description="Full sequences (≤50 tokens)",
                    learning_rate_scale=0.6
                )
            ]
        
        self.phases = phases
        self.adaptive = adaptive
        self.patience = patience
        
        # Tracking variables
        self.current_phase_idx = 0
        self.epochs_in_phase = 0
        self.total_epochs = 0
        self.best_loss = float('inf')
        self.epochs_without_improvement = 0
        
        # Compute total planned epochs
        self.total_planned_epochs = sum(phase.num_epochs for phase in phases)
        
        # Teacher forcing schedule per phase
        # Phase 1: High guidance (0.9 → 0.7)
        # Phase 2: Medium guidance (0.7 → 0.3)  
        # Phase 3: Low guidance (0.3 → 0.1)
        self.teacher_forcing_schedule = [
            {'start': 0.9, 'end': 0.7, 'min_ratio': 0.05},  # Phase 1
            {'start': 0.7, 'end': 0.3, 'min_ratio': 0.05},  # Phase 2
            {'start': 0.3, 'end': 0.1, 'min_ratio': 0.05}   # Phase 3
        ]
    
    @property
    def current_phase(self) -> CurriculumPhase:
        """Get the current curriculum phase."""
        return self.phases[self.current_phase_idx]
    
    @property
    def is_complete(self) -> bool:
        """Check if curriculum is complete."""
        return self.current_phase_idx >= len(self.phases)
    
    def get_max_length(self) -> int:
        """Get current maximum sequence length."""
        if self.is_complete:
            return self.phases[-1].max_length
        return self.current_phase.max_length
    
    def get_learning_rate_scale(self) -> float:
        """Get learning rate scaling factor for current phase."""
        if self.is_complete:
            return self.phases[-1].learning_rate_scale
        return self.current_phase.learning_rate_scale
    
    def get_teacher_forcing_ratio(self) -> float:
        """
        Get current teacher forcing ratio with adaptive linear decay within phase.
        
        Uses linear interpolation between start and end ratios based on progress
        within the current curriculum phase. Implements the scheduled sampling
        strategy for stable RNN autoencoder training.
        
        Returns:
            Teacher forcing ratio between 0 and 1, with minimum threshold
        """
        if self.is_complete:
            # Use final phase minimum when curriculum complete
            return self.teacher_forcing_schedule[-1]['min_ratio']
        
        # Get current phase schedule
        current_schedule = self.teacher_forcing_schedule[self.current_phase_idx]
        start_ratio = current_schedule['start']
        end_ratio = current_schedule['end']
        min_ratio = current_schedule['min_ratio']
        
        # Calculate progress within current phase (0.0 to 1.0)
        phase_progress = self.epochs_in_phase / self.current_phase.num_epochs
        phase_progress = min(1.0, phase_progress)  # Clamp to [0, 1]
        
        # Linear decay within phase
        current_ratio = start_ratio - (phase_progress * (start_ratio - end_ratio))
        
        # Apply minimum threshold
        current_ratio = max(current_ratio, min_ratio)
        
        return current_ratio
    
    def truncate_batch(
        self,
        batch_dict: Dict[str, torch.Tensor],
        max_length: Optional[int] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Truncate sequences in batch to current curriculum length.
        
        Args:
            batch_dict: Batch dictionary from DataLoader
            max_length: Override max length (uses current phase if None)
        
        Returns:
            Truncated batch dictionary
        """
        if max_length is None:
            max_length = self.get_max_length()
        
        # Get current sequence length
        current_length = batch_dict['input_sequences'].shape[1]
        
        if current_length <= max_length:
            return batch_dict
        
        # Truncate sequences and mask
        truncated_batch = {
            'input_sequences': batch_dict['input_sequences'][:, :max_length, :],
            'metadata': batch_dict['metadata']
        }
        
        if 'attention_mask' in batch_dict:
            truncated_batch['attention_mask'] = batch_dict['attention_mask'][:, :max_length]
        
        return truncated_batch
    
    def step(self, val_loss: Optional[float] = None) -> bool:
        """
        Update curriculum state after an epoch.
        
        Args:
            val_loss: Validation loss for adaptive scheduling
        
        Returns:
            True if moved to next phase, False otherwise
        """
        self.epochs_in_phase += 1
        self.total_epochs += 1
        
        # Check if should move to next phase
        should_advance = False
        
        if self.adaptive and val_loss is not None:
            # Adaptive scheduling based on validation performance
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                self.epochs_without_improvement = 0
            else:
                self.epochs_without_improvement += 1
            
            # Advance if no improvement for patience epochs
            if self.epochs_without_improvement >= self.patience:
                should_advance = True
        else:
            # Fixed scheduling based on epoch count
            if self.epochs_in_phase >= self.current_phase.num_epochs:
                should_advance = True
        
        if should_advance and not self.is_complete:
            self.advance_phase()
            return True
        
        return False
    
    def advance_phase(self):
        """Move to the next curriculum phase."""
        if self.current_phase_idx < len(self.phases) - 1:
            self.current_phase_idx += 1
            self.epochs_in_phase = 0
            self.epochs_without_improvement = 0
            self.best_loss = float('inf')
            
            print(f"\n{'='*60}")
            print(f"Advancing to Phase {self.current_phase_idx + 1}: {self.current_phase.description}")
            print(f"  Max sequence length: {self.current_phase.max_length}")
            print(f"  Planned epochs: {self.current_phase.num_epochs}")
            print(f"  Learning rate scale: {self.current_phase.learning_rate_scale}")
            print(f"{'='*60}\n")
    
    def get_progress(self) -> Dict[str, Any]:
        """
        Get current curriculum progress information.
        
        Returns:
            Dictionary with progress metrics
        """
        return {
            'current_phase': self.current_phase_idx + 1,
            'total_phases': len(self.phases),
            'phase_description': self.current_phase.description,
            'epochs_in_phase': self.epochs_in_phase,
            'total_epochs': self.total_epochs,
            'max_length': self.get_max_length(),
            'lr_scale': self.get_learning_rate_scale(),
            'progress_percent': (self.total_epochs / self.total_planned_epochs) * 100
        }
    
    def reset(self):
        """Reset curriculum to start from beginning."""
        self.current_phase_idx = 0
        self.epochs_in_phase = 0
        self.total_epochs = 0
        self.best_loss = float('inf')
        self.epochs_without_improvement = 0
    
    def summary(self) -> str:
        """
        Generate a summary of the curriculum schedule.
        
        Returns:
            String summary of all phases
        """
        lines = ["Curriculum Learning Schedule", "="*40]
        
        for i, phase in enumerate(self.phases, 1):
            status = "CURRENT" if i-1 == self.current_phase_idx else ""
            lines.append(f"Phase {i}: {phase.description} {status}")
            lines.append(f"  - Max length: {phase.max_length}")
            lines.append(f"  - Epochs: {phase.num_epochs}")
            lines.append(f"  - LR scale: {phase.learning_rate_scale}")
        
        lines.append(f"\nTotal planned epochs: {self.total_planned_epochs}")
        lines.append(f"Current progress: {self.total_epochs}/{self.total_planned_epochs} epochs")
        
        return "\n".join(lines)