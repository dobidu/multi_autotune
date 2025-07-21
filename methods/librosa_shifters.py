#!/usr/bin/env python3
"""
LibROSA implementations for pitch shifting.
"""

import numpy as np
from .base_shifter import BasePitchShifter

class LibROSAStandardShifter(BasePitchShifter):
    """
    Pitch shifting using LibROSA with standard settings.

Characteristics:
- Speed: ★★★★☆ (fast)
- Quality: ★★★★☆ (good)
    - Recursos: ★★★★☆ (eficiente)
    - Ideal use: General use, development, analysis
    """
    
    def shift_pitch(self, audio: np.ndarray, sr: int, semitones: float) -> np.ndarray:
        """Standard LibROSA implementation."""
        try:
            import librosa
            
            return librosa.effects.pitch_shift(
                y=audio,
                sr=sr,
                n_steps=semitones,
                bins_per_octave=12,
                res_type='kaiser_fast'
            )
            
        except Exception as e:
            raise RuntimeError(f"LibROSA standard pitch shift failed: {e}")

class LibROSAHiFiShifter(BasePitchShifter):
    """
    Pitch shifting using LibROSA with high quality settings.
    
    Characteristics:
    - Speed: ★★★☆☆ (medium)
    - Quality: ★★★★★ (excellent)
    - Resources: ★★★☆☆ (moderate usage)
    - Ideal use: High quality, mastering
    """
    
    def shift_pitch(self, audio: np.ndarray, sr: int, semitones: float) -> np.ndarray:
        """High quality LibROSA implementation."""
        try:
            import librosa
            
            return librosa.effects.pitch_shift(
                y=audio,
                sr=sr,
                n_steps=semitones,
                bins_per_octave=12,
                res_type='kaiser_best',
                hop_length=self.config.hop_length // 2  # Maior resolução
            )
            
        except Exception as e:
            raise RuntimeError(f"LibROSA hi-fi pitch shift failed: {e}")