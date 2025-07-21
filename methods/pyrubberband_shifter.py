#!/usr/bin/env python3
"""
PyRubberband implementation for pitch shifting.
"""

import subprocess
import numpy as np
from .base_shifter import BasePitchShifter

class PyRubberbandShifter(BasePitchShifter):
    """
    Pitch shifting usando pyrubberband.
    
    Characteristics:
- Speed: ★★★☆☆ (medium)
- Quality: ★★★★★ (excellent)
    - Recursos: ★★★☆☆ (requer rubberband-cli)
    - Uso ideal: Produção profissional, masterização
    """
    
    def __init__(self, config):
        """Initialize with external dependency verification."""
        super().__init__(config)
        self._check_external_dependency()
    
    def _check_external_dependency(self) -> bool:
        """Check if rubberband-cli is available."""
        try:
            result = subprocess.run(["rubberband", "--help"], 
                                  capture_output=True, timeout=5)
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            raise RuntimeError(
                "pyrubberband requires rubberband-cli installed.\n"
                "Ubuntu/Debian: sudo apt-get install rubberband-cli\n"
                "macOS: brew install rubberband\n"
                "Windows: download from https://breakfastquay.com/rubberband/"
            )
    
    def shift_pitch(self, audio: np.ndarray, sr: int, semitones: float) -> np.ndarray:
        """Implementation with pyrubberband."""
        try:
            import pyrubberband as pyrb
            
            # Optimized settings
            return pyrb.pitch_shift(
                audio, sr, 
                n_steps=semitones,
                rbargs={'--crisp': '6'}  # High quality
            )
            
        except Exception as e:
            raise RuntimeError(f"Pyrubberband pitch shift failed: {e}")