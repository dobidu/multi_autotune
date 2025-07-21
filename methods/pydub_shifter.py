#!/usr/bin/env python3
"""
PyDub implementation for pitch shifting.
"""

import numpy as np
from .base_shifter import BasePitchShifter

class PyDubSpeedShifter(BasePitchShifter):
    """
    Pitch shifting using PyDub (speed change).

Characteristics:
- Speed: ★★★★★ (fastest)
- Quality: ★★☆☆☆ (alters duration)
- Resources: ★★★★★ (low usage)
    - Ideal use: Prototyping, creative effects, real-time
    """
    
    def shift_pitch(self, audio: np.ndarray, sr: int, semitones: float) -> np.ndarray:
        """PyDub implementation with quality control."""
        try:
            from pydub import AudioSegment
            
            # Converte para formato PyDub
            audio_int = (audio * 32767).astype(np.int16)
            audio_bytes = audio_int.tobytes()
            segment = AudioSegment(audio_bytes, frame_rate=sr, sample_width=2, channels=1)
            
            # Calculate speed factor
            speed_factor = 2 ** (semitones / 12)
            
            # Apply speed change
            shifted_segment = segment._spawn(
                segment.raw_data, 
                overrides={'frame_rate': int(sr * speed_factor)}
            ).set_frame_rate(sr)
            
            # Converte de volta para numpy
            shifted_bytes = shifted_segment.raw_data
            shifted_audio = np.frombuffer(shifted_bytes, dtype=np.int16)
            
            return shifted_audio.astype(np.float32) / 32767
            
        except Exception as e:
            raise RuntimeError(f"PyDub pitch shift failed: {e}")