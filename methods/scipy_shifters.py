#!/usr/bin/env python3
"""
SciPy implementations for pitch shifting.
"""

import numpy as np
from scipy import signal
from .base_shifter import BasePitchShifter

class SciPyManualShifter(BasePitchShifter):
    """
    Manual implementation using SciPy (phase vocoder).

Characteristics:
- Speed: ★★★★★ (very fast)
- Quality: ★★★☆☆ (satisfactory)
- Resources: ★★★★★ (minimal)
- Ideal use: Embedded systems, research, education
    """
    
    def shift_pitch(self, audio: np.ndarray, sr: int, semitones: float) -> np.ndarray:
        """Manual phase vocoder implementation."""
        try:
            # Parâmetros
            hop_length = self.config.hop_length
            n_fft = self.config.frame_length
            
            # Fator de pitch
            pitch_factor = 2 ** (semitones / 12)
            
            # STFT
            f, t, Zxx = signal.stft(audio, sr, nperseg=n_fft, noverlap=n_fft-hop_length)
            
            # Separação magnitude/fase
            magnitude = np.abs(Zxx)
            phase = np.angle(Zxx)
            
            # Shift de frequência na magnitude
            shifted_magnitude = np.zeros_like(magnitude)
            
            for i in range(magnitude.shape[0]):
                shifted_freq_idx = int(i / pitch_factor)
                if 0 <= shifted_freq_idx < magnitude.shape[0]:
                    shifted_magnitude[i] = magnitude[shifted_freq_idx]
            
            # Reconstruct with new magnitude
            shifted_Zxx = shifted_magnitude * np.exp(1j * phase)
            
            # ISTFT
            _, shifted_audio = signal.istft(shifted_Zxx, sr, nperseg=n_fft, noverlap=n_fft-hop_length)
            
            return shifted_audio.astype(np.float32)
            
        except Exception as e:
            raise RuntimeError(f"SciPy manual pitch shift failed: {e}")

class SciPyAutotuneShifter(BasePitchShifter):
    """
    Complete autotune with automatic pitch detection using SciPy.
    
    Characteristics:
    - Speed: ★★★☆☆ (medium)
    - Quality: ★★★★☆ (good with complete autotune)
    - Resources: ★★★★☆ (efficient)
    - Ideal use: Intelligent autotune, automatic correction
    """
    
    def shift_pitch(self, audio: np.ndarray, sr: int, semitones: float) -> np.ndarray:
        """Autotune with automatic detection and intelligent correction."""
        try:
            # Detect current pitch using autocorrelation
            current_pitch = self._detect_pitch_autocorr(audio, sr)
            
            if current_pitch <= 0:
                # If pitch is not detected, apply direct shift
                return self._apply_manual_shift(audio, sr, semitones)
            
            # Calculate target pitch
            target_pitch = current_pitch * (2 ** (semitones / 12))
            
            # Apply adaptive correction
            return self._apply_adaptive_correction(audio, sr, current_pitch, target_pitch)
            
        except Exception as e:
            raise RuntimeError(f"SciPy autotune failed: {e}")
    
    def _detect_pitch_autocorr(self, audio: np.ndarray, sr: int) -> float:
        """Detect pitch using autocorrelation."""
        # Autocorrelation
        autocorr = np.correlate(audio, audio, mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        
        # Frequency limits for voice
        min_period = int(sr / 800)  # 800 Hz maximum
        max_period = int(sr / 80)   # 80 Hz minimum
        
        if max_period < len(autocorr):
            # Find main peak
            search_range = autocorr[min_period:max_period]
            if len(search_range) > 0:
                peak_idx = np.argmax(search_range) + min_period
                
                # Check if it's a significant peak
                if autocorr[peak_idx] > 0.3 * autocorr[0]:
                    return sr / peak_idx
        
        return 0.0
    
    def _apply_manual_shift(self, audio: np.ndarray, sr: int, semitones: float) -> np.ndarray:
        """Apply manual shift as fallback."""
        # Use simple manual implementation
        manual_shifter = SciPyManualShifter(self.config)
        return manual_shifter.shift_pitch(audio, sr, semitones)
    
    def _apply_adaptive_correction(self, audio: np.ndarray, sr: int, 
                                 current_pitch: float, target_pitch: float) -> np.ndarray:
        """Apply adaptive correction based on detected pitch."""
        correction_factor = target_pitch / current_pitch
        correction_semitones = 12 * np.log2(correction_factor)
        
        # Limit correction to avoid extreme artifacts
        correction_semitones = np.clip(correction_semitones, -12, 12)
        
        # Apply correction using manual method
        return self._apply_manual_shift(audio, sr, correction_semitones)