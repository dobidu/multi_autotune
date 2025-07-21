#!/usr/bin/env python3
"""
Base class for pitch shifting implementations.
"""

import time
import logging
from typing import Tuple, Dict
import numpy as np
from scipy import signal

class BasePitchShifter:
    """Base class for pitch shifting implementations."""
    
    def __init__(self, config):
        """Initialize shifter."""
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.method_name = self.__class__.__name__
    
    def shift_pitch(self, audio: np.ndarray, sr: int, semitones: float) -> np.ndarray:
        """
        Apply pitch shifting. Must be implemented by subclasses.
        
        Args:
            audio: Audio signal
            sr: Sample rate
            semitones: Number of semitones to shift
            
        Returns:
            Audio with altered pitch
        """
        raise NotImplementedError("Subclasses must implement shift_pitch")
    
    def validate_inputs(self, audio: np.ndarray, sr: int, semitones: float) -> Tuple[bool, str]:
        """Validates method inputs."""
        if len(audio) == 0:
            return False, "Empty audio"
        
        if sr <= 0:
            return False, f"Invalid sample rate: {sr}"
        
        if not -24 <= semitones <= 24:
            return False, f"Semitones out of range (-24 to 24): {semitones}"
        
        if abs(semitones) < 0.01:
            return False, "Shift too small (< 0.01 semitones)"
        
        return True, ""
    
    def preprocess_audio(self, audio: np.ndarray) -> np.ndarray:
        """Audio preprocessing."""
        if not self.config.enable_preprocessing:
            return audio
        
        processed = audio.copy()
        
        # Normalization
        max_val = np.max(np.abs(processed))
        if max_val > 1.0:
            processed = processed / max_val * 0.95
            self.logger.debug("Audio normalized in preprocessing")
        
        # Remove DC offset
        dc_offset = np.mean(processed)
        if abs(dc_offset) > 0.01:
            processed = processed - dc_offset
            self.logger.debug(f"DC offset removed: {dc_offset:.4f}")
        
        return processed
    
    def postprocess_audio(self, audio: np.ndarray, original_length: int) -> np.ndarray:
        """Audio postprocessing."""
        if not self.config.enable_postprocessing:
            return audio
        
        processed = audio.copy()
        
        # Adjust length if necessary
        if len(processed) != original_length:
            if len(processed) > original_length:
                processed = processed[:original_length]
            else:
                # Pad with zeros
                processed = np.pad(processed, (0, original_length - len(processed)), 'constant')
            self.logger.debug(f"Length adjusted: {len(audio)} → {original_length}")
        
        # Smooth edges to avoid clicks
        fade_samples = min(256, len(processed) // 20)
        if fade_samples > 1:
            # Fade in
            fade_in = np.linspace(0, 1, fade_samples)
            processed[:fade_samples] *= fade_in
            
            # Fade out
            fade_out = np.linspace(1, 0, fade_samples)
            processed[-fade_samples:] *= fade_out
            
            self.logger.debug(f"Fade applied: {fade_samples} samples")
        
        # Final normalization
        max_val = np.max(np.abs(processed))
        if max_val > 0.99:
            processed = processed / max_val * 0.95
            self.logger.debug("Final normalization applied")
        
        return processed
    
    def process_with_monitoring(self, audio: np.ndarray, sr: int, 
                               semitones: float) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Process with metrics monitoring.
        
        Returns:
            Tuple[processed_audio, metrics]
        """
        start_time = time.time()
        original_length = len(audio)
        
        # Validation
        valid, error_msg = self.validate_inputs(audio, sr, semitones)
        if not valid:
            raise ValueError(f"Invalid input: {error_msg}")
        
        # Preprocessing
        preprocessed = self.preprocess_audio(audio)
        
        # Main processing
        try:
            processed = self.shift_pitch(preprocessed, sr, semitones)
        except Exception as e:
            raise RuntimeError(f"Error in pitch shifting: {e}")
        
        # Postprocessing
        final_audio = self.postprocess_audio(processed, original_length)
        
        # Calculate metrics
        processing_time = time.time() - start_time
        
        metrics = {
            'processing_time': processing_time,
            'original_rms': float(np.sqrt(np.mean(audio ** 2))),
            'processed_rms': float(np.sqrt(np.mean(final_audio ** 2))),
            'length_preserved': len(final_audio) == original_length,
            'max_amplitude': float(np.max(np.abs(final_audio)))
        }
        
        # Basic SNR
        if len(final_audio) == len(audio):
            noise = final_audio - audio
            signal_power = np.mean(audio ** 2)
            noise_power = np.mean(noise ** 2)
            
            if noise_power > 0 and signal_power > 0:
                metrics['snr'] = float(10 * np.log10(signal_power / noise_power))
            else:
                metrics['snr'] = float('inf')
        
        return final_audio, metrics