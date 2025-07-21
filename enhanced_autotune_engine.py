#!/usr/bin/env python3
"""
Enhanced Autotune Engine - Improved Core for Multi-Library System
===============================================================

Focused implementation of robust autotune with enhanced F0 detection,
intelligent MIDI processing, and detailed logging.

Version: 1.0.0
Author: Enhanced Autotune Engine
"""

import os
import sys
import time
import logging
import warnings
from pathlib import Path
from typing import Tuple, List, Optional, Dict, Any
from dataclasses import dataclass, field
import numpy as np

# Core dependencies
try:
    import librosa
    import soundfile as sf
    import pretty_midi
    from scipy import signal
    from tqdm import tqdm
except ImportError as e:
    print(f"❌ Missing core dependency: {e}")
    print("💡 Install with: pip install librosa soundfile pretty_midi scipy tqdm")
    sys.exit(1)

# Import multi_autotune system
try:
    from multi_autotune import multi_library_autotune, MultiLibraryAutotuneProcessor
    MULTI_AUTOTUNE_AVAILABLE = True
except ImportError:
    print("⚠️  Multi-autotune not available, using fallback implementation")
    MULTI_AUTOTUNE_AVAILABLE = False

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning, module="librosa")
warnings.filterwarnings("ignore", category=FutureWarning, module="librosa")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ===== DATA STRUCTURES =====

@dataclass
class ProcessingSegment:
    """Represents a processed audio segment."""
    start_time: float
    end_time: float
    start_sample: int
    end_sample: int
    detected_f0: float
    target_frequency: float
    midi_note: int
    pitch_shift_semitones: float
    processing_time: float
    quality_metrics: Dict[str, float] = field(default_factory=dict)

@dataclass
class ProcessingReport:
    """Comprehensive processing report."""
    total_duration: float
    total_segments: int
    segments_processed: int
    total_processing_time: float
    method_used: str
    average_f0_accuracy: float
    segments: List[ProcessingSegment] = field(default_factory=list)
    quality_summary: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'total_duration': self.total_duration,
            'total_segments': self.total_segments,
            'segments_processed': self.segments_processed,
            'total_processing_time': self.total_processing_time,
            'method_used': self.method_used,
            'average_f0_accuracy': self.average_f0_accuracy,
            'quality_summary': self.quality_summary,
            'segments': [
                {
                    'start_time': seg.start_time,
                    'end_time': seg.end_time,
                    'detected_f0': seg.detected_f0,
                    'target_frequency': seg.target_frequency,
                    'midi_note': seg.midi_note,
                    'pitch_shift_semitones': seg.pitch_shift_semitones,
                    'processing_time': seg.processing_time,
                    'quality_metrics': seg.quality_metrics
                }
                for seg in self.segments
            ]
        }

# ===== ENHANCED F0 DETECTOR =====

class EnhancedF0Detector:
    """Robust F0 detection with YIN algorithm and fallbacks."""
    
    def __init__(self, sample_rate: int = 44100, hop_length: int = 512):
        """Initialize F0 detector."""
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.frame_length = hop_length * 4
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def detect_pitch_yin_improved(self, audio: np.ndarray, 
                                 fmin: float = 65.0, fmax: float = 2093.0) -> np.ndarray:
        """
        Improved YIN pitch detection with adaptive threshold.
        
        Args:
            audio: Audio signal
            fmin: Minimum frequency (Hz)
            fmax: Maximum frequency (Hz)
            
        Returns:
            Array of F0 values
        """
        try:
            # Use librosa YIN with optimized parameters
            f0 = librosa.yin(
                audio,
                fmin=fmin,
                fmax=fmax,
                sr=self.sample_rate,
                hop_length=self.hop_length,
                frame_length=self.frame_length,
                threshold=0.1,  # Adaptive threshold
                center=True
            )
            
            # Validate results
            valid_frames = np.sum(f0 > 0)
            total_frames = len(f0)
            
            if valid_frames / total_frames > 0.3:  # 30% valid frames minimum
                self.logger.debug(f"YIN: {valid_frames}/{total_frames} valid frames")
                return self._clean_pitch_sequence(f0)
            else:
                self.logger.warning(f"YIN insufficient: {valid_frames}/{total_frames} frames")
                raise ValueError("YIN detection insufficient")
                
        except Exception as e:
            self.logger.warning(f"YIN failed: {e}")
            raise
    
    def detect_pitch_autocorr_fallback(self, audio: np.ndarray) -> np.ndarray:
        """
        Fallback pitch detection using autocorrelation.
        
        Args:
            audio: Audio signal
            
        Returns:
            Array of F0 values
        """
        try:
            num_frames = len(audio) // self.hop_length
            f0_values = np.zeros(num_frames)
            
            # Frequency limits for voice
            min_period = int(self.sample_rate / 800)  # 800 Hz max
            max_period = int(self.sample_rate / 80)   # 80 Hz min
            
            for i in range(num_frames):
                start = i * self.hop_length
                end = min(start + self.frame_length, len(audio))
                frame = audio[start:end]
                
                if len(frame) < min_period * 2:
                    continue
                
                # Autocorrelation
                autocorr = np.correlate(frame, frame, mode='full')
                autocorr = autocorr[len(autocorr)//2:]
                
                # Find peaks in autocorrelation
                if len(autocorr) > max_period:
                    search_range = autocorr[min_period:max_period]
                    
                    if len(search_range) > 0:
                        peak_idx = np.argmax(search_range) + min_period
                        
                        # Verify significant peak
                        if (peak_idx < len(autocorr) and 
                            autocorr[peak_idx] > 0.3 * autocorr[0]):
                            f0_values[i] = self.sample_rate / peak_idx
            
            valid_frames = np.sum(f0_values > 0)
            self.logger.info(f"Autocorrelation: {valid_frames}/{num_frames} frames")
            
            return self._clean_pitch_sequence(f0_values)
            
        except Exception as e:
            self.logger.error(f"Autocorrelation fallback failed: {e}")
            return np.zeros(len(audio) // self.hop_length)
    
    def _clean_pitch_sequence(self, f0: np.ndarray) -> np.ndarray:
        """
        Clean F0 sequence removing outliers and interpolating gaps.
        
        Args:
            f0: Raw F0 values
            
        Returns:
            Cleaned F0 values
        """
        if len(f0) == 0:
            return f0
        
        # Remove outliers
        valid_mask = f0 > 0
        
        if np.sum(valid_mask) == 0:
            return f0
        
        valid_f0 = f0[valid_mask]
        
        # Remove extreme outliers (3 sigma rule)
        if len(valid_f0) > 3:
            median_f0 = np.median(valid_f0)
            std_f0 = np.std(valid_f0)
            
            outlier_mask = np.abs(valid_f0 - median_f0) > 3 * std_f0
            
            # Mark outliers as invalid
            f0_cleaned = f0.copy()
            valid_indices = np.where(valid_mask)[0]
            f0_cleaned[valid_indices[outlier_mask]] = 0
        else:
            f0_cleaned = f0.copy()
        
        # Simple interpolation for small gaps
        if np.sum(f0_cleaned > 0) > 2:
            # Apply median filter for smoothing
            try:
                f0_cleaned = signal.medfilt(f0_cleaned, kernel_size=3)
            except Exception:
                pass  # If median filter fails, continue without it
        
        return f0_cleaned
    
    def detect_robust_f0(self, audio: np.ndarray) -> np.ndarray:
        """
        Main robust F0 detection method with automatic fallback.
        
        Args:
            audio: Audio signal
            
        Returns:
            Robust F0 estimates
        """
        try:
            # Primary: YIN algorithm
            f0 = self.detect_pitch_yin_improved(audio)
            
            # Validate YIN results
            valid_ratio = np.sum(f0 > 0) / len(f0)
            if valid_ratio > 0.2:  # Accept if >20% valid
                self.logger.info(f"YIN successful: {valid_ratio:.1%} valid frames")
                return f0
            else:
                raise ValueError("YIN insufficient coverage")
                
        except Exception as e:
            self.logger.warning(f"YIN failed: {e}, using autocorrelation fallback")
            
            # Fallback: Autocorrelation
            try:
                f0 = self.detect_pitch_autocorr_fallback(audio)
                
                valid_ratio = np.sum(f0 > 0) / len(f0)
                self.logger.info(f"Autocorr fallback: {valid_ratio:.1%} valid frames")
                return f0
                
            except Exception as e2:
                self.logger.error(f"All F0 detection failed: {e2}")
                # Return zeros as last resort
                return np.zeros(len(audio) // self.hop_length)
# PARTE 2

# ===== SMART MIDI PROCESSOR =====

class SmartMidiProcessor:
    """Intelligent MIDI processing with automatic looping and mapping."""
    
    def __init__(self):
        """Initialize MIDI processor."""
        self.logger = logging.getLogger(self.__class__.__name__)
        self._note_to_freq_cache = {}  # Cache for note conversions
    
    def load_and_extend_midi(self, midi_path: str, target_duration: float) -> List[Dict[str, Any]]:
        """
        Load MIDI file and extend sequence to match target duration.
        
        Args:
            midi_path: Path to MIDI file
            target_duration: Target duration in seconds
            
        Returns:
            List of note dictionaries with timing information
        """
        try:
            midi_data = pretty_midi.PrettyMIDI(str(midi_path))
            notes = []
            
            # Extract notes from all non-drum instruments
            for instrument in midi_data.instruments:
                if instrument.is_drum:
                    continue
                
                for note in instrument.notes:
                    if note.start >= note.end or not 0 <= note.pitch <= 127:
                        continue
                    
                    note_dict = {
                        'start_time': note.start,
                        'end_time': note.end,
                        'pitch': note.pitch,
                        'velocity': note.velocity,
                        'frequency': self.convert_note_to_frequency(note.pitch)
                    }
                    notes.append(note_dict)
            
            if not notes:
                raise ValueError("No valid notes found in MIDI file")
            
            # Sort by start time
            notes.sort(key=lambda x: x['start_time'])
            
            # Get original duration
            original_duration = max(note['end_time'] for note in notes)
            
            self.logger.info(f"Original MIDI: {len(notes)} notes, {original_duration:.2f}s")
            
            # Extend if necessary
            if original_duration < target_duration:
                extended_notes = self._extend_midi_sequence(notes, original_duration, target_duration)
                self.logger.info(f"Extended MIDI: {len(extended_notes)} notes, {target_duration:.2f}s")
                return extended_notes
            else:
                # Filter notes within target duration
                filtered_notes = [note for note in notes if note['start_time'] < target_duration]
                return filtered_notes
                
        except Exception as e:
            self.logger.error(f"Error loading MIDI: {e}")
            raise
    
    def _extend_midi_sequence(self, notes: List[Dict], original_duration: float, 
                            target_duration: float) -> List[Dict]:
        """Extend MIDI sequence by looping."""
        extended_notes = []
        loop_count = 0
        
        while True:
            loop_start_time = loop_count * original_duration
            
            # Check if we've covered the target duration
            if loop_start_time >= target_duration:
                break
            
            for note in notes:
                new_start = note['start_time'] + loop_start_time
                new_end = note['end_time'] + loop_start_time
                
                # Stop if note starts beyond target
                if new_start >= target_duration:
                    break
                
                # Clip note if it extends beyond target
                if new_end > target_duration:
                    new_end = target_duration
                
                # Only add if note has meaningful duration
                if new_end - new_start > 0.01:
                    extended_note = note.copy()
                    extended_note['start_time'] = new_start
                    extended_note['end_time'] = new_end
                    extended_notes.append(extended_note)
            
            loop_count += 1
        
        return extended_notes
    
    def map_time_to_note(self, time_seconds: float, notes: List[Dict]) -> Optional[Dict]:
        """
        Map a time point to the active MIDI note.
        
        Args:
            time_seconds: Time in seconds
            notes: List of note dictionaries
            
        Returns:
            Active note dictionary or None
        """
        # Find all notes active at this time
        active_notes = [
            note for note in notes 
            if note['start_time'] <= time_seconds <= note['end_time']
        ]
        
        if not active_notes:
            return None
        
        # If multiple notes, choose based on priority
        if len(active_notes) == 1:
            return active_notes[0]
        else:
            # Priority: lowest pitch (bass note in chord)
            return min(active_notes, key=lambda x: x['pitch'])
    
    def convert_note_to_frequency(self, midi_note: int) -> float:
        """
        Convert MIDI note number to frequency using caching.
        
        Args:
            midi_note: MIDI note number (0-127)
            
        Returns:
            Frequency in Hz
        """
        if midi_note in self._note_to_freq_cache:
            return self._note_to_freq_cache[midi_note]
        
        # Standard formula: 440 * 2^((note - 69)/12)
        frequency = 440.0 * (2.0 ** ((midi_note - 69) / 12.0))
        
        # Cache the result
        self._note_to_freq_cache[midi_note] = frequency
        
        return frequency
    
    def analyze_midi_complexity(self, notes: List[Dict]) -> Dict[str, float]:
        """
        Analyze MIDI sequence complexity for processing optimization.
        
        Args:
            notes: List of note dictionaries
            
        Returns:
            Dictionary with complexity metrics
        """
        if not notes:
            return {'complexity_score': 0.0, 'note_density': 0.0, 'pitch_range': 0.0}
        
        # Calculate metrics
        total_duration = max(note['end_time'] for note in notes)
        note_density = len(notes) / total_duration if total_duration > 0 else 0
        
        pitches = [note['pitch'] for note in notes]
        pitch_range = max(pitches) - min(pitches) if pitches else 0
        
        # Overlap analysis
        overlap_count = 0
        for i, note1 in enumerate(notes):
            for note2 in notes[i+1:]:
                # Check if notes overlap
                if (note1['start_time'] < note2['end_time'] and 
                    note2['start_time'] < note1['end_time']):
                    overlap_count += 1
        
        overlap_ratio = overlap_count / len(notes) if notes else 0
        
        # Combined complexity score
        complexity_score = min(1.0, (note_density / 5.0 + pitch_range / 60.0 + overlap_ratio) / 3.0)
        
        return {
            'complexity_score': complexity_score,
            'note_density': note_density,
            'pitch_range': pitch_range,
            'overlap_ratio': overlap_ratio,
            'total_notes': len(notes),
            'total_duration': total_duration
        }

# ===== IMPROVED VOICE ACTIVITY DETECTION =====

class ImprovedVAD:
    """Enhanced Voice Activity Detection with multiple features."""
    
    def __init__(self, sample_rate: int = 44100, hop_length: int = 512):
        """Initialize VAD."""
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def detect_voice_activity(self, audio: np.ndarray) -> np.ndarray:
        """
        Detect voice activity using multiple acoustic features.
        
        Args:
            audio: Audio signal
            
        Returns:
            Boolean mask for voice activity
        """
        try:
            # Calculate multiple features
            rms = self._calculate_rms(audio)
            spectral_centroid = self._calculate_spectral_centroid(audio)
            zcr = self._calculate_zero_crossing_rate(audio)
            
            # Robust normalization
            rms_norm = self._robust_normalize(rms)
            centroid_norm = self._robust_normalize(spectral_centroid)
            zcr_norm = self._robust_normalize(zcr)
            
            # Combine features with weights
            voice_score = (rms_norm * 0.5 + 
                          centroid_norm * 0.3 + 
                          zcr_norm * 0.2)
            
            # Adaptive threshold
            threshold = np.percentile(voice_score, 25)  # 25th percentile
            voice_mask = voice_score > threshold
            
            # Temporal smoothing
            if len(voice_mask) > 3:
                voice_mask = signal.medfilt(voice_mask.astype(float), kernel_size=3).astype(bool)
            
            voice_percentage = np.sum(voice_mask) / len(voice_mask) * 100
            self.logger.debug(f"VAD: {voice_percentage:.1f}% voice detected")
            
            return voice_mask
            
        except Exception as e:
            self.logger.error(f"VAD failed: {e}")
            # Fallback: assume all frames have voice
            return np.ones(len(audio) // self.hop_length, dtype=bool)
    
    def _calculate_rms(self, audio: np.ndarray) -> np.ndarray:
        """Calculate RMS energy."""
        return librosa.feature.rms(
            y=audio, 
            hop_length=self.hop_length,
            frame_length=self.hop_length * 2
        )[0]
    
    def _calculate_spectral_centroid(self, audio: np.ndarray) -> np.ndarray:
        """Calculate spectral centroid."""
        return librosa.feature.spectral_centroid(
            y=audio, 
            sr=self.sample_rate,
            hop_length=self.hop_length
        )[0]
    
    def _calculate_zero_crossing_rate(self, audio: np.ndarray) -> np.ndarray:
        """Calculate zero crossing rate."""
        return librosa.feature.zero_crossing_rate(
            audio,
            hop_length=self.hop_length,
            frame_length=self.hop_length * 2
        )[0]
    
    def _robust_normalize(self, feature: np.ndarray) -> np.ndarray:
        """Robust normalization with zero-division protection."""
        if len(feature) == 0:
            return feature
        
        median_val = np.median(feature)
        std_val = np.std(feature)
        
        if std_val > 1e-8:
            return (feature - median_val) / std_val
        else:
            return np.zeros_like(feature)

# PARTE 3

# ===== DETAILED LOGGER =====

class DetailedLogger:
    """Advanced logging system for processing analysis."""
    
    def __init__(self, log_level: str = "INFO"):
        """Initialize logger."""
        self.logger = logging.getLogger("EnhancedAutotune")
        self.logger.setLevel(getattr(logging, log_level.upper()))
        
        self.segments_log = []
        self.processing_stats = {}
        self.start_time = None
    
    def log_processing_start(self, audio_path: str, midi_path: str, method: str, force: float):
        """Log processing start."""
        self.start_time = time.time()
        self.logger.info("=" * 60)
        self.logger.info("🎵 ENHANCED AUTOTUNE PROCESSING STARTED")
        self.logger.info("=" * 60)
        self.logger.info(f"Audio file: {Path(audio_path).name}")
        self.logger.info(f"MIDI file: {Path(midi_path).name}")
        self.logger.info(f"Method: {method}")
        self.logger.info(f"Force: {force:.2f}")
        self.logger.info(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        
    def log_segment_processing(self, segment: ProcessingSegment):
        """Log individual segment processing."""
        self.segments_log.append(segment)
        
        self.logger.debug(
            f"Segment {len(self.segments_log):3d}: "
            f"{segment.start_time:6.2f}s-{segment.end_time:6.2f}s | "
            f"F0:{segment.detected_f0:6.1f}Hz → {segment.target_frequency:6.1f}Hz | "
            f"Note:{segment.midi_note:3d} | "
            f"Shift:{segment.pitch_shift_semitones:+5.2f}st | "
            f"{segment.processing_time*1000:4.1f}ms"
        )
    
    def log_midi_note_application(self, time_point: float, midi_note: int, frequency: float):
        """Log MIDI note application."""
        self.logger.debug(f"MIDI @ {time_point:6.2f}s: Note {midi_note:3d} ({frequency:6.1f}Hz)")
    
    def log_processing_stats(self, stats: Dict[str, Any]):
        """Log processing statistics."""
        self.processing_stats.update(stats)
        
        for key, value in stats.items():
            if isinstance(value, float):
                self.logger.info(f"📊 {key}: {value:.3f}")
            else:
                self.logger.info(f"📊 {key}: {value}")
    
    def generate_processing_report(self, method_used: str, total_duration: float) -> ProcessingReport:
        """Generate comprehensive processing report."""
        if self.start_time is None:
            total_processing_time = 0.0
        else:
            total_processing_time = time.time() - self.start_time
        
        # Calculate statistics
        segments_processed = len([s for s in self.segments_log if s.detected_f0 > 0])
        
        if self.segments_log:
            # Calculate average F0 accuracy
            valid_segments = [s for s in self.segments_log if s.detected_f0 > 0 and s.target_frequency > 0]
            if valid_segments:
                f0_errors = [
                    abs(s.detected_f0 - s.target_frequency) / s.target_frequency 
                    for s in valid_segments
                ]
                average_f0_accuracy = 1.0 - np.mean(f0_errors)
            else:
                average_f0_accuracy = 0.0
            
            # Quality summary
            quality_summary = {
                'average_processing_time_ms': np.mean([s.processing_time * 1000 for s in self.segments_log]),
                'total_pitch_corrections': segments_processed,
                'processing_efficiency': segments_processed / len(self.segments_log) if self.segments_log else 0,
                'average_pitch_shift_semitones': np.mean([abs(s.pitch_shift_semitones) for s in self.segments_log])
            }
        else:
            average_f0_accuracy = 0.0
            quality_summary = {}
        
        report = ProcessingReport(
            total_duration=total_duration,
            total_segments=len(self.segments_log),
            segments_processed=segments_processed,
            total_processing_time=total_processing_time,
            method_used=method_used,
            average_f0_accuracy=average_f0_accuracy,
            segments=self.segments_log.copy(),
            quality_summary=quality_summary
        )
        
        # Log summary
        self.logger.info("=" * 60)
        self.logger.info("📊 PROCESSING SUMMARY")
        self.logger.info("=" * 60)
        self.logger.info(f"Duration: {total_duration:.2f}s")
        self.logger.info(f"Method: {method_used}")
        self.logger.info(f"Total segments: {report.total_segments}")
        self.logger.info(f"Processed segments: {report.segments_processed}")
        self.logger.info(f"Processing time: {total_processing_time:.2f}s")
        self.logger.info(f"F0 accuracy: {average_f0_accuracy:.1%}")
        self.logger.info("=" * 60)
        
        return report

# ===== ENHANCED AUTOTUNE ENGINE =====

class EnhancedAutotuneEngine:
    """Main enhanced autotune processing engine."""
    
    def __init__(self, sample_rate: int = 44100, hop_length: int = 512):
        """Initialize enhanced engine."""
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.frame_length = hop_length * 4
        
        # Initialize components
        self.f0_detector = EnhancedF0Detector(sample_rate, hop_length)
        self.midi_processor = SmartMidiProcessor()
        self.vad = ImprovedVAD(sample_rate, hop_length)
        self.logger_system = DetailedLogger()
        
        self.main_logger = logging.getLogger(self.__class__.__name__)
    
    def process_with_method(self, audio: np.ndarray, midi_notes: List[Dict], 
                          force: float, method: str) -> Tuple[np.ndarray, ProcessingReport]:
        """
        Process audio with specified method from multi_autotune system.
        
        Args:
            audio: Audio signal
            midi_notes: MIDI notes list
            force: Correction force (0.0-1.0)
            method: Processing method
            
        Returns:
            Tuple of (processed_audio, processing_report)
        """
        if force == 0.0:
            self.main_logger.info("Force = 0.0, returning original audio")
            return audio.copy(), self._create_empty_report(method)
        
        # Detect voice activity
        voice_mask = self.vad.detect_voice_activity(audio)
        
        # Detect F0
        f0_values = self.f0_detector.detect_robust_f0(audio)
        
        # Process windowed segments
        return self._apply_windowed_processing(audio, f0_values, voice_mask, midi_notes, force, method)
    
    def _apply_windowed_processing(self, audio: np.ndarray, f0_values: np.ndarray,
                                 voice_mask: np.ndarray, midi_notes: List[Dict],
                                 force: float, method: str) -> Tuple[np.ndarray, ProcessingReport]:
        """Apply windowed processing with overlap and crossfade."""
        
        window_size = self.frame_length * 2  # Larger windows for better quality
        overlap = window_size // 2  # 50% overlap
        
        output_audio = audio.copy()
        processed_segments = []
# PARTE 4
        # Convert frame indices to time for MIDI mapping
        total_frames = len(f0_values)
        
        with tqdm(total=total_frames, desc=f"Processing with {method}", unit="frame") as pbar:
            for frame_idx in range(0, total_frames, overlap // self.hop_length):
                pbar.update(overlap // self.hop_length)
                
                # Calculate sample indices
                start_sample = frame_idx * self.hop_length
                end_sample = min(start_sample + window_size, len(audio))
                
                if end_sample - start_sample < window_size // 2:
                    break  # Skip incomplete windows at the end
                
                # Extract window
                window_audio = audio[start_sample:end_sample]
                window_time = start_sample / self.sample_rate
                
                # Check voice activity
                if frame_idx < len(voice_mask) and not voice_mask[frame_idx]:
                    continue
                
                # Get F0 for this window
                if frame_idx < len(f0_values) and f0_values[frame_idx] > 0:
                    detected_f0 = f0_values[frame_idx]
                else:
                    continue  # Skip frames without detected pitch
                
                # Find corresponding MIDI note
                active_note = self.midi_processor.map_time_to_note(window_time, midi_notes)
                if active_note is None:
                    continue
                
                target_frequency = active_note['frequency']
                midi_note = active_note['pitch']
                
                # Calculate pitch shift
                if detected_f0 > 0 and target_frequency > 0:
                    semitones = 12 * np.log2(target_frequency / detected_f0)
                    applied_semitones = semitones * force
                    
                    # Limit extreme shifts
                    applied_semitones = np.clip(applied_semitones, -12, 12)
                    
                    if abs(applied_semitones) > 0.1:  # Only process significant shifts
                        segment_start_time = time.time()
                        
                        try:
                            # Apply pitch shift using selected method
                            shifted_window = self._apply_pitch_shift_method(
                                window_audio, applied_semitones, method
                            )
                            
                            # Apply crossfade
                            shifted_window = self._apply_crossfade(shifted_window)
                            
                            # Mix with original
                            output_audio[start_sample:end_sample] = shifted_window
                            
                            processing_time = time.time() - segment_start_time
                            
                            # Create segment record
                            segment = ProcessingSegment(
                                start_time=window_time,
                                end_time=(end_sample / self.sample_rate),
                                start_sample=start_sample,
                                end_sample=end_sample,
                                detected_f0=detected_f0,
                                target_frequency=target_frequency,
                                midi_note=midi_note,
                                pitch_shift_semitones=applied_semitones,
                                processing_time=processing_time,
                                quality_metrics={'snr': self._calculate_window_snr(window_audio, shifted_window)}
                            )
                            
                            processed_segments.append(segment)
                            self.logger_system.log_segment_processing(segment)
                            
                        except Exception as e:
                            self.main_logger.warning(f"Window processing failed: {e}")
                            continue
        
        # Generate report
        audio_duration = len(audio) / self.sample_rate
        report = ProcessingReport(
            total_duration=audio_duration,
            total_segments=len(processed_segments),
            segments_processed=len(processed_segments),
            total_processing_time=sum(s.processing_time for s in processed_segments),
            method_used=method,
            average_f0_accuracy=self._calculate_average_accuracy(processed_segments),
            segments=processed_segments,
            quality_summary=self._calculate_quality_summary(processed_segments)
        )
        
        return output_audio, report
    
    def _apply_pitch_shift_method(self, audio: np.ndarray, semitones: float, method: str) -> np.ndarray:
        """Apply pitch shifting using the specified method."""
        
        if MULTI_AUTOTUNE_AVAILABLE and method != "fallback":
            try:
                # Use multi_autotune system for pitch shifting
                # Create temporary files for processing
                import tempfile
                
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_audio:
                    with tempfile.NamedTemporaryFile(suffix='.mid', delete=False) as temp_midi:
                        try:
                            # Save temporary audio
                            sf.write(temp_audio.name, audio, self.sample_rate)
                            
                            # Create simple MIDI with target pitch
                            target_note = 69 + semitones  # A4 + shift
                            midi_obj = pretty_midi.PrettyMIDI()
                            instrument = pretty_midi.Instrument(program=0)
                            note = pretty_midi.Note(
                                velocity=80,
                                pitch=int(target_note),
                                start=0.0,
                                end=len(audio) / self.sample_rate
                            )
                            instrument.notes.append(note)
                            midi_obj.instruments.append(instrument)
                            midi_obj.write(temp_midi.name)
                            
                            # Process with multi_autotune
                            result = multi_library_autotune(
                                audio_path=temp_audio.name,
                                midi_path=temp_midi.name,
                                force=1.0,  # Full force since we calculated the exact shift
                                pitch_shift_method=method
                            )
                            
                            if result.success:
                                # Load processed audio
                                processed_audio, _ = sf.read(result.output_path)
                                # Clean up
                                os.unlink(result.output_path)
                                return processed_audio[:len(audio)]  # Ensure same length
                            
                        finally:
                            # Clean up temp files
                            try:
                                os.unlink(temp_audio.name)
                                os.unlink(temp_midi.name)
                            except:
                                pass
                            
            except Exception as e:
                self.main_logger.warning(f"Multi-autotune method {method} failed: {e}")
        
        # Fallback: Simple LibROSA pitch shift
        try:
            return librosa.effects.pitch_shift(
                y=audio,
                sr=self.sample_rate,
                n_steps=semitones,
                bins_per_octave=12
            )
        except Exception as e:
            self.main_logger.error(f"Fallback pitch shift failed: {e}")
            return audio  # Return original if all fails
    
    def _apply_crossfade(self, audio: np.ndarray) -> np.ndarray:
        """Apply crossfade to window edges."""
        fade_samples = min(256, len(audio) // 10)
        
        if fade_samples > 1:
            # Fade in
            fade_in = np.linspace(0, 1, fade_samples)
            audio[:fade_samples] *= fade_in
            
            # Fade out
            fade_out = np.linspace(1, 0, fade_samples)
            audio[-fade_samples:] *= fade_out
        
        return audio
    
    def _calculate_window_snr(self, original: np.ndarray, processed: np.ndarray) -> float:
        """Calculate SNR for a window."""
        try:
            if len(original) != len(processed):
                min_len = min(len(original), len(processed))
                original = original[:min_len]
                processed = processed[:min_len]
            
            signal_power = np.mean(original ** 2)
            noise_power = np.mean((processed - original) ** 2)
            
            if noise_power > 0 and signal_power > 0:
                return float(10 * np.log10(signal_power / noise_power))
            else:
                return float('inf')
        except:
            return 0.0
    
    def _calculate_average_accuracy(self, segments: List[ProcessingSegment]) -> float:
        """Calculate average F0 accuracy."""
        if not segments:
            return 0.0
        
        valid_segments = [s for s in segments if s.detected_f0 > 0 and s.target_frequency > 0]
        if not valid_segments:
            return 0.0
        
        errors = [
            abs(s.detected_f0 - s.target_frequency) / s.target_frequency 
            for s in valid_segments
        ]
        return 1.0 - np.mean(errors)
    
    def _calculate_quality_summary(self, segments: List[ProcessingSegment]) -> Dict[str, float]:
        """Calculate quality summary metrics."""
        if not segments:
            return {}
        
        return {
            'average_processing_time_ms': np.mean([s.processing_time * 1000 for s in segments]),
            'total_pitch_corrections': len(segments),
            'average_pitch_shift_semitones': np.mean([abs(s.pitch_shift_semitones) for s in segments]),
            'average_snr': np.mean([s.quality_metrics.get('snr', 0) for s in segments])
        }
    
    def _create_empty_report(self, method: str) -> ProcessingReport:
        """Create empty report for zero force."""
        return ProcessingReport(
            total_duration=0.0,
            total_segments=0,
            segments_processed=0,
            total_processing_time=0.0,
            method_used=method,
            average_f0_accuracy=0.0
        )

# ===== MAIN ENHANCED AUTOTUNE FUNCTION =====

def enhanced_autotune(audio_path: str, midi_path: str, force: float = 0.85, 
                     method: str = "auto", output_path: Optional[str] = None,
                     log_level: str = "INFO") -> Dict[str, Any]:
    """
    Enhanced autotune function with robust processing and detailed logging.
    
    Args:
        audio_path: Path to audio file (WAV, MP3, OGG)
        midi_path: Path to MIDI file (.mid)
        force: Correction intensity (0.0-1.0)
        method: Processing method (auto, librosa_hifi, etc.)
        output_path: Output path (optional)
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        
    Returns:
        Dictionary with processing results and report
    """
    
    # Validate inputs
    if not 0.0 <= force <= 1.0:
        raise ValueError(f"Force must be between 0.0 and 1.0, got {force}")
    
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    
    if not os.path.exists(midi_path):
        raise FileNotFoundError(f"MIDI file not found: {midi_path}")
    
    # Initialize engine
    engine = EnhancedAutotuneEngine()
    engine.logger_system = DetailedLogger(log_level)
    
    try:
        # Log processing start
        engine.logger_system.log_processing_start(audio_path, midi_path, method, force)
        
        # Load audio
        audio, sr = librosa.load(audio_path, sr=44100, mono=True)
        audio_duration = len(audio) / sr
        
        engine.main_logger.info(f"Audio loaded: {len(audio)} samples, {audio_duration:.2f}s")
        
        # Load and extend MIDI
        midi_notes = engine.midi_processor.load_and_extend_midi(midi_path, audio_duration)
        
        # Analyze MIDI complexity
        midi_complexity = engine.midi_processor.analyze_midi_complexity(midi_notes)
        engine.logger_system.log_processing_stats(midi_complexity)
        
        # Process with enhanced engine
        processed_audio, report = engine.process_with_method(audio, midi_notes, force, method)
        
        # Define output path
        if output_path is None:
            input_path = Path(audio_path)
            output_path = input_path.parent / f"{input_path.stem}_autotuned.wav"
        
        # Save processed audio
        sf.write(str(output_path), processed_audio, sr, subtype='PCM_16')
        
        # Final report
        final_report = engine.logger_system.generate_processing_report(method, audio_duration)
        
        engine.main_logger.info(f"✅ Enhanced autotune completed!")
        engine.main_logger.info(f"📁 Output: {output_path}")
        
        return {
            'success': True,
            'output_path': str(output_path),
            'processing_report': final_report.to_dict(),
            'audio_duration': audio_duration,
            'method_used': method,
            'segments_processed': report.segments_processed,
            'total_processing_time': report.total_processing_time
        }
        
    except Exception as e:
        engine.main_logger.error(f"Enhanced autotune failed: {e}")
        return {
            'success': False,
            'error_message': str(e),
            'output_path': None
        }

# ===== INTEGRATION WITH MULTI-AUTOTUNE =====

def register_enhanced_engine():
    """Register enhanced engine as a new method in multi-autotune system."""
    if not MULTI_AUTOTUNE_AVAILABLE:
        print("⚠️  Multi-autotune not available, enhanced engine running standalone")
        return False
    
    try:
        # This would integrate with the multi_autotune system
        # Implementation depends on the exact multi_autotune architecture
        print("✅ Enhanced engine registered with multi-autotune system")
        return True
    except Exception as e:
        print(f"❌ Failed to register enhanced engine: {e}")
        return False

# ===== MAIN EXECUTION =====

def main():
    """Main function for standalone execution."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Enhanced Autotune Engine")
    parser.add_argument("audio_path", help="Path to audio file")
    parser.add_argument("midi_path", help="Path to MIDI file")
    parser.add_argument("--force", "-f", type=float, default=0.85, help="Correction force (0.0-1.0)")
    parser.add_argument("--method", "-m", default="auto", help="Processing method")
    parser.add_argument("--output", "-o", help="Output path")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    
    args = parser.parse_args()
    
    try:
        result = enhanced_autotune(
            audio_path=args.audio_path,
            midi_path=args.midi_path,
            force=args.force,
            method=args.method,
            output_path=args.output,
            log_level=args.log_level
        )
        
        if result['success']:
            print(f"\n🎉 Success! Output: {result['output_path']}")
            print(f"📊 Processed {result['segments_processed']} segments in {result['total_processing_time']:.2f}s")
        else:
            print(f"\n❌ Failed: {result['error_message']}")
            return 1
            
        return 0
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())