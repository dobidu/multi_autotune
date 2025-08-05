#!/usr/bin/env python3
"""
Multi-Library Autotune System
=============================

Advanced autotune system that integrates multiple pitch shifting libraries,
allowing dynamic algorithm selection based on automatic analysis or user preference.

Integrates 9 different methods:
- pydub_speed: PyDub (fast, low quality)
- pyrubberband_shift: Rubberband (high quality)
- librosa_standard: LibROSA standard (balanced)
- librosa_hifi: LibROSA high quality
- pedalboard_shift: Spotify Pedalboard (professional)
- scipy_manual: SciPy manual (educational/research)
- scipy_autotune: SciPy with complete autotune
- soundtouch_shifter: SoundTouch (high quality)
- vst_plugin_shifter: VST Plugin (professional)

Version: 2.0.0
Author: Multi-Library Autotune System
"""

import os
import sys
import time
import json
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple, Any
from dataclasses import dataclass, field, asdict
from enum import Enum
import numpy as np

# Suppress unnecessary warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ===== ENUMS AND CONSTANTS =====

class ProcessingProfile(Enum):
    """Predefined processing profiles."""
    REALTIME = "realtime"
    PRODUCTION = "production"
    BROADCAST = "broadcast"
    RESEARCH = "research"
    CREATIVE = "creative"

class QualityLevel(Enum):
    """Quality levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    ULTRA = "ultra"

# Available methods categorized
METHOD_CATEGORIES = {
    "speed_optimized": ["pydub_speed", "scipy_manual"],
    "quality_optimized": ["pyrubberband_shift", "librosa_hifi", "pedalboard_shift"],
    "balanced": ["librosa_standard"],
    "autotune_complete": ["scipy_autotune"],
    "external_dependency": ["pyrubberband_shift", "pedalboard_shift"],
    "pure_python": ["pydub_speed", "scipy_manual", "scipy_autotune"],
    "professional_grade": ["pyrubberband_shift", "pedalboard_shift", "librosa_hifi"]
}

VALID_METHODS = [
    "auto", "pydub_speed", "pyrubberband_shift", 
    "librosa_standard", "librosa_hifi", "pedalboard_shift",
    "scipy_manual", "scipy_autotune", "soundtouch_shift", "vst_plugin_shift"
]

# Fallback chains optimized by similarity
FALLBACK_CHAINS = {
    "pyrubberband_shift": ["soundtouch_shift", "pedalboard_shift", "librosa_hifi"],
    "soundtouch_shift": ["pyrubberband_shift", "pedalboard_shift", "librosa_hifi"],
    "pedalboard_shift": ["pyrubberband_shift", "soundtouch_shift", "librosa_hifi"],
    "librosa_hifi": ["librosa_standard", "pedalboard_shift", "scipy_autotune"],
    "librosa_standard": ["librosa_hifi", "scipy_autotune", "scipy_manual"],
    "scipy_autotune": ["librosa_standard", "scipy_manual", "pydub_speed"],
    "scipy_manual": ["pydub_speed", "librosa_standard", "scipy_autotune"],
    "pydub_speed": ["scipy_manual", "librosa_standard", "scipy_autotune"]
}

# ===== CONFIGURATION DATACLASSES =====

@dataclass
class AutotuneConfig:
    """Main configuration for the autotune system."""
    sample_rate: int = 44100
    hop_length: int = 256
    frame_length: int = 1024
    quality_priority: float = 0.7  # 0.0=speed, 1.0=quality
    enable_preprocessing: bool = True
    enable_postprocessing: bool = True
    
    # Processing settings
    voice_activity_threshold: float = 0.3
    pitch_smoothing: bool = True
    preserve_formants: bool = True
    
    # Fallback settings
    fallback_enabled: bool = True
    max_fallback_attempts: int = 3
    
    # Quality settings
    quality_threshold: float = 0.8
    snr_threshold: float = 20.0
    
    # Performance settings
    processing_timeout: float = 30.0
    memory_limit_mb: float = 1024.0
    
    def validate(self) -> Tuple[bool, List[str]]:
        """Validates configuration."""
        errors = []
        
        if not 8000 <= self.sample_rate <= 192000:
            errors.append(f"Invalid sample rate: {self.sample_rate}")
        
        if not 0.0 <= self.quality_priority <= 1.0:
            errors.append(f"Quality priority must be between 0.0 and 1.0: {self.quality_priority}")
        
        if self.hop_length <= 0 or self.frame_length <= 0:
            errors.append("hop_length and frame_length must be positive")
        
        if self.frame_length < self.hop_length:
            errors.append("frame_length must be >= hop_length")
        
        return len(errors) == 0, errors

@dataclass
class MethodInfo:
    """Information about a pitch shifting method."""
    name: str
    library_required: str
    available: bool
    quality_score: float = 0.0  # 0.0-1.0
    speed_score: float = 0.0    # 0.0-1.0
    memory_efficiency: float = 0.0  # 0.0-1.0
    best_use_cases: List[str] = field(default_factory=list)
    limitations: List[str] = field(default_factory=list)
    requires_external: bool = False
    installation_notes: str = ""
    
    def get_overall_score(self, quality_weight: float = 0.5) -> float:
        """Calculates overall score balancing quality and speed."""
        speed_weight = 1.0 - quality_weight
        return (self.quality_score * quality_weight + 
                self.speed_score * speed_weight) * (1.0 if self.available else 0.0)

@dataclass
class AudioCharacteristics:
    """Characteristics of the analyzed audio."""
    duration: float = 0.0
    complexity_score: float = 0.0  # 0-1, based on spectral analysis
    noise_level: float = 0.0
    dynamic_range: float = 0.0
    harmonic_content: float = 0.0
    spectral_centroid: float = 0.0
    
    @classmethod
    def analyze_audio(cls, audio: np.ndarray, sr: int) -> 'AudioCharacteristics':
        """Analyzes audio characteristics."""
        duration = len(audio) / sr
        
        # RMS for noise level
        rms = np.sqrt(np.mean(audio ** 2))
        noise_level = 1.0 - min(1.0, rms * 10)  # Approximation
        
        # Dynamic range
        peak = np.max(np.abs(audio))
        dynamic_range = 20 * np.log10(peak / (rms + 1e-10))
        
        # Spectral complexity (approximation using spectral variance)
        try:
            fft = np.fft.fft(audio)
            magnitude = np.abs(fft)
            freqs = np.fft.fftfreq(len(audio), 1/sr)
            
            # Spectral centroid
            positive_freqs = freqs[:len(freqs)//2]
            positive_magnitude = magnitude[:len(magnitude)//2]
            spectral_centroid = np.sum(positive_freqs * positive_magnitude) / np.sum(positive_magnitude)
            
            # Complexity based on spectral distribution
            spectral_variance = np.var(positive_magnitude)
            complexity_score = min(1.0, spectral_variance / (np.mean(positive_magnitude) + 1e-10))
            
            # Harmonic content (approximation)
            harmonic_content = min(1.0, np.sum(positive_magnitude[:len(positive_magnitude)//4]) / 
                                 np.sum(positive_magnitude))
            
        except Exception:
            spectral_centroid = 1000.0
            complexity_score = 0.5
            harmonic_content = 0.5
        
        return cls(
            duration=duration,
            complexity_score=complexity_score,
            noise_level=noise_level,
            dynamic_range=dynamic_range,
            harmonic_content=harmonic_content,
            spectral_centroid=spectral_centroid
        )

@dataclass
class UserRequirements:
    """User requirements."""
    quality_priority: float = 0.7  # 0-1
    speed_priority: float = 0.3    # 0-1
    use_case: str = "general"      # "realtime", "production", etc.
    max_processing_time: float = 30.0
    min_quality_threshold: float = 0.8
    
    def __post_init__(self):
        """Normalizes priorities."""
        total = self.quality_priority + self.speed_priority
        if total > 0:
            self.quality_priority /= total
            self.speed_priority /= total

@dataclass
class SystemConstraints:
    """System constraints."""
    available_methods: List[str] = field(default_factory=list)
    cpu_cores: int = 1
    memory_limit: float = 1024.0  # MB
    processing_timeout: float = 30.0
    allow_external_dependencies: bool = True
    
    @classmethod
    def detect_system_constraints(cls) -> 'SystemConstraints':
        """Automatically detects system constraints."""
        try:
            import psutil
            cpu_cores = psutil.cpu_count()
            memory_gb = psutil.virtual_memory().total / (1024**3)
            memory_limit = min(8192, memory_gb * 1024 * 0.8)  # 80% of available RAM
        except ImportError:
            cpu_cores = 1
            memory_limit = 1024.0
        
        return cls(
            cpu_cores=cpu_cores,
            memory_limit=memory_limit
        )

@dataclass
class BenchmarkResult:
    """Benchmark result for a method."""
    method_name: str
    avg_processing_time: float = 0.0
    avg_quality_score: float = 0.0
    memory_usage: float = 0.0
    success_rate: float = 0.0
    recommended_for: List[str] = field(default_factory=list)
    test_cases_passed: int = 0
    total_test_cases: int = 0
    
    @property
    def efficiency_score(self) -> float:
        """Efficiency score combining time and quality."""
        if self.avg_processing_time <= 0:
            return 0.0
        
        # Normalizes time (assumes maximum of 10s as reference)
        time_score = max(0, 1.0 - self.avg_processing_time / 10.0)
        
        # Combines with quality and success rate
        return (time_score * 0.4 + self.avg_quality_score * 0.4 + 
                self.success_rate * 0.2)

@dataclass
class AutotuneResult:
    """Result of autotune processing."""
    success: bool
    output_path: Optional[str] = None
    method_used: Optional[str] = None
    processing_time: float = 0.0
    quality_metrics: Dict[str, float] = field(default_factory=dict)
    fallback_methods_tried: List[str] = field(default_factory=list)
    error_message: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    
    # Detailed metrics
    original_audio_stats: Dict[str, float] = field(default_factory=dict)
    processed_audio_stats: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Converts to dictionary."""
        return asdict(self)

# PARTE 2

# ===== LIBRARY DETECTION =====

class LibraryDetector:
    """Detects available libraries and their capabilities."""
    
    @staticmethod
    def detect_all_libraries() -> Dict[str, MethodInfo]:
        """Detects all available libraries."""
        methods = {}
        
        # PyDub
        methods["pydub_speed"] = LibraryDetector._detect_pydub()
        
        # PyRubberband
        methods["pyrubberband_shift"] = LibraryDetector._detect_pyrubberband()
        
        # LibROSA
        librosa_info = LibraryDetector._detect_librosa()
        methods["librosa_standard"] = librosa_info
        
        # LibROSA Hi-Fi (same library, different configuration)
        librosa_hifi = MethodInfo(
            name="LibROSA Hi-Fi",
            library_required="librosa",
            available=librosa_info.available,
            quality_score=0.95,
            speed_score=0.6,
            memory_efficiency=0.7,
            best_use_cases=["high_quality", "mastering", "production"],
            limitations=["slower_processing", "higher_memory"]
        )
        methods["librosa_hifi"] = librosa_hifi
        
        # Pedalboard
        methods["pedalboard_shift"] = LibraryDetector._detect_pedalboard()
        
        # SciPy Manual
        methods["scipy_manual"] = LibraryDetector._detect_scipy_manual()
        
        # SciPy Autotune
        methods["scipy_autotune"] = LibraryDetector._detect_scipy_autotune()

        # SoundTouch
        methods["soundtouch_shift"] = LibraryDetector._detect_soundtouch()
        
        # VST Plugin
        methods["vst_plugin_shift"] = LibraryDetector._detect_vst_support()

        return methods
    
    @staticmethod
    def _detect_pydub() -> MethodInfo:
        """Detects PyDub."""
        try:
            import pydub
            return MethodInfo(
                name="PyDub Speed",
                library_required="pydub",
                available=True,
                quality_score=0.4,
                speed_score=1.0,
                memory_efficiency=0.9,
                best_use_cases=["realtime", "prototyping", "creative_effects"],
                limitations=["changes_duration", "low_quality", "artifacts"],
                installation_notes="pip install pydub"
            )
        except ImportError as e:
            return MethodInfo(
                name="PyDub Speed",
                library_required="pydub",
                available=False,
                installation_notes=f"pip install pydub (Error: {e})"
            )
    
    @staticmethod
    def _detect_pyrubberband() -> MethodInfo:
        """Detects PyRubberband."""
        try:
            import pyrubberband
            
            # Checks external dependency
            import subprocess
            try:
                subprocess.run(["rubberband", "--help"], 
                             capture_output=True, timeout=5)
                external_available = True
                limitations = ["requires_external_binary"]
            except (subprocess.TimeoutExpired, FileNotFoundError):
                external_available = False
                limitations = ["requires_external_binary", "rubberband_cli_not_found"]
            
            return MethodInfo(
                name="PyRubberband",
                library_required="pyrubberband",
                available=external_available,
                quality_score=0.95,
                speed_score=0.7,
                memory_efficiency=0.8,
                best_use_cases=["professional", "mastering", "high_quality"],
                limitations=limitations,
                requires_external=True,
                installation_notes="pip install pyrubberband + install rubberband-cli"
            )
            
        except ImportError as e:
            return MethodInfo(
                name="PyRubberband",
                library_required="pyrubberband",
                available=False,
                requires_external=True,
                installation_notes=f"pip install pyrubberband (Error: {e})"
            )
    
    @staticmethod
    def _detect_librosa() -> MethodInfo:
        """Detects LibROSA."""
        try:
            import librosa
            return MethodInfo(
                name="LibROSA Standard",
                library_required="librosa",
                available=True,
                quality_score=0.8,
                speed_score=0.8,
                memory_efficiency=0.8,
                best_use_cases=["general", "analysis", "development"],
                limitations=["moderate_quality"],
                installation_notes="pip install librosa soundfile"
            )
        except ImportError as e:
            return MethodInfo(
                name="LibROSA Standard",
                library_required="librosa",
                available=False,
                installation_notes=f"pip install librosa soundfile (Error: {e})"
            )
    
    @staticmethod
    def _detect_pedalboard() -> MethodInfo:
        """Detects Pedalboard."""
        try:
            import pedalboard
            return MethodInfo(
                name="Spotify Pedalboard",
                library_required="pedalboard",
                available=True,
                quality_score=0.9,
                speed_score=0.8,
                memory_efficiency=0.9,
                best_use_cases=["professional", "music_production", "plugins"],
                limitations=["newer_library"],
                installation_notes="pip install pedalboard"
            )
        except ImportError as e:
            return MethodInfo(
                name="Spotify Pedalboard",
                library_required="pedalboard",
                available=False,
                installation_notes=f"pip install pedalboard (Error: {e})"
            )
    
    @staticmethod
    def _detect_scipy_manual() -> MethodInfo:
        """Detects SciPy for manual implementation."""
        try:
            import scipy
            import numpy
            return MethodInfo(
                name="SciPy Manual",
                library_required="scipy",
                available=True,
                quality_score=0.6,
                speed_score=0.9,
                memory_efficiency=1.0,
                best_use_cases=["educational", "research", "embedded"],
                limitations=["basic_algorithm", "manual_implementation"],
                installation_notes="pip install scipy numpy"
            )
        except ImportError as e:
            return MethodInfo(
                name="SciPy Manual",
                library_required="scipy",
                available=False,
                installation_notes=f"pip install scipy numpy (Error: {e})"
            )
    
    @staticmethod
    def _detect_scipy_autotune() -> MethodInfo:
        """Detects SciPy for complete autotune."""
        try:
            import scipy
            import numpy
            return MethodInfo(
                name="SciPy Autotune",
                library_required="scipy",
                available=True,
                quality_score=0.75,
                speed_score=0.7,
                memory_efficiency=0.9,
                best_use_cases=["intelligent_correction", "automatic", "research"],
                limitations=["experimental", "basic_pitch_detection"],
                installation_notes="pip install scipy numpy"
            )
        except ImportError as e:
            return MethodInfo(
                name="SciPy Autotune",
                library_required="scipy",
                available=False,
                installation_notes=f"pip install scipy numpy (Error: {e})"
            )
# Em multi_autotune.py, dentro da classe LibraryDetector

    @staticmethod
    def _detect_soundtouch() -> MethodInfo:
        """Detects if the 'soundstretch' command-line tool is available."""
        
        def is_tool_available():
            # Função auxiliar para verificar se o comando existe no PATH
            for path in os.environ.get("PATH", "").split(os.pathsep):
                for exe in ["soundstretch", "soundstretch.exe"]:
                    if (Path(path) / exe).is_file():
                        return True
            return False

        available = is_tool_available()
        
        return MethodInfo(
            name="SoundTouch Shifter (CLI)",
            library_required="soundtouch-cli", # Nome mais descritivo
            available=available,
            quality_score=0.9,
            speed_score=0.8,
            memory_efficiency=0.9, # Mais eficiente por ser um processo separado
            best_use_cases=["production", "broadcast", "robust_processing"],
            limitations=["Requires SoundTouch command-line tools to be in the system's PATH"],
            installation_notes="Compile and install SoundTouch from source (https://codeberg.org/soundtouch/soundtouch )"
        )

    
    @staticmethod
    def _detect_vst_support() -> MethodInfo:
        """Detects VST support via Pedalboard."""
        try:
            import pedalboard
            # A disponibilidade real depende de um caminho de VST válido ser fornecido
            return MethodInfo(
                name="VST Plugin Shifter",
                library_required="pedalboard",
                available=True, # Disponível se pedalboard estiver instalado
                quality_score=0.95, # Depende do VST
                speed_score=0.7,  # Depende do VST
                memory_efficiency=0.7, # Depende do VST
                best_use_cases=["professional_autotune", "custom_effects"],
                limitations=["requires valid VST3 path", "quality depends on plugin"],
                installation_notes="pip install pedalboard"
            )
        except ImportError:
            return MethodInfo(name="VST Plugin Shifter", library_required="pedalboard", available=False, installation_notes="pip install pedalboard")

# ===== PARAMETER VALIDATION =====

class ParameterValidator:
    """Rigorous parameter validator."""
    
    @staticmethod
    def validate_method(method: Union[str, List[str]]) -> Tuple[bool, List[str], Union[str, List[str]]]:
        """Validates selected method(s)."""
        errors = []
        
        if isinstance(method, str):
            if method not in VALID_METHODS:
                errors.append(f"Invalid method: '{method}'")
                errors.append(f"Valid methods: {', '.join(VALID_METHODS)}")
                return False, errors, method
        
        elif isinstance(method, list):
            invalid_methods = [m for m in method if m not in VALID_METHODS]
            if invalid_methods:
                errors.append(f"Invalid methods: {invalid_methods}")
                errors.append(f"Valid methods: {', '.join(VALID_METHODS)}")
                return False, errors, method
        
        else:
            errors.append(f"Invalid type for method: {type(method)}")
            errors.append("Must be string or list of strings")
            return False, errors, method
        
        return True, errors, method
    
    @staticmethod
    def validate_force(force: float) -> Tuple[bool, List[str], float]:
        """Validates force parameter."""
        errors = []
        
        if not isinstance(force, (int, float)):
            errors.append(f"Force must be numeric, received: {type(force)}")
            return False, errors, 0.85
        
        if not 0.0 <= force <= 1.0:
            errors.append(f"Force must be between 0.0 and 1.0, received: {force}")
            if force < 0.0:
                errors.append("Suggestion: Use 0.0 to disable autotune")
            elif force > 1.0:
                errors.append("Suggestion: Use 1.0 for maximum correction")
            return False, errors, np.clip(force, 0.0, 1.0)
        
        # Suggestions based on value
        if force == 0.0:
            errors.append("INFO: Force 0.0 disables autotune")
        elif force < 0.3:
            errors.append("INFO: Low force - subtle correction")
        elif force > 0.9:
            errors.append("INFO: High force - aggressive correction, may generate artifacts")
        
        return True, errors, force
    
    @staticmethod
    def validate_profile(profile: Optional[str]) -> Tuple[bool, List[str], Optional[str]]:
        """Validates processing profile."""
        if profile is None:
            return True, [], None
        
        errors = []
        valid_profiles = [p.value for p in ProcessingProfile]
        
        if profile not in valid_profiles:
            errors.append(f"Invalid profile: '{profile}'")
            errors.append(f"Valid profiles: {', '.join(valid_profiles)}")
            return False, errors, None
        
        return True, errors, profile
    
    @staticmethod
    def validate_file_paths(audio_path: Union[str, Path], 
                          midi_path: Union[str, Path]) -> Tuple[bool, List[str]]:
        """Validates file paths."""
        errors = []
        
        # Convert to Path
        audio_path = Path(audio_path)
        midi_path = Path(midi_path)
        
        # Check existence
        if not audio_path.exists():
            errors.append(f"Audio file not found: {audio_path}")
        elif not audio_path.is_file():
            errors.append(f"Audio path is not a file: {audio_path}")
        
        if not midi_path.exists():
            errors.append(f"MIDI file not found: {midi_path}")
        elif not midi_path.is_file():
            errors.append(f"MIDI path is not a file: {midi_path}")
        
        # Check extensions
        audio_extensions = {'.wav', '.mp3', '.ogg', '.flac', '.m4a', '.aac'}
        if audio_path.suffix.lower() not in audio_extensions:
            errors.append(f"Unsupported audio extension: {audio_path.suffix}")
            errors.append(f"Supported extensions: {', '.join(audio_extensions)}")
        
        midi_extensions = {'.mid', '.midi'}
        if midi_path.suffix.lower() not in midi_extensions:
            errors.append(f"Unsupported MIDI extension: {midi_path.suffix}")
            errors.append(f"Supported extensions: {', '.join(midi_extensions)}")
        
        return len(errors) == 0, errors
    
    @staticmethod
    def validate_all_parameters(
        audio_path: Union[str, Path],
        midi_path: Union[str, Path],
        force: float,
        method: Union[str, List[str]],
        profile: Optional[str] = None
    ) -> Tuple[bool, List[str]]:
        """Complete validation of all parameters."""
        all_errors = []
        
        # Validate files
        files_valid, file_errors = ParameterValidator.validate_file_paths(audio_path, midi_path)
        all_errors.extend(file_errors)
        
        # Validate force
        force_valid, force_errors, _ = ParameterValidator.validate_force(force)
        all_errors.extend(force_errors)
        
        # Validate method
        method_valid, method_errors, _ = ParameterValidator.validate_method(method)
        all_errors.extend(method_errors)
        
        # Validate profile
        profile_valid, profile_errors, _ = ParameterValidator.validate_profile(profile)
        all_errors.extend(profile_errors)
        
        return len(all_errors) == 0, all_errors

# ===== ADVANCED LOGGING SYSTEM =====

class AutotuneLogger:
    """Specialized logging system for autotune."""
    
    def __init__(self, level: str = "INFO", log_file: Optional[str] = None):
        """Initializes logger."""
        self.logger = logging.getLogger("MultiLibraryAutotune")
        self.logger.setLevel(getattr(logging, level.upper()))
        
        # Configure handlers
        if not self.logger.handlers:
            # Console handler
            console_handler = logging.StreamHandler()
            console_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            console_handler.setFormatter(console_formatter)
            self.logger.addHandler(console_handler)
            
            # File handler if specified
            if log_file:
                file_handler = logging.FileHandler(log_file)
                file_formatter = logging.Formatter(
                    '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
                )
                file_handler.setFormatter(file_formatter)
                self.logger.addHandler(file_handler)
    
    def log_method_selection(self, method: str, reason: str, audio_characteristics: AudioCharacteristics):
        """Log of method selection with justification."""
        self.logger.info(f"Selected method: {method}")
        self.logger.info(f"Reason: {reason}")
        self.logger.debug(f"Audio characteristics: duration={audio_characteristics.duration:.2f}s, "
                         f"complexity={audio_characteristics.complexity_score:.3f}")
    
    def log_fallback_usage(self, original: str, fallback: str, error: str):
        """Log of fallback usage with error details."""
        self.logger.warning(f"Fallback activated: {original} → {fallback}")
        self.logger.warning(f"Original error: {error}")
    
    def log_processing_start(self, method: str, audio_path: str, midi_path: str, force: float):
        """Log of processing start."""
        self.logger.info(f"Starting processing with {method}")
        self.logger.info(f"Audio: {Path(audio_path).name}")
        self.logger.info(f"MIDI: {Path(midi_path).name}")
        self.logger.info(f"Force: {force:.2f}")
    
    def log_processing_complete(self, method: str, processing_time: float, 
                              quality_metrics: Dict[str, float]):
        """Log of processing completion."""
        self.logger.info(f"Processing completed with {method} in {processing_time:.2f}s")
        if quality_metrics:
            snr = quality_metrics.get('snr', 0)
            if snr > 0:
                self.logger.info(f"Resulting SNR: {snr:.1f} dB")
    
    def log_benchmark_results(self, results: List[BenchmarkResult]):
        """Structured log of benchmark results."""
        self.logger.info("=== BENCHMARK RESULTS ===")
        for result in results:
            self.logger.info(f"{result.method_name}:")
            self.logger.info(f"  Time: {result.avg_processing_time:.3f}s")
            self.logger.info(f"  Quality: {result.avg_quality_score:.3f}")
            self.logger.info(f"  Success: {result.success_rate:.1%}")
    
    def log_error(self, error: Exception, context: str = ""):
        """Log of error with context."""
        if context:
            self.logger.error(f"Error in {context}: {error}")
        else:
            self.logger.error(f"Error: {error}")

# PARTE 3
    def get_logger(self):
        """Returns logger instance."""
        return self.logger

# ===== AUTOMATIC SELECTION SYSTEM =====

class AutoMethodSelector:
    """Intelligent automatic method selector."""
    
    def __init__(self, available_methods: Dict[str, MethodInfo]):
        """Initializes selector."""
        self.available_methods = available_methods
        self.performance_history = {}
        self.decision_tree = self._build_decision_tree()
        self.logger = AutotuneLogger().get_logger()
    
    def select_optimal_method(self, 
                            audio_characteristics: AudioCharacteristics,
                            user_requirements: UserRequirements,
                            system_constraints: SystemConstraints) -> Tuple[str, str]:
        """
        Selection based on multiple factors.
        
        Returns:
            Tuple[selected_method, justification]
        """
        available_methods = [m for m, info in self.available_methods.items() 
                           if info.available and m != "auto"]
        
        if not available_methods:
            return "scipy_manual", "No method available, using fallback"
        
        # Apply decision tree
        selected_method = self._apply_decision_tree(
            audio_characteristics, user_requirements, system_constraints, available_methods
        )
        
        # Generate justification
        reason = self._generate_selection_reason(
            selected_method, audio_characteristics, user_requirements
        )
        
        return selected_method, reason
    
    def _build_decision_tree(self) -> Dict[str, Any]:
        """Builds decision tree for automatic selection."""
        return {
            "rules": [
                {
                    "condition": lambda ac, ur, sc: ur.use_case == "realtime",
                    "method": "pydub_speed",
                    "reason": "Optimized for real-time"
                },
                {
                    "condition": lambda ac, ur, sc: ur.quality_priority > 0.9 and ac.duration < 30,
                    "method": "pyrubberband_shift",
                    "fallback": "librosa_hifi",
                    "reason": "Maximum quality for short audio"
                },
                {
                    "condition": lambda ac, ur, sc: ac.duration > 300,  # > 5 minutes
                    "method": "librosa_standard",
                    "reason": "Balanced for long files"
                },
                {
                    "condition": lambda ac, ur, sc: ac.complexity_score > 0.8,
                    "method": "pedalboard_shift",
                    "fallback": "librosa_hifi",
                    "reason": "High quality for complex audio"
                },
                {
                    "condition": lambda ac, ur, sc: ur.speed_priority > 0.8,
                    "method": "scipy_manual",
                    "reason": "Speed priority"
                },
                {
                    "condition": lambda ac, ur, sc: ac.harmonic_content > 0.7,
                    "method": "scipy_autotune",
                    "reason": "High harmonic content - intelligent autotune"
                },
                                {
                    "condition": lambda ac, ur, sc: ur.quality_priority > 0.85,
                    "method": "soundtouch_shift", # Adicionar soundtouch como opção de alta qualidade
                    "fallback": "pyrubberband_shift",
                    "reason": "High quality priority, using SoundTouch"
                }
            ],
            "default": "librosa_standard"
        }
    
    def _apply_decision_tree(self, 
                           audio_characteristics: AudioCharacteristics,
                           user_requirements: UserRequirements,
                           system_constraints: SystemConstraints,
                           available_methods: List[str]) -> str:
        """Applies decision tree."""
        
        # Test rules in order
        for rule in self.decision_tree["rules"]:
            try:
                if rule["condition"](audio_characteristics, user_requirements, system_constraints):
                    method = rule["method"]
                    
                    # Check if method is available
                    if method in available_methods:
                        return method
                    
                    # Try fallback if defined
                    if "fallback" in rule and rule["fallback"] in available_methods:
                        return rule["fallback"]
            except Exception as e:
                self.logger.warning(f"Error evaluating decision rule: {e}")
                continue
        
        # Default method
        default = self.decision_tree["default"]
        if default in available_methods:
            return default
        
        # Last resort: first available method
        return available_methods[0]
    
    def _generate_selection_reason(self, 
                                 method: str,
                                 audio_characteristics: AudioCharacteristics,
                                 user_requirements: UserRequirements) -> str:
        """Generates justification for the selection."""
        reasons = []
        
        method_info = self.available_methods.get(method)
        if not method_info:
            return f"Method {method} selected as fallback"
        
        # Based on audio characteristics
        if audio_characteristics.duration < 10:
            reasons.append("short audio")
        elif audio_characteristics.duration > 180:
            reasons.append("long audio")
        
        if audio_characteristics.complexity_score > 0.7:
            reasons.append("high spectral complexity")
        elif audio_characteristics.complexity_score < 0.3:
            reasons.append("low complexity")
        
        # Based on user requirements
        if user_requirements.quality_priority > 0.8:
            reasons.append("quality priority")
        elif user_requirements.speed_priority > 0.8:
            reasons.append("speed priority")
        
        # Based on method characteristics
        if method_info.quality_score > 0.9:
            reasons.append("high quality")
        if method_info.speed_score > 0.9:
            reasons.append("high speed")
        
        reason_text = ", ".join(reasons) if reasons else "general balancing"
        return f"Selected for: {reason_text}"

# ===== FALLBACK SYSTEM =====

class FallbackManager:
    """Intelligent fallback manager."""
    
    def __init__(self, available_methods: Dict[str, MethodInfo]):
        """Initializes fallback manager."""
        self.available_methods = available_methods
        self.logger = AutotuneLogger().get_logger()
    
    def build_fallback_chain(self, preferred_method: str) -> List[str]:
        """Builds fallback chain based on similarity."""
        if preferred_method == "auto":
            # For auto, uses all available methods ordered by quality
            available = [(name, info) for name, info in self.available_methods.items() 
                        if info.available and name != "auto"]
            available.sort(key=lambda x: x[1].get_overall_score(), reverse=True)
            return [name for name, _ in available]
        
        # Uses predefined chain if available
        if preferred_method in FALLBACK_CHAINS:
            base_chain = FALLBACK_CHAINS[preferred_method]
            # Filters only available methods
            return [method for method in base_chain 
                   if method in self.available_methods and 
                   self.available_methods[method].available]
        
        # Generic fallback based on scores
        available = [(name, info) for name, info in self.available_methods.items() 
                    if info.available and name != preferred_method and name != "auto"]
        available.sort(key=lambda x: x[1].get_overall_score(), reverse=True)
        
        return [name for name, _ in available]
    
    def execute_with_fallback(self, 
                            method_chain: List[str],
                            processor_func: callable,
                            *args, **kwargs) -> Tuple[Any, str, List[str]]:
        """
        Executes processing with automatic fallback.
        
        Returns:
            Tuple[result, method_used, failures_occurred]
        """
        failures = []
        
        for method in method_chain:
            try:
                self.logger.info(f"Trying method: {method}")
                result = processor_func(method, *args, **kwargs)
                
                if result is not None:
                    if failures:
                        self.logger.info(f"Success with fallback: {method}")
                    return result, method, failures
                else:
                    raise ValueError(f"Method {method} returned None")
                    
            except Exception as e:
                error_msg = f"{method}: {str(e)}"
                failures.append(error_msg)
                self.logger.warning(f"Failure in {method}: {e}")
                
                # If not the last method, continue
                if method != method_chain[-1]:
                    self.logger.info(f"Trying next method...")
                    continue
                else:
                    # Last method also failed
                    break
        
        # All methods failed
        raise RuntimeError(f"All methods failed: {failures}")

# ===== BASE CLASS FOR PITCH SHIFTERS =====

class BasePitchShifter:
    """Base class for pitch shifting implementations."""
    
    def __init__(self, config: AutotuneConfig):
        """Initializes shifter."""
        self.config = config
        self.logger = AutotuneLogger().get_logger()
        self.method_name = self.__class__.__name__
    
    def shift_pitch(self, audio: np.ndarray, sr: int, semitones: float) -> np.ndarray:
        """
        Applies pitch shifting. Must be implemented by subclasses.
        
        Args:
            audio: Audio signal
            sr: Sample rate
            semitones: Number of semitones for shift
            
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
        
        # Smoothing at edges to avoid clicks
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
        Processes with metric monitoring.
        
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

# ===== FACTORY FOR METHOD CREATION (CORRECTED) =====

class PitchShiftMethodFactory:
    """Factory for creating pitch shifting methods."""
    
    def __init__(self, config: AutotuneConfig):
        """Initializes factory."""
        self.config = config
        self.available_methods = LibraryDetector.detect_all_libraries()
        self.logger = AutotuneLogger().get_logger()
    
    def create_method(self, method_name: str) -> BasePitchShifter:
        """Creates instance of the specified method."""
        if method_name not in self.available_methods:
            raise ValueError(f"Unknown method: {method_name}")
        
        method_info = self.available_methods[method_name]
        if not method_info.available:
            raise RuntimeError(f"Method {method_name} is not available: {method_info.installation_notes}")
        
        # Import and create the appropriate class
        if method_name == "pydub_speed":
            from methods.pydub_shifter import PyDubSpeedShifter
            return PyDubSpeedShifter(self.config)
        
        elif method_name == "pyrubberband_shift":
            from methods.pyrubberband_shifter import PyRubberbandShifter
            return PyRubberbandShifter(self.config)
        
        elif method_name == "librosa_standard":
            from methods.librosa_shifters import LibROSAStandardShifter
            return LibROSAStandardShifter(self.config)
        
        elif method_name == "librosa_hifi":
            from methods.librosa_shifters import LibROSAHiFiShifter
            return LibROSAHiFiShifter(self.config)
        
        elif method_name == "pedalboard_shift":
            from methods.pedalboard_shifter import PedalboardShifter
            return PedalboardShifter(self.config)
        
        elif method_name == "scipy_manual":
            from methods.scipy_shifters import SciPyManualShifter
            return SciPyManualShifter(self.config)
        
        elif method_name == "scipy_autotune":
            from methods.scipy_shifters import SciPyAutotuneShifter
            return SciPyAutotuneShifter(self.config)
        
        elif method_name == "soundtouch_shift":
            from methods.soundtouch_shifter import SoundTouchShifter
            return SoundTouchShifter(self.config)
        
        elif method_name == "vst_plugin_shift":
            from methods.vst_plugin_shifter import VSTPluginShifter
            return VSTPluginShifter(self.config)
        
        else:
            raise ValueError(f"Implementation not found for: {method_name}")
    
    def get_available_methods(self) -> List[str]:
        """Returns list of available methods."""
        return [name for name, info in self.available_methods.items() 
                if info.available and name != "auto"]
    
    def get_method_info(self, method_name: str) -> MethodInfo:
        """Returns information about a method."""
        if method_name not in self.available_methods:
            raise ValueError(f"Unknown method: {method_name}")
        return self.available_methods[method_name]
# PARTE 4

# ===== PITCH SHIFTING METHOD IMPLEMENTATIONS =====

class PyDubSpeedShifter(BasePitchShifter):
    """
    Pitch shifting using PyDub (speed change).
    
    Characteristics:
    - Speed: ★★★★★ (fastest)
    - Quality: ★★☆☆☆ (changes duration)  
    - Resources: ★★★★★ (low usage)
    - Ideal use: Prototyping, creative effects, real-time
    """
    
    def shift_pitch(self, audio: np.ndarray, sr: int, semitones: float) -> np.ndarray:
        """PyDub implementation with quality control."""
        try:
            from pydub import AudioSegment
            
            # Convert to PyDub format
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
            
            # Convert back to numpy
            shifted_bytes = shifted_segment.raw_data
            shifted_audio = np.frombuffer(shifted_bytes, dtype=np.int16)
            
            return shifted_audio.astype(np.float32) / 32767
            
        except Exception as e:
            raise RuntimeError(f"PyDub pitch shift failed: {e}")

class PyRubberbandShifter(BasePitchShifter):
    """
    Pitch shifting using pyrubberband.
    
    Characteristics:
    - Speed: ★★★☆☆ (medium)
    - Quality: ★★★★★ (excellent)
    - Resources: ★★★☆☆ (requires rubberband-cli)
    - Ideal use: Professional production, mastering
    """
    
    def __init__(self, config: AutotuneConfig):
        """Initializes with external dependency verification."""
        super().__init__(config)
        self._check_external_dependency()
    
    def _check_external_dependency(self) -> bool:
        """Checks if rubberband-cli is available."""
        try:
            import subprocess
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

class LibROSAStandardShifter(BasePitchShifter):
    """
    Pitch shifting using LibROSA with standard settings.
    
    Characteristics:
    - Speed: ★★★★☆ (fast)
    - Quality: ★★★★☆ (good)
    - Resources: ★★★★☆ (efficient)
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
                hop_length=self.config.hop_length // 2  # Higher resolution
            )
            
        except Exception as e:
            raise RuntimeError(f"LibROSA hi-fi pitch shift failed: {e}")

class PedalboardShifter(BasePitchShifter):
    """
    Pitch shifting using Spotify Pedalboard.
    
    Characteristics:
    - Speed: ★★★★☆ (fast)
    - Quality: ★★★★★ (excellent)
    - Resources: ★★★★☆ (optimized)
    - Ideal use: Music production, professional plugins
    """
    
    def shift_pitch(self, audio: np.ndarray, sr: int, semitones: float) -> np.ndarray:
        """Implementation with Spotify Pedalboard."""
        try:
            import pedalboard
            
            # Create pitch shift effect
            pitch_shifter = pedalboard.PitchShift(semitones=semitones)
            
            # Apply effect
            return pitch_shifter(audio, sample_rate=sr)
            
        except Exception as e:
            raise RuntimeError(f"Pedalboard pitch shift failed: {e}")

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
            from scipy import signal
            
            # Parameters
            hop_length = self.config.hop_length
            n_fft = self.config.frame_length
            
            # Pitch factor
            pitch_factor = 2 ** (semitones / 12)
            
            # STFT
            f, t, Zxx = signal.stft(audio, sr, nperseg=n_fft, noverlap=n_fft-hop_length)
            
            # Magnitude/phase separation
            magnitude = np.abs(Zxx)
            phase = np.angle(Zxx)
            
            # Frequency shift in magnitude
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
                # If no pitch detected, apply direct shift
                return self._apply_manual_shift(audio, sr, semitones)
            
            # Calculate target pitch
            target_pitch = current_pitch * (2 ** (semitones / 12))
            
            # Apply adaptive correction
            return self._apply_adaptive_correction(audio, sr, current_pitch, target_pitch)
            
        except Exception as e:
            raise RuntimeError(f"SciPy autotune failed: {e}")
    
    def _detect_pitch_autocorr(self, audio: np.ndarray, sr: int) -> float:
        """Detects pitch using autocorrelation."""
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
        """Applies manual shift as fallback."""
        # Use simple manual implementation
        manual_shifter = SciPyManualShifter(self.config)
        return manual_shifter.shift_pitch(audio, sr, semitones)

# PARTE 5

    def _apply_adaptive_correction(self, audio: np.ndarray, sr: int, 
                                 current_pitch: float, target_pitch: float) -> np.ndarray:
        """Applies adaptive correction based on detected pitch."""
        correction_factor = target_pitch / current_pitch
        correction_semitones = 12 * np.log2(correction_factor)
        
        # Limit correction to avoid extreme artifacts
        correction_semitones = np.clip(correction_semitones, -12, 12)
        
        # Apply correction using manual method
        return self._apply_manual_shift(audio, sr, correction_semitones)

# ===== MAIN PROCESSOR =====

class MultiLibraryAutotuneProcessor:
    """Main processor of the multi-library system."""
    
    def __init__(self, config: Optional[AutotuneConfig] = None):
        """Initializes processor."""
        self.config = config or AutotuneConfig()
        self.logger = AutotuneLogger()
        
        # Detect available methods
        self.available_methods = LibraryDetector.detect_all_libraries()
        
        # Initialize components
        self.method_factory = PitchShiftMethodFactory(self.config)
        self.auto_selector = AutoMethodSelector(self.available_methods)
        self.fallback_manager = FallbackManager(self.available_methods)
        
        # Benchmark cache
        self.benchmark_cache: Dict[str, BenchmarkResult] = {}
        
        self.logger.get_logger().info(f"Processor initialized with {len(self.get_available_methods())} methods")
    
    def get_available_methods(self) -> List[str]:
        """Returns available methods."""
        return [name for name, info in self.available_methods.items() 
                if info.available and name != "auto"]
    
    def get_method_info(self, method_name: str) -> MethodInfo:
        """Returns information about a method."""
        return self.method_factory.get_method_info(method_name)
    
    def process_autotune(self, 
                        audio_path: Union[str, Path],
                        midi_path: Union[str, Path],
                        force: float = 0.85,
                        method: Union[str, List[str]] = "auto",
                        output_path: Optional[Union[str, Path]] = None) -> AutotuneResult:
        """
        Processes autotune with specified method or automatic selection.
        
        Args:
            audio_path: Path to audio file
            midi_path: Path to MIDI file
            force: Correction intensity (0.0-1.0)
            method: Specific method, "auto", or list of methods
            output_path: Output path (optional)
            
        Returns:
            Processing result
        """
        start_time = time.time()
        
        try:
            # Parameter validation
            validation_result = self._validate_parameters(audio_path, midi_path, force, method)
            if not validation_result.success:
                return validation_result
            
            # Load files
            audio, sr = self._load_audio(audio_path)
            midi_notes = self._load_midi(midi_path)
            
            # Analyze audio characteristics
            audio_characteristics = AudioCharacteristics.analyze_audio(audio, sr)
            
            # Select method(s)
            if isinstance(method, str) and method == "auto":
                user_requirements = UserRequirements(quality_priority=self.config.quality_priority)
                system_constraints = SystemConstraints.detect_system_constraints()
                system_constraints.available_methods = self.get_available_methods()
                
                selected_method, reason = self.auto_selector.select_optimal_method(
                    audio_characteristics, user_requirements, system_constraints
                )
                method_chain = [selected_method]
                
                self.logger.log_method_selection(selected_method, reason, audio_characteristics)
                
            elif isinstance(method, str):
                method_chain = [method]
            else:
                method_chain = method
            
            # Build fallback chain
            fallback_chain = self.fallback_manager.build_fallback_chain(method_chain[0])
            full_chain = method_chain + [m for m in fallback_chain if m not in method_chain]
            
            # Log processing start
            self.logger.log_processing_start(full_chain[0], str(audio_path), str(midi_path), force)
            
            # Execute processing with fallback
            try:
                processed_audio, method_used, failures = self.fallback_manager.execute_with_fallback(
                    full_chain,
                    self._process_with_method,
                    audio, sr, midi_notes, force
                )
                
                # Define output path
                if output_path is None:
                    input_path = Path(audio_path)
                    output_path = input_path.parent / f"{input_path.stem}_autotuned_{method_used}.wav"
                
                # Save result
                self._save_audio(processed_audio, sr, output_path)
                
                # Calculate quality metrics
                quality_metrics = self._calculate_quality_metrics(audio, processed_audio, sr)
                
                processing_time = time.time() - start_time
                
                # Log completion
                self.logger.log_processing_complete(method_used, processing_time, quality_metrics)
                
                return AutotuneResult(
                    success=True,
                    output_path=str(output_path),
                    method_used=method_used,
                    processing_time=processing_time,
                    quality_metrics=quality_metrics,
                    fallback_methods_tried=[f.split(':')[0] for f in failures],
                    original_audio_stats=self._calculate_audio_stats(audio, sr),
                    processed_audio_stats=self._calculate_audio_stats(processed_audio, sr)
                )
                
            except RuntimeError as e:
                return AutotuneResult(
                    success=False,
                    error_message=str(e),
                    processing_time=time.time() - start_time,
                    fallback_methods_tried=full_chain
                )
                
        except Exception as e:
            self.logger.log_error(e, "process_autotune")
            return AutotuneResult(
                success=False,
                error_message=f"Unexpected error: {e}",
                processing_time=time.time() - start_time
            )
    
    def _validate_parameters(self, audio_path, midi_path, force, method) -> AutotuneResult:
        """Complete parameter validation."""
        is_valid, errors = ParameterValidator.validate_all_parameters(
            audio_path, midi_path, force, method
        )
        
        if not is_valid:
            return AutotuneResult(
                success=False,
                error_message=f"Invalid parameters: {'; '.join(errors)}"
            )
        
        return AutotuneResult(success=True)
    
    def _load_audio(self, audio_path: Union[str, Path]) -> Tuple[np.ndarray, int]:
        """Loads audio file."""
        try:
            import librosa
            audio, sr = librosa.load(str(audio_path), sr=self.config.sample_rate, mono=True)
            
            if len(audio) == 0:
                raise ValueError("Audio file is empty")
            
            return audio.astype(np.float32), sr
            
        except Exception as e:
            raise RuntimeError(f"Error loading audio: {e}")
    
    def _load_midi(self, midi_path: Union[str, Path]) -> List[Dict[str, float]]:
        """Loads MIDI file."""
        try:
            import pretty_midi
            
            midi_data = pretty_midi.PrettyMIDI(str(midi_path))
            notes = []
            
            for instrument in midi_data.instruments:
                if instrument.is_drum:
                    continue
                
                for note in instrument.notes:
                    if note.start >= note.end:
                        continue
                    
                    notes.append({
                        'start_time': note.start,
                        'end_time': note.end,
                        'note_number': note.pitch,
                        'frequency': 440.0 * (2.0 ** ((note.pitch - 69) / 12.0))
                    })
            
            if not notes:
                raise ValueError("No notes found in MIDI file")
            
            notes.sort(key=lambda x: x['start_time'])
            return notes
            
        except Exception as e:
            raise RuntimeError(f"Error loading MIDI: {e}")
    
    def _process_with_method(self, method_name: str, audio: np.ndarray, sr: int,
                           midi_notes: List[Dict], force: float) -> np.ndarray:
        """Processes audio with specific method."""
        if method_name not in self.available_methods:
            raise ValueError(f"Method not available: {method_name}")
        
        if not self.available_methods[method_name].available:
            raise RuntimeError(f"Method {method_name} is not available")
        
        # Create method instance
        method_instance = self.method_factory.create_method(method_name)
        
        # For methods that are not complete autotune, simulate basic autotune
        if method_name != "scipy_autotune":
            return self._simulate_autotune(method_instance, audio, sr, midi_notes, force)
        else:
            # For scipy_autotune, use native capability
            # (assumes the method can process multiple notes)
            return self._apply_full_autotune(method_instance, audio, sr, midi_notes, force)
    
    def _simulate_autotune(self, method_instance: BasePitchShifter, 
                          audio: np.ndarray, sr: int,
                          midi_notes: List[Dict], force: float) -> np.ndarray:
        """Simulates autotune using pitch shifting method."""
        # Simplified implementation: applies shift based on first note
        if not midi_notes:
            return audio
        
        # Detect current pitch (approximation)
        current_pitch = self._estimate_dominant_pitch(audio, sr)
        
        if current_pitch <= 0:
            return audio
        
        # Use first note as reference
        target_freq = midi_notes[0]['frequency']
        
        # Calculate necessary shift
        semitones = 12 * np.log2(target_freq / current_pitch)
        applied_semitones = semitones * force
        
        # Apply shift
        return method_instance.shift_pitch(audio, sr, applied_semitones)
    
    def _apply_full_autotune(self, method_instance: BasePitchShifter,
                           audio: np.ndarray, sr: int,
                           midi_notes: List[Dict], force: float) -> np.ndarray:
        """Applies complete autotune for methods that support it."""
        # For scipy_autotune, can process more intelligently
        # For simplicity, uses the same logic
        return self._simulate_autotune(method_instance, audio, sr, midi_notes, force)
    
    def _estimate_dominant_pitch(self, audio: np.ndarray, sr: int) -> float:
        """Estimates dominant pitch of the audio."""
        try:
            # Simple autocorrelation
            autocorr = np.correlate(audio, audio, mode='full')
            autocorr = autocorr[len(autocorr)//2:]
            
            # Limits for voice
            min_period = int(sr / 800)
            max_period = int(sr / 80)
            
            if max_period < len(autocorr):
                search_range = autocorr[min_period:max_period]
                if len(search_range) > 0:
                    peak_idx = np.argmax(search_range) + min_period
                    if autocorr[peak_idx] > 0.3 * autocorr[0]:
                        return sr / peak_idx
            
            return 0.0
            
        except Exception:
            return 0.0
    
    def _save_audio(self, audio: np.ndarray, sr: int, output_path: Union[str, Path]):
        """Saves processed audio."""
        try:
            import soundfile as sf
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            sf.write(str(output_path), audio, sr, subtype='PCM_16')
            
        except Exception as e:
            raise RuntimeError(f"Error saving audio: {e}")
    
    def _calculate_quality_metrics(self, original: np.ndarray, 
                                 processed: np.ndarray, sr: int) -> Dict[str, float]:
        """Calculates quality metrics."""
        metrics = {}
        
        try:
            # SNR
            if len(original) == len(processed):
                signal_power = np.mean(original ** 2)
                noise_power = np.mean((processed - original) ** 2)
                
                if noise_power > 0 and signal_power > 0:
                    metrics['snr'] = float(10 * np.log10(signal_power / noise_power))
                else:
                    metrics['snr'] = float('inf')
            
            # Correlation
            if len(original) == len(processed):
                correlation = np.corrcoef(original, processed)[0, 1]
                metrics['correlation'] = float(correlation)
            
            # Energy preservation
            original_energy = np.sum(original ** 2)
            processed_energy = np.sum(processed ** 2)
            
            if max(original_energy, processed_energy) > 0:
                energy_ratio = min(original_energy, processed_energy) / max(original_energy, processed_energy)
                metrics['energy_preservation'] = float(energy_ratio)
            
        except Exception as e:
            self.logger.log_error(e, "calculate_quality_metrics")
        
        return metrics
    
    def _calculate_audio_stats(self, audio: np.ndarray, sr: int) -> Dict[str, float]:
        """Calculates audio statistics."""
        return {
            'duration': len(audio) / sr,
            'rms': float(np.sqrt(np.mean(audio ** 2))),
            'peak': float(np.max(np.abs(audio))),
            'dynamic_range': float(20 * np.log10(np.max(np.abs(audio)) / (np.sqrt(np.mean(audio ** 2)) + 1e-10)))
        }

# ===== MAIN INTERFACE FUNCTION =====

def multi_library_autotune(
    audio_path: Union[str, Path],
    midi_path: Union[str, Path],
    force: float = 0.85,
    pitch_shift_method: Union[str, List[str]] = "auto",
    output_path: Optional[Union[str, Path]] = None,
    profile: Optional[str] = None,
    enable_benchmarking: bool = False,
    quality_threshold: float = 0.8,
    fallback_enabled: bool = True,
    preprocessing_options: Optional[Dict] = None,
    postprocessing_options: Optional[Dict] = None,
    config: Optional[AutotuneConfig] = None
) -> AutotuneResult:
    """
    Main function for multi-library autotune.
    
    Args:
        audio_path: Path to audio file
        midi_path: Path to MIDI file
        force: Correction intensity (0.0-1.0)
        pitch_shift_method: 
            - "auto": Automatic selection based on analysis
            - Specific method: one of the 7 available methods
            - List of methods: tries in order until success
        output_path: Output path (optional)
        profile: Predefined profile ("realtime", "production", "broadcast", "research", "creative")
        enable_benchmarking: Whether to run comparative benchmark
        quality_threshold: Minimum quality threshold
        fallback_enabled: Whether to use automatic fallback
        preprocessing_options: Preprocessing options
        postprocessing_options: Postprocessing options
        config: Custom configuration
        
    Returns:
        Processing result
    """
    
    # Apply profile if specified
    if profile:
        config = _apply_profile(profile, config)
    
    # Apply processing options
    if config is None:
        config = AutotuneConfig()
    
    if preprocessing_options:
        config.enable_preprocessing = preprocessing_options.get('enable', config.enable_preprocessing)
    
    if postprocessing_options:
        config.enable_postprocessing = postprocessing_options.get('enable', config.enable_postprocessing)
    
    # Configure fallback
    config.fallback_enabled = fallback_enabled
    config.quality_threshold = quality_threshold
    
    # Initialize processor
    processor = MultiLibraryAutotuneProcessor(config)
    
    # Execute processing
    result = processor.process_autotune(
        audio_path=audio_path,
        midi_path=midi_path,
        force=force,
        method=pitch_shift_method,
        output_path=output_path
    )
    
    # Execute benchmark if requested
    if enable_benchmarking and result.success:
        benchmark_results = _run_comparative_benchmark(
            processor, audio_path, midi_path, force
        )
        result.quality_metrics['benchmark_results'] = benchmark_results
    
    return result

def _apply_profile(profile: str, base_config: Optional[AutotuneConfig] = None) -> AutotuneConfig:
    """Applies predefined profile to configuration."""
    config = base_config or AutotuneConfig()
    
    profiles = {
        "realtime": {
            "quality_priority": 0.3,
            "enable_preprocessing": True,
            "enable_postprocessing": False,
            "processing_timeout": 5.0
        },
        "production": {
            "quality_priority": 0.9,
            "enable_preprocessing": True,
            "enable_postprocessing": True,
            "processing_timeout": 60.0,
            "quality_threshold": 0.9
        },
        "broadcast": {
            "quality_priority": 0.8,
            "enable_preprocessing": True,
            "enable_postprocessing": True,
            "processing_timeout": 30.0,
            "quality_threshold": 0.85
        },
        "research": {
            "quality_priority": 0.7,
            "enable_preprocessing": False,
            "enable_postprocessing": False,
            "processing_timeout": 120.0
        },
        "creative": {
            "quality_priority": 0.6,
            "enable_preprocessing": True,
            "enable_postprocessing": True,
            "processing_timeout": 45.0
        }
    }
    
    if profile in profiles:
        profile_settings = profiles[profile]
        for key, value in profile_settings.items():
            if hasattr(config, key):
                setattr(config, key, value)
    
    return config

def _run_comparative_benchmark(processor: MultiLibraryAutotuneProcessor,
                             audio_path: Union[str, Path],
                             midi_path: Union[str, Path],
                             force: float) -> Dict[str, Any]:
    """Executes comparative benchmark between available methods."""
    available_methods = processor.get_available_methods()
    benchmark_results = {}
    
    for method in available_methods:
        try:
            start_time = time.time()
            result = processor.process_autotune(
                audio_path=audio_path,
                midi_path=midi_path,
                force=force,
                method=method
            )
            
            benchmark_results[method] = {
                'success': result.success,
                'processing_time': result.processing_time,
                'quality_metrics': result.quality_metrics,
                'method_info': processor.get_method_info(method).get_overall_score()
            }
            
        except Exception as e:
            benchmark_results[method] = {
                'success': False,
                'error': str(e)
            }
    
    return benchmark_results

# ===== SYSTEM VALIDATION =====

def validate_system_setup() -> Dict[str, Any]:
    """Validates complete system configuration."""
    results = {
        'overall_status': True,
        'available_methods': [],
        'missing_dependencies': [],
        'warnings': [],
        'recommendations': []
    }
    
    # Detect libraries
    available_methods = LibraryDetector.detect_all_libraries()
    
    for method_name, method_info in available_methods.items():
        if method_name == "auto":
            continue
            
        if method_info.available:
            results['available_methods'].append({
                'name': method_name,
                'library': method_info.library_required,
                'quality_score': method_info.quality_score,
                'speed_score': method_info.speed_score
            })
        else:
            results['missing_dependencies'].append({
                'name': method_name,
                'library': method_info.library_required,
                'installation_notes': method_info.installation_notes,
                'requires_external': method_info.requires_external
            })
    
    # Check if there's at least one method available
    if not results['available_methods']:
        results['overall_status'] = False
        results['warnings'].append("No pitch shifting method available")
    
    # Recommendations
    if len(results['available_methods']) < 3:
        results['recommendations'].append("Install more libraries for greater flexibility")
    
    if not any(m['quality_score'] > 0.8 for m in results['available_methods']):
        results['recommendations'].append("Consider installing pyrubberband or pedalboard for high quality")
    
    return results

if __name__ == "__main__":
    # Usage example
    print("Multi-Library Autotune System v2.0")
    print("===================================")
    
    # Validate system
    validation = validate_system_setup()
    print(f"System status: {'✅' if validation['overall_status'] else '❌'}")
    print(f"Available methods: {len(validation['available_methods'])}")
    
    for method in validation['available_methods']:
        print(f"  • {method['name']} (Q:{method['quality_score']:.1f}, S:{method['speed_score']:.1f})")
    
    if validation['missing_dependencies']:
        print(f"\nMissing dependencies:")
        for dep in validation['missing_dependencies']:
            print(f"  ❌ {dep['name']}: {dep['installation_notes']}")
    
    if validation['recommendations']:
        print(f"\nRecommendations:")
        for rec in validation['recommendations']:
            print(f"  💡 {rec}")