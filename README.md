# Multi-Library MIDI Autotune System

A comprehensive, intelligent autotune system that automatically corrects pitch in audio files using MIDI sequences as reference. The system integrates multiple pitch-shifting libraries and provides automatic method selection based on audio characteristics and user requirements.

## 🎵 Features

### Core Functionality
- **MIDI-guided autotune**: Uses MIDI files as pitch reference for precise correction
- **Multi-library integration**: Supports 9 different pitch-shifting methods across 7 libraries
- **Intelligent method selection**: Automatically chooses optimal method based on audio analysis
- **Enhanced autotune engine**: Advanced F0 detection with multiple algorithms and intelligent processing
- **Robust fallback system**: Automatic failover when methods encounter issues
- **Real-time processing**: Optimized for various use cases from real-time to production quality
- **Comprehensive quality metrics**: Advanced audio analysis with SNR, spectral features, and artifact detection

### Supported Libraries & Methods
- **LibROSA**: Standard and high-quality implementations (`librosa_standard`, `librosa_hifi`)
- **PyRubberband**: Professional-grade quality (`pyrubberband_shift`) - requires rubberband-cli
- **Spotify Pedalboard**: Modern audio processing (`pedalboard_shift`)
- **PyDub**: Fast processing for real-time applications (`pydub_speed`)
- **SciPy**: Custom implementations for research (`scipy_manual`, `scipy_autotune`)
- **SoundTouch**: High-quality pitch shifting (`soundtouch_shift`) - requires soundtouch-cli
- **VST Plugin**: Professional VST integration (`vst_plugin_shift`) - requires pedalboard
- **Enhanced Engine**: Advanced autotune with robust F0 detection (`enhanced_engine`)

### Advanced Features
- **Audio analysis**: Automatic characterization of audio complexity and content
- **Enhanced F0 detection**: YIN algorithm with autocorrelation and spectral fallbacks
- **Smart MIDI processing**: Automatic looping and intelligent note mapping
- **Voice activity detection**: Improved VAD with multiple acoustic features
- **Quality metrics**: SNR, correlation, spectral analysis, and artifact detection
- **Predefined profiles**: Optimized settings for different use cases
- **Benchmarking system**: Comparative analysis of all available methods
- **Comprehensive logging**: Detailed processing information and debugging
- **CLI interface**: Complete command-line interface with extensive options

## 🚀 Quick Start

### Basic Installation

```bash
# Install core dependencies
pip install librosa soundfile numpy scipy pretty_midi tqdm

# Install optional libraries for better quality
pip install pyrubberband pedalboard pydub soundtouch

# For Ubuntu/Debian users (required for pyrubberband and soundtouch)
sudo apt-get install rubberband-cli soundtouch

# For macOS users
brew install rubberband soundtouch

# For Windows users
# Download rubberband and soundtouch binaries from their official websites
```

### Simple Usage

```python
from multi_autotune import multi_library_autotune

# Basic autotune with automatic method selection
result = multi_library_autotune(
    audio_path="vocal.wav",
    midi_path="melody.mid",
    force=0.85
)

if result.success:
    print(f"Success! Output: {result.output_path}")
    print(f"Method used: {result.method_used}")
    print(f"Processing time: {result.processing_time:.2f}s")
    print(f"Quality (SNR): {result.quality_metrics.get('snr', 0):.1f} dB")
else:
    print(f"Failed: {result.error_message}")
```

### Enhanced Engine Usage

```python
from enhanced_autotune_engine import enhanced_autotune

# Use the enhanced engine with advanced F0 detection
result = enhanced_autotune(
    audio_path="vocal.wav",
    midi_path="melody.mid",
    force=0.85,
    method="auto",  # or specific method
    log_level="INFO"
)

if result['success']:
    print(f"Enhanced processing completed!")
    print(f"Segments processed: {result['segments_processed']}")
    print(f"Total time: {result['total_processing_time']:.2f}s")
    # Access detailed processing report
    report = result['processing_report']
else:
    print(f"Failed: {result['error_message']}")
```

### Command Line Usage

```bash
# Basic processing
python multi_autotune_cli.py vocal.wav melody.mid

# With specific method
python multi_autotune_cli.py vocal.wav melody.mid --method librosa_hifi

# Using enhanced engine
python enhanced_autotune_engine.py vocal.wav melody.mid --method auto

# Using predefined profile
python multi_autotune_cli.py vocal.wav melody.mid --profile production

# Run benchmark comparison
python multi_autotune_cli.py vocal.wav melody.mid --benchmark

# System validation
python multi_autotune_cli.py --validate-setup
```

## 📦 Installation

### Method 1: Automatic Installation

```bash
# Clone the repository
git clone https://github.com/your-username/midi-autotune-system.git
cd midi-autotune-system

# Install dependencies
pip install -r requirements.txt

# Verify installation
python multi_autotune_cli.py --validate-setup
```

### Method 2: Manual Installation

```bash
# 1. Install Python dependencies
pip install -r requirements.txt

# 2. Install system dependencies (Ubuntu/Debian)
sudo apt-get install build-essential libsndfile1 ffmpeg rubberband-cli soundtouch

# 3. Install system dependencies (macOS)
brew install rubberband soundtouch ffmpeg

# 4. Verify installation
python multi_autotune_cli.py --validate-setup
```

### Method 3: Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv autotune_env

# Activate environment
source autotune_env/bin/activate  # Linux/Mac
# autotune_env\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Run quick test
python example_complete.py
```

## 🎛️ Available Methods

| Method | Library | Speed | Quality | Resources | Best Use Cases | External Deps |
|--------|---------|-------|---------|-----------|----------------|---------------|
| `auto` | Multi | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | Intelligent selection | - |
| `enhanced_engine` | Enhanced | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | Advanced F0 detection | - |
| `pyrubberband_shift` | PyRubberband | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | Professional production | rubberband-cli |
| `soundtouch_shift` | SoundTouch | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | High-quality processing | soundtouch-cli |
| `pedalboard_shift` | Pedalboard | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | Music production | - |
| `librosa_hifi` | LibROSA | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | High-quality mastering | - |
| `librosa_standard` | LibROSA | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | General purpose | - |
| `scipy_autotune` | SciPy | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | Intelligent correction | - |
| `scipy_manual` | SciPy | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | Research/embedded | - |
| `pydub_speed` | PyDub | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ | Real-time/creative | - |
| `vst_plugin_shift` | VST | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | Professional VST | pedalboard |

## 🎯 Usage Examples

### 1. Basic Processing

```python
from multi_autotune import multi_library_autotune

# Simple autotune
result = multi_library_autotune(
    audio_path="vocals.wav",
    midi_path="melody.mid"
)
```

### 2. Method Selection

```python
# Specific method
result = multi_library_autotune(
    audio_path="vocals.wav",
    midi_path="melody.mid",
    pitch_shift_method="librosa_hifi"
)

# Fallback chain
result = multi_library_autotune(
    audio_path="vocals.wav",
    midi_path="melody.mid",
    pitch_shift_method=["pyrubberband_shift", "librosa_hifi", "librosa_standard"]
)
```

### 3. Enhanced Engine Features

```python
from enhanced_autotune_engine import enhanced_autotune

# Advanced processing with detailed logging
result = enhanced_autotune(
    audio_path="vocals.wav",
    midi_path="melody.mid",
    force=0.85,
    method="auto",
    log_level="DEBUG"  # Detailed processing information
)

# Access detailed processing report
if result['success']:
    report = result['processing_report']
    print(f"F0 accuracy: {report['average_f0_accuracy']:.1%}")
    print(f"Processing efficiency: {report['quality_summary']['processing_efficiency']:.1%}")
    
    # Individual segment analysis
    for segment in report['segments'][:5]:  # First 5 segments
        print(f"Segment {segment['start_time']:.2f}s: "
              f"F0 {segment['detected_f0']:.1f}Hz → {segment['target_frequency']:.1f}Hz")
```

### 4. Predefined Profiles

```python
# Real-time processing (maximum speed)
result = multi_library_autotune(
    audio_path="vocals.wav",
    midi_path="melody.mid",
    profile="realtime"
)

# Production quality
result = multi_library_autotune(
    audio_path="vocals.wav",
    midi_path="melody.mid",
    profile="production"
)

# Broadcast quality
result = multi_library_autotune(
    audio_path="vocals.wav",
    midi_path="melody.mid",
    profile="broadcast"
)
```

### 5. Custom Configuration

```python
from multi_autotune import AutotuneConfig, multi_library_autotune

# Custom settings
config = AutotuneConfig(
    quality_priority=0.9,           # Prioritize quality over speed
    enable_preprocessing=True,       # Apply audio preprocessing
    enable_postprocessing=True,      # Apply audio postprocessing
    fallback_enabled=True,          # Enable automatic fallback
    quality_threshold=0.85          # Minimum quality threshold
)

result = multi_library_autotune(
    audio_path="vocals.wav",
    midi_path="melody.mid",
    config=config
)
```

### 6. Benchmarking

```python
# Run automatic benchmark
result = multi_library_autotune(
    audio_path="vocals.wav",
    midi_path="melody.mid",
    enable_benchmarking=True
)

# Access benchmark results
if 'benchmark_results' in result.quality_metrics:
    benchmark_data = result.quality_metrics['benchmark_results']
    for method, data in benchmark_data.items():
        if data['success']:
            print(f"{method}: {data['processing_time']:.2f}s, "
                  f"SNR: {data['quality_metrics']['snr']:.1f}dB")
```

### 7. Advanced Quality Analysis

```python
from multi_autotune import multi_library_autotune
import soundfile as sf

# Load original and processed audio
original, sr = sf.read("vocals.wav")
processed, _ = sf.read("vocals_autotuned.wav")

# Process with quality analysis
result = multi_library_autotune(
    audio_path="vocals.wav",
    midi_path="melody.mid",
    enable_benchmarking=True
)

if result.success:
    print(f"Overall quality score: {result.quality_metrics.get('quality_score', 0):.1f}")
    print(f"SNR: {result.quality_metrics.get('snr', 0):.1f} dB")
    print(f"Correlation: {result.quality_metrics.get('correlation', 0):.1f}")
```

## 💻 Command Line Interface

### Basic Commands

```bash
# Process with default settings
python multi_autotune_cli.py vocals.wav melody.mid

# Specify method and force
python multi_autotune_cli.py vocals.wav melody.mid --method librosa_hifi --force 0.9

# Use predefined profile
python multi_autotune_cli.py vocals.wav melody.mid --profile production

# Enhanced engine processing
python enhanced_autotune_engine.py vocals.wav melody.mid --method auto --log-level DEBUG
```

### Information Commands

```bash
# List available methods
python multi_autotune_cli.py --list-methods

# Validate system setup
python multi_autotune_cli.py --validate-setup

# Check dependencies
python multi_autotune_cli.py --check-dependencies

# Get method information
python multi_autotune_cli.py --method-info librosa_hifi
```

### Benchmarking Commands

```bash
# Run full benchmark
python multi_autotune_cli.py vocals.wav melody.mid --benchmark

# Compare specific methods
python multi_autotune_cli.py vocals.wav melody.mid --compare-methods pyrubberband_shift,librosa_hifi

# Save benchmark report
python multi_autotune_cli.py vocals.wav melody.mid --benchmark --benchmark-report results.json
```

### Advanced Options

```bash
# Custom quality and fallback settings
python multi_autotune_cli.py vocals.wav melody.mid \
    --quality-priority 0.9 \
    --quality-threshold 0.85 \
    --fallback-chain librosa_hifi,librosa_standard,scipy_manual

# Disable preprocessing/postprocessing
python multi_autotune_cli.py vocals.wav melody.mid \
    --disable-preprocessing \
    --disable-postprocessing

# Verbose logging with enhanced engine
python enhanced_autotune_engine.py vocals.wav melody.mid \
    --log-level DEBUG \
    --method auto \
    --force 0.9
```

## 🔧 Configuration

### Profiles

The system includes predefined profiles optimized for different use cases:

- **`realtime`**: Maximum speed for real-time processing
- **`production`**: Professional quality for music production  
- **`broadcast`**: Broadcast quality with optimized processing
- **`research`**: Configuration for research and analysis
- **`creative`**: Creative effects and experimentation

### Configuration File

Create a custom configuration file:

```json
{
  "audio": {
    "sample_rate": 44100,
    "hop_length": 256,
    "frame_length": 1024
  },
  "processing": {
    "default_force": 0.85,
    "voice_activity": {
      "threshold_percentile": 30
    },
    "pitch_detection": {
      "method": "yin",
      "fmin": 65.41,
      "fmax": 2093.0
    }
  },
  "output": {
    "normalize": true,
    "target_level_db": -3.0,
    "format": "wav"
  }
}
```

## 📊 Quality Metrics & Analysis

The system provides comprehensive quality analysis:

### Basic Metrics
- **SNR (Signal-to-Noise Ratio)**: Quality of the processed audio
- **Correlation**: Similarity between original and processed audio
- **Energy Preservation**: How well the original energy is maintained
- **Dynamic Range**: Preservation of audio dynamics

### Advanced Metrics (Enhanced Engine)
- **F0 Detection Accuracy**: Precision of fundamental frequency detection
- **Pitch Correction Accuracy**: How accurately pitch was corrected to target
- **Spectral Analysis**: Frequency domain characteristics and preservation
- **Processing Efficiency**: Ratio of successfully processed segments
- **Voice Activity Detection**: Accuracy of vocal segment identification

### Quality Analysis Tools

```python
# Use the built-in quality analysis
result = multi_library_autotune(
    audio_path="vocals.wav",
    midi_path="melody.mid",
    enable_benchmarking=True
)

# Access detailed metrics
if result.success:
    print("Quality Metrics:", result.quality_metrics)
    print("Audio Stats:", result.processed_audio_stats)
```

## 🧪 Testing and Validation

### Quick Test

```bash
# Run quick system test
python example_complete.py
```

### Complete Example

```bash
# Run complete demonstration
python example_complete.py
```

### Enhanced Engine Test

```bash
# Test enhanced engine specifically
python enhanced_autotune_engine.py --help
python enhanced_autotune_engine.py vocal.wav melody.mid --method auto
```

### Validation

```bash
# Validate system setup
python multi_autotune_cli.py --validate-setup

# Check all dependencies
python multi_autotune_cli.py --check-dependencies

# Test SoundTouch specifically
python test_soundtouch.py
```

## 🛠️ Development

### Project Structure

```
multi_autotune/
├── multi_autotune.py              # Main system module
├── multi_autotune_cli.py          # Command line interface
├── enhanced_autotune_engine.py    # Enhanced autotune engine
├── example_complete.py            # Complete usage example
├── test_soundtouch.py             # SoundTouch testing
├── requirements.txt               # Python dependencies
├── vst_config.json               # VST configuration
├── methods/                      # Pitch shifting implementations
│   ├── __init__.py
│   ├── base_shifter.py
│   ├── librosa_shifters.py
│   ├── pyrubberband_shifter.py
│   ├── pedalboard_shifter.py
│   ├── pydub_shifter.py
│   ├── scipy_shifters.py
│   ├── soundtouch_shifter.py
│   └── vst_plugin_shifter.py
├── vocal.wav                     # Sample audio file
├── melody.mid                    # Sample MIDI file
└── README.md
```

### Running Tests

```bash
# Run quick test
python example_complete.py

# Test enhanced engine
python enhanced_autotune_engine.py --help

# Test SoundTouch
python test_soundtouch.py

# Validate system
python multi_autotune_cli.py --validate-setup
```

### Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes and test thoroughly
4. Run the test suite: `python example_complete.py`
5. Test with enhanced engine: `python enhanced_autotune_engine.py test_audio.wav test_midi.mid`
6. Submit a pull request

## 📋 Requirements

### Python Version
- Python 3.8 or higher

### Core Dependencies
```
numpy>=1.21.0
scipy>=1.9.0
librosa>=0.10.0
soundfile>=0.12.1
pretty_midi>=0.2.9
tqdm>=4.64.0
```

### Optional Dependencies
```
pydub>=0.25.0                # pydub_speed
pyrubberband>=0.3.0          # pyrubberband_shift (requires rubberband-cli)
pedalboard>=0.7.0            # pedalboard_shift, vst_plugin_shift
soundtouch>=0.3.2            # soundtouch_shift (requires soundtouch-cli)
```

### System Dependencies

**Ubuntu/Debian:**
```bash
sudo apt-get install build-essential libsndfile1 ffmpeg rubberband-cli soundtouch
```

**macOS:**
```bash
brew install rubberband soundtouch ffmpeg
```

**Windows:**
- Download rubberband from: https://breakfastquay.com/rubberband/
- Download soundtouch from: https://www.surina.net/soundtouch/
- Install Microsoft Visual C++ Build Tools

## 🔍 Troubleshooting

### Common Issues

#### 1. Import Errors
```bash
# Install missing dependencies
pip install -r requirements.txt

# Verify installation
python -c "import librosa, soundfile, pretty_midi; print('OK')"
```

#### 2. Rubberband Not Found
```bash
# Ubuntu/Debian
sudo apt-get install rubberband-cli

# macOS
brew install rubberband

# Verify
rubberband --help
```

#### 3. SoundTouch Not Found
```bash
# Ubuntu/Debian
sudo apt-get install soundtouch

# macOS
brew install soundtouch

# Verify
soundstretch --help
```

#### 4. Enhanced Engine Issues
```bash
# Test enhanced engine specifically
python enhanced_autotune_engine.py --help

# Check if multi_autotune is properly imported
python -c "from enhanced_autotune_engine import enhanced_autotune; print('Enhanced engine OK')"
```

#### 5. Audio File Issues
```bash
# Check supported formats
python multi_autotune_cli.py --help

# Supported: WAV, MP3, OGG, FLAC, M4A, AAC
```

#### 6. Performance Issues
```bash
# Use faster methods
python multi_autotune_cli.py vocals.wav melody.mid --method pydub_speed

# Use realtime profile
python multi_autotune_cli.py vocals.wav melody.mid --profile realtime

# Enhanced engine with optimized settings
python enhanced_autotune_engine.py vocals.wav melody.mid --method scipy_manual
```

### Debug Mode

```bash
# Enable verbose logging
python multi_autotune_cli.py vocals.wav melody.mid --verbose --log-file debug.log

# Enhanced engine debug mode
python enhanced_autotune_engine.py vocals.wav melody.mid --log-level DEBUG

# Check system status
python multi_autotune_cli.py --validate-setup
```

## 📈 Performance Optimization

### For Real-time Applications
- Use `profile="realtime"`
- Choose `pydub_speed` or `scipy_manual` methods
- Reduce `quality_priority` to 0.3 or lower
- Disable postprocessing for maximum speed

### For Production Quality
- Use `profile="production"`
- Choose `pyrubberband_shift`, `soundtouch_shift`, or enhanced engine
- Set `quality_priority` to 0.9 or higher
- Enable all preprocessing and postprocessing

### For Research and Analysis
- Use enhanced engine with detailed logging
- Enable comprehensive quality metrics
- Use `scipy_autotune` for intelligent processing
- Save processing reports for analysis

### For Batch Processing
- Use `scipy_autotune` for intelligent processing
- Enable fallback chains for reliability
- Use benchmarking to find optimal settings
- Consider enhanced engine for complex audio

## 📚 API Reference

### Main Function

```python
multi_library_autotune(
    audio_path: str,
    midi_path: str,
    force: float = 0.85,
    pitch_shift_method: str = "auto",
    output_path: Optional[str] = None,
    profile: Optional[str] = None,
    enable_benchmarking: bool = False,
    config: Optional[AutotuneConfig] = None
) -> AutotuneResult
```

### Enhanced Engine Function

```python
enhanced_autotune(
    audio_path: str,
    midi_path: str,
    force: float = 0.85,
    method: str = "auto",
    output_path: Optional[str] = None,
    log_level: str = "INFO"
) -> Dict[str, Any]
```

### Configuration Classes

```python
AutotuneConfig(
    sample_rate: int = 44100,
    hop_length: int = 256,
    frame_length: int = 1024,
    quality_priority: float = 0.7,
    enable_preprocessing: bool = True,
    enable_postprocessing: bool = True,
    fallback_enabled: bool = True,
    quality_threshold: float = 0.8
)
```

### Result Classes

```python
AutotuneResult(
    success: bool,
    output_path: Optional[str],
    method_used: Optional[str],
    processing_time: float,
    quality_metrics: Dict[str, float],
    fallback_methods_tried: List[str],
    error_message: Optional[str]
)
```

## 🎓 Advanced Features

### Enhanced F0 Detection
The enhanced engine provides multiple F0 detection algorithms:
- **YIN Algorithm**: Primary method for clean signals
- **Autocorrelation**: Fallback for challenging audio
- **Spectral Peaks**: Last resort for complex signals
- **Adaptive Cleaning**: Outlier removal and interpolation

### Smart MIDI Processing
- **Automatic Looping**: Extends short MIDI sequences
- **Intelligent Mapping**: Maps audio time to active MIDI notes
- **Complexity Analysis**: Analyzes MIDI characteristics for optimization

### Voice Activity Detection
- **Multi-feature VAD**: RMS, spectral centroid, zero-crossing rate
- **Adaptive Thresholding**: Dynamic threshold based on audio content
- **Temporal Smoothing**: Reduces false positives

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Acknowledgments

- **LibROSA**: Audio analysis and processing
- **Spotify Pedalboard**: Professional audio effects
- **Rubber Band**: High-quality pitch shifting
- **SoundTouch**: Professional audio processing
- **Pretty MIDI**: MIDI file processing
- **SciPy/NumPy**: Scientific computing foundations
- **PyDub**: Simple audio manipulation

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/your-username/midi-autotune-system/issues)
- **Documentation**: Check the [examples](example_complete.py) and [tests](test_soundtouch.py)
- **System Validation**: Run `python multi_autotune_cli.py --validate-setup`
- **Enhanced Engine**: Use `python enhanced_autotune_engine.py --help` for advanced features

## 🔬 Research and Development

This system is designed for both practical use and research purposes. The enhanced engine provides detailed processing reports and quality metrics that can be valuable for:

- **Audio Processing Research**: Comparative analysis of pitch shifting algorithms
- **Music Information Retrieval**: F0 detection accuracy studies
- **Audio Quality Assessment**: Comprehensive quality metrics
- **Real-time Audio Processing**: Performance optimization studies

For research use, enable detailed logging and save processing reports:

```python
# Research-oriented processing
result = enhanced_autotune(
    audio_path="research_audio.wav",
    midi_path="reference_melody.mid",
    method="auto",
    log_level="DEBUG"
)

# Save detailed report for analysis
import json
with open("processing_report.json", "w") as f:
    json.dump(result['processing_report'], f, indent=2)
```

---

**Note**: This system is designed for educational, research, and professional use. For commercial applications, please ensure compliance with all relevant audio processing licenses and patents.
