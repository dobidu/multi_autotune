#!/usr/bin/env python3
"""
CLI Interface for Multi-Library Autotune System
==============================================

Complete and unambiguous command line interface for the
multi-library autotune system.

Run: python multi_autotune_cli.py --help to see all options
"""

import argparse
import sys
import json
import time
from pathlib import Path
from typing import List, Optional, Dict, Any

# Required system imports
try:
    from multi_autotune import (
        multi_library_autotune,
        MultiLibraryAutotuneProcessor,
        AutotuneConfig,
        validate_system_setup,
        LibraryDetector,
        AutotuneLogger,
        VALID_METHODS,
        METHOD_CATEGORIES
    )
except ImportError as e:
    print(f"❌ Error importing system modules: {e}")
    print("💡 Make sure multi_autotune.py is in the same directory")
    sys.exit(1)

# ===== CLI INTERFACE =====

class MultiAutotuneArgumentParser:
    """Specialized argument parser."""
    
    def __init__(self):
        """Initializes parser."""
        self.parser = self._create_parser()
    
    def _create_parser(self) -> argparse.ArgumentParser:
        """Creates main parser."""
        parser = argparse.ArgumentParser(
            description="🎵 Multi-Library Autotune System v2.0",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog=self._get_examples_text()
        )
        
        # Main arguments
        parser.add_argument(
            "audio_path",
            nargs="?",
            help="Path to audio file (WAV, MP3, OGG, FLAC, M4A)"
        )
        parser.add_argument(
            "midi_path", 
            nargs="?",
            help="Path to MIDI file (.mid)"
        )
        
        # Processing parameters
        processing_group = parser.add_argument_group("Processing Parameters")
        processing_group.add_argument(
            "--force", "-f",
            type=float,
            default=0.85,
            help="Correction intensity (0.0-1.0, default: 0.85)"
        )
        processing_group.add_argument(
            "--method", "-m",
            type=str,
            default="auto",
            help="Pitch shifting method (default: auto)"
        )
        processing_group.add_argument(
            "--output", "-o",
            type=str,
            help="Output file (default: {input}_autotuned_{method}.wav)"
        )
        
        # Predefined profiles
        profile_group = parser.add_argument_group("Predefined Profiles")
        profile_group.add_argument(
            "--profile", "-p",
            choices=["realtime", "production", "broadcast", "research", "creative"],
            help="Predefined processing profile"
        )
        
        # Method selection
        method_group = parser.add_argument_group("Method Selection")
        method_group.add_argument(
            "--list-methods",
            action="store_true",
            help="List available methods and their characteristics"
        )
        method_group.add_argument(
            "--method-info",
            type=str,
            metavar="METHOD",
            help="Show detailed information about a specific method"
        )
        method_group.add_argument(
            "--fallback-chain",
            type=str,
            help="Fallback methods separated by comma (ex: librosa_hifi,pedalboard_shift)"
        )
        
        # Benchmarking and comparison
        benchmark_group = parser.add_argument_group("Benchmarking and Comparison")
        benchmark_group.add_argument(
            "--benchmark",
            action="store_true",
            help="Run benchmark of all available methods"
        )
        benchmark_group.add_argument(
            "--compare-methods",
            type=str,
            help="Compare specific methods (separated by comma)"
        )
        benchmark_group.add_argument(
            "--benchmark-report",
            type=str,
            help="Save benchmark report to JSON file"
        )
        
        # Advanced settings
        advanced_group = parser.add_argument_group("Advanced Settings")
        advanced_group.add_argument(
            "--quality-priority",
            type=float,
            default=0.7,
            help="Quality vs speed priority (0.0-1.0, default: 0.7)"
        )
        advanced_group.add_argument(
            "--quality-threshold",
            type=float,
            default=0.8,
            help="Minimum quality threshold (0.0-1.0, default: 0.8)"
        )
        advanced_group.add_argument(
            "--disable-fallback",
            action="store_true",
            help="Disable automatic fallback system"
        )
        advanced_group.add_argument(
            "--disable-preprocessing",
            action="store_true",
            help="Disable audio preprocessing"
        )
        advanced_group.add_argument(
            "--disable-postprocessing",
            action="store_true",
            help="Disable audio postprocessing"
        )
        
        # Logging settings
        logging_group = parser.add_argument_group("Logging and Debug")
        logging_group.add_argument(
            "--verbose", "-v",
            action="store_true",
            help="Detailed output (DEBUG level)"
        )
        logging_group.add_argument(
            "--quiet", "-q",
            action="store_true",
            help="Minimal output (errors only)"
        )
        logging_group.add_argument(
            "--log-file",
            type=str,
            help="File to save logs"
        )
        
        # Validation and setup
        validation_group = parser.add_argument_group("Validation and Setup")
        validation_group.add_argument(
            "--validate-setup",
            action="store_true",
            help="Validate system configuration"
        )
        validation_group.add_argument(
            "--check-dependencies",
            action="store_true",
            help="Check installed dependencies"
        )
        validation_group.add_argument(
            "--install-missing",
            action="store_true",
            help="Try to install missing dependencies"
        )
        
        return parser
    
    def _get_examples_text(self) -> str:
        """Returns usage examples text."""
        return """
🎯 USAGE EXAMPLES:

Basic usage:
  %(prog)s vocal.wav melody.mid --method librosa_hifi
  %(prog)s vocal.wav melody.mid --method pyrubberband_shift
  %(prog)s vocal.wav melody.mid --method scipy_autotune

Automatic selection:
  %(prog)s vocal.wav melody.mid --method auto --quality-priority 0.9
  %(prog)s vocal.wav melody.mid --method auto --profile production

Custom fallback chain:
  %(prog)s vocal.wav melody.mid --fallback-chain pyrubberband_shift,pedalboard_shift,librosa_hifi

Predefined profiles:
  %(prog)s vocal.wav melody.mid --profile production
  %(prog)s vocal.wav melody.mid --profile realtime
  %(prog)s vocal.wav melody.mid --profile broadcast

Benchmarking:
  %(prog)s vocal.wav melody.mid --benchmark
  %(prog)s vocal.wav melody.mid --compare-methods pyrubberband_shift,pedalboard_shift
  %(prog)s vocal.wav melody.mid --benchmark --benchmark-report results.json

Validation and information:
  %(prog)s --validate-setup
  %(prog)s --list-methods
  %(prog)s --method-info librosa_hifi
  %(prog)s --check-dependencies

Advanced settings:
  %(prog)s vocal.wav melody.mid --quality-priority 0.9 --disable-fallback
  %(prog)s vocal.wav melody.mid --verbose --log-file processing.log

🔧 AVAILABLE METHODS:
  auto                - Intelligent automatic selection
  pydub_speed         - PyDub (fast, low quality)
  pyrubberband_shift  - Rubberband (high quality, requires rubberband-cli)
  librosa_standard    - LibROSA standard (balanced)
  librosa_hifi        - LibROSA high quality
  pedalboard_shift    - Spotify Pedalboard (professional)
  scipy_manual        - SciPy manual (educational/research)
  scipy_autotune      - SciPy complete autotune (intelligent)

📋 AVAILABLE PROFILES:
  realtime    - Maximum speed for real-time processing
  production  - Professional quality for music production
  broadcast   - Broadcast quality with optimized processing
  research    - Configuration for research and analysis
  creative    - Creative effects and experimentation

🆘 SUPPORT:
  %(prog)s --help            # This help
  %(prog)s --validate-setup  # System diagnosis
  %(prog)s --list-methods    # Available methods
        """
    
    def parse_args(self, args: Optional[List[str]] = None) -> argparse.Namespace:
        """Parse arguments with validation."""
        parsed_args = self.parser.parse_args(args)
        
        # Custom validation
        validation_errors = self._validate_parsed_args(parsed_args)
        if validation_errors:
            for error in validation_errors:
                print(f"❌ Error: {error}", file=sys.stderr)
            sys.exit(1)
        
        return parsed_args
    
    def _validate_parsed_args(self, args: argparse.Namespace) -> List[str]:
        """Validates parsed arguments."""
        errors = []
        
        # Check if it's an information command or processing command
        info_commands = [
            args.validate_setup, args.check_dependencies, args.list_methods,
            args.method_info, args.install_missing
        ]
        
        if not any(info_commands):
            # Processing commands need audio and midi
            if not args.audio_path:
                errors.append("Audio file is required for processing")
            if not args.midi_path:
                errors.append("MIDI file is required for processing")
        
        # Validate force
        if not 0.0 <= args.force <= 1.0:
            errors.append(f"Force must be between 0.0 and 1.0, received: {args.force}")
        
        # Validate quality_priority
        if not 0.0 <= args.quality_priority <= 1.0:
            errors.append(f"Quality priority must be between 0.0 and 1.0: {args.quality_priority}")
        
        # Validate quality_threshold
        if not 0.0 <= args.quality_threshold <= 1.0:
            errors.append(f"Quality threshold must be between 0.0 and 1.0: {args.quality_threshold}")
        
        # Validate method
        if args.method and args.method not in VALID_METHODS:
            errors.append(f"Invalid method: {args.method}")
            errors.append(f"Valid methods: {', '.join(VALID_METHODS)}")
        
        # Conflicts between verbose and quiet
        if args.verbose and args.quiet:
            errors.append("Cannot use --verbose and --quiet simultaneously")
        
        return errors

# PART 2

# ===== CLI FUNCTIONS =====

class MultiAutotuneCLI:
    """Main CLI interface."""
    
    def __init__(self):
        """Initialize CLI."""
        self.arg_parser = MultiAutotuneArgumentParser()
        self.processor = None
    
    def run(self, args: Optional[List[str]] = None) -> int:
        """Execute CLI."""
        try:
            parsed_args = self.arg_parser.parse_args(args)
            
            # Configure logging
            self._setup_logging(parsed_args)
            
            # Execute appropriate command
            if parsed_args.validate_setup:
                return self._cmd_validate_setup()
            elif parsed_args.check_dependencies:
                return self._cmd_check_dependencies()
            elif parsed_args.list_methods:
                return self._cmd_list_methods()
            elif parsed_args.method_info:
                return self._cmd_method_info(parsed_args.method_info)
            elif parsed_args.install_missing:
                return self._cmd_install_missing()
            elif parsed_args.benchmark:
                return self._cmd_benchmark(parsed_args)
            elif parsed_args.compare_methods:
                return self._cmd_compare_methods(parsed_args)
            else:
                return self._cmd_process_audio(parsed_args)
                
        except KeyboardInterrupt:
            print("\n⚠️  Operation cancelled by user")
            return 130
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            return 1
    
    def _setup_logging(self, args: argparse.Namespace):
        """Configure logging system."""
        if args.quiet:
            level = "ERROR"
        elif args.verbose:
            level = "DEBUG"
        else:
            level = "INFO"
        
        # Initialize global logger
        AutotuneLogger(level=level, log_file=args.log_file)
    
    def _cmd_validate_setup(self) -> int:
        """Command: validate system configuration."""
        print("🔍 Validating system configuration...")
        
        validation = validate_system_setup()
        
        print(f"\n📊 Overall status: {'✅ OK' if validation['overall_status'] else '❌ PROBLEMS'}")
        print(f"🔧 Available methods: {len(validation['available_methods'])}")
        
        if validation['available_methods']:
            print("\n✅ Functional methods:")
            for method in validation['available_methods']:
                print(f"   • {method['name']:20} "
                      f"(Quality: {method['quality_score']:.1f}, "
                      f"Speed: {method['speed_score']:.1f})")
        
        if validation['missing_dependencies']:
            print(f"\n❌ Missing dependencies ({len(validation['missing_dependencies'])}):")
            for dep in validation['missing_dependencies']:
                external_note = " (requires external software)" if dep.get('requires_external') else ""
                print(f"   • {dep['name']:20} - {dep['installation_notes']}{external_note}")
        
        if validation['warnings']:
            print(f"\n⚠️  Warnings:")
            for warning in validation['warnings']:
                print(f"   • {warning}")
        
        if validation['recommendations']:
            print(f"\n💡 Recommendations:")
            for rec in validation['recommendations']:
                print(f"   • {rec}")
        
        return 0 if validation['overall_status'] else 1
    
    def _cmd_check_dependencies(self) -> int:
        """Command: check dependencies."""
        print("🔍 Checking dependencies...")
        
        methods = LibraryDetector.detect_all_libraries()
        
        available_count = 0
        total_count = 0
        
        for method_name, method_info in methods.items():
            if method_name == "auto":
                continue
                
            total_count += 1
            status = "✅" if method_info.available else "❌"
            external = " (external)" if method_info.requires_external else ""
            
            print(f"{status} {method_info.name:25} {method_info.library_required}{external}")
            
            if method_info.available:
                available_count += 1
            else:
                print(f"     💡 {method_info.installation_notes}")
        
        print(f"\n📊 Summary: {available_count}/{total_count} methods available")
        
        if available_count == 0:
            print("❌ No methods available! Run: python -m pip install librosa soundfile")
            return 1
        elif available_count < total_count // 2:
            print("⚠️  Few methods available. Consider installing more libraries.")
            return 0
        else:
            print("✅ Adequate configuration!")
            return 0
    
    def _cmd_list_methods(self) -> int:
        """Command: list available methods."""
        print("🔧 Available pitch shifting methods:")
        print("=" * 60)
        
        methods = LibraryDetector.detect_all_libraries()
        
        # Organize by category
        categories = {
            "🚀 Optimized for speed": METHOD_CATEGORIES["speed_optimized"],
            "⭐ Optimized for quality": METHOD_CATEGORIES["quality_optimized"],
            "⚖️  Balanced": METHOD_CATEGORIES["balanced"],
            "🤖 Complete autotune": METHOD_CATEGORIES["autotune_complete"]
        }
        
        for category_name, method_list in categories.items():
            print(f"\n{category_name}:")
            
            for method_name in method_list:
                if method_name in methods:
                    method_info = methods[method_name]
                    status = "✅" if method_info.available else "❌"
                    
                    print(f"  {status} {method_name:20} - {method_info.name}")
                    
                    if method_info.available:
                        # Quality and speed stars
                        quality_stars = "★" * int(method_info.quality_score * 5)
                        speed_stars = "★" * int(method_info.speed_score * 5)
                        
                        print(f"       Quality: {quality_stars:<5} ({method_info.quality_score:.1f})")
                        print(f"       Speed: {speed_stars:<5} ({method_info.speed_score:.1f})")
                        
                        if method_info.best_use_cases:
                            use_cases = ", ".join(method_info.best_use_cases)
                            print(f"       Ideal for: {use_cases}")
                    else:
                        print(f"       💡 {method_info.installation_notes}")
        
        print(f"\n🎯 Automatic selection:")
        print(f"  ✨ auto                  - Intelligent selection based on context")
        
        return 0
    
    def _cmd_method_info(self, method_name: str) -> int:
        """Command: information about specific method."""
        methods = LibraryDetector.detect_all_libraries()
        
        if method_name not in methods:
            print(f"❌ Unknown method: {method_name}")
            print(f"💡 Use --list-methods to see available methods")
            return 1
        
        method_info = methods[method_name]
        
        print(f"📋 Detailed information: {method_name}")
        print("=" * 50)
        print(f"Full name: {method_info.name}")
        print(f"Library: {method_info.library_required}")
        print(f"Status: {'✅ Available' if method_info.available else '❌ Not available'}")
        
        if method_info.available:
            print(f"\n📊 Performance metrics:")
            print(f"  Quality: {'★' * int(method_info.quality_score * 5):<5} ({method_info.quality_score:.1f}/1.0)")
            print(f"  Speed: {'★' * int(method_info.speed_score * 5):<5} ({method_info.speed_score:.1f}/1.0)")
            print(f"  Memory efficiency: {'★' * int(method_info.memory_efficiency * 5):<5} ({method_info.memory_efficiency:.1f}/1.0)")
            
            overall_score = method_info.get_overall_score()
            print(f"  Overall score: {'★' * int(overall_score * 5):<5} ({overall_score:.1f}/1.0)")
            
            if method_info.best_use_cases:
                print(f"\n🎯 Ideal use cases:")
                for use_case in method_info.best_use_cases:
                    print(f"  • {use_case}")
            
            if method_info.limitations:
                print(f"\n⚠️  Limitations:")
                for limitation in method_info.limitations:
                    print(f"  • {limitation}")
        
        else:
            print(f"\n💡 Installation:")
            print(f"  {method_info.installation_notes}")
            
            if method_info.requires_external:
                print(f"  ⚠️  Requires additional external software")
        
        return 0
    
    def _cmd_install_missing(self) -> int:
        """Command: install missing dependencies."""
        print("🔧 Installing missing dependencies...")
        
        # Simplified implementation - in production, would use subprocess
        print("💡 To install missing dependencies, run:")
        print()
        
        methods = LibraryDetector.detect_all_libraries()
        missing_libs = set()
        
        for method_info in methods.values():
            if not method_info.available and method_info.library_required:
                missing_libs.add(method_info.library_required)
        
        if missing_libs:
            print(f"pip install {' '.join(missing_libs)}")
            print()
            
            # Special instructions
            if "pyrubberband" in missing_libs:
                print("For pyrubberband, also install rubberband-cli:")
                print("  Ubuntu/Debian: sudo apt-get install rubberband-cli")
                print("  macOS: brew install rubberband")
                print("  Windows: download from https://breakfastquay.com/rubberband/")
        else:
            print("✅ All basic dependencies are installed!")
        
        return 0

# PART 3

    def _cmd_benchmark(self, args: argparse.Namespace) -> int:
        """Command: run benchmark."""
        print("🏃 Running comparative benchmark...")
        
        # Initialize processor
        config = self._build_config_from_args(args)
        self.processor = MultiLibraryAutotuneProcessor(config)
        
        # Run benchmark
        start_time = time.time()
        
        try:
            result = multi_library_autotune(
                audio_path=args.audio_path,
                midi_path=args.midi_path,
                force=args.force,
                enable_benchmarking=True,
                config=config
            )
            
            if result.success:
                benchmark_data = result.quality_metrics.get('benchmark_results', {})
                
                print(f"\n📊 Benchmark results:")
                print("=" * 50)
                
                # Sort by overall score
                sorted_methods = sorted(
                    benchmark_data.items(),
                    key=lambda x: x[1].get('method_info', 0) if x[1]['success'] else 0,
                    reverse=True
                )
                
                for method, data in sorted_methods:
                    if data['success']:
                        time_ms = data['processing_time'] * 1000
                        snr = data['quality_metrics'].get('snr', 0)
                        score = data['method_info']
                        
                        print(f"✅ {method:20} {time_ms:6.1f}ms  SNR:{snr:5.1f}dB  Score:{score:.2f}")
                    else:
                        print(f"❌ {method:20} FAILED - {data.get('error', 'Unknown error')}")
                
                # Save report if requested
                if args.benchmark_report:
                    self._save_benchmark_report(benchmark_data, args.benchmark_report)
                
                total_time = time.time() - start_time
                print(f"\n⏱️  Benchmark completed in {total_time:.1f}s")
                
                return 0
            else:
                print(f"❌ Benchmark error: {result.error_message}")
                return 1
                
        except Exception as e:
            print(f"❌ Benchmark error: {e}")
            return 1
    
    def _cmd_compare_methods(self, args: argparse.Namespace) -> int:
        """Command: compare specific methods."""
        methods_to_compare = [m.strip() for m in args.compare_methods.split(',')]
        
        print(f"⚖️  Comparing methods: {', '.join(methods_to_compare)}")
        
        # Validate methods
        invalid_methods = [m for m in methods_to_compare if m not in VALID_METHODS]
        if invalid_methods:
            print(f"❌ Invalid methods: {invalid_methods}")
            return 1
        
        # Execute comparison
        config = self._build_config_from_args(args)
        results = {}
        
        for method in methods_to_compare:
            print(f"\n🔄 Testing {method}...")
            
            try:
                result = multi_library_autotune(
                    audio_path=args.audio_path,
                    midi_path=args.midi_path,
                    force=args.force,
                    pitch_shift_method=method,
                    config=config
                )
                
                results[method] = result
                
                if result.success:
                    print(f"   ✅ Success in {result.processing_time:.2f}s")
                    if 'snr' in result.quality_metrics:
                        print(f"   📊 SNR: {result.quality_metrics['snr']:.1f} dB")
                else:
                    print(f"   ❌ Failed: {result.error_message}")
                    
            except Exception as e:
                print(f"   ❌ Error: {e}")
                results[method] = None
        
        # Comparative report
        print(f"\n📊 Comparative report:")
        print("=" * 50)
        
        successful_results = [(m, r) for m, r in results.items() if r and r.success]
        successful_results.sort(key=lambda x: x[1].processing_time)
        
        if successful_results:
            print("🏆 Speed ranking:")
            for i, (method, result) in enumerate(successful_results, 1):
                snr = result.quality_metrics.get('snr', 0)
                print(f"{i}. {method:20} {result.processing_time:.2f}s  SNR:{snr:.1f}dB")
        
        return 0 if successful_results else 1
    
    def _cmd_process_audio(self, args: argparse.Namespace) -> int:
        """Command: process audio."""
        print(f"🎵 Processing audio with multi-library system...")
        
        config = self._build_config_from_args(args)
        
        try:
            # Configure method and fallback
            method = args.method
            if args.fallback_chain:
                fallback_methods = [m.strip() for m in args.fallback_chain.split(',')]
                if isinstance(method, str):
                    method = [method] + fallback_methods
            
            # Execute processing
            result = multi_library_autotune(
                audio_path=args.audio_path,
                midi_path=args.midi_path,
                force=args.force,
                pitch_shift_method=method,
                output_path=args.output,
                profile=args.profile,
                config=config
            )
            
            if result.success:
                print(f"\n✅ Processing completed!")
                print(f"   📁 Output file: {result.output_path}")
                print(f"   🔧 Method used: {result.method_used}")
                print(f"   ⏱️  Time: {result.processing_time:.2f}s")
                
                if result.quality_metrics:
                    if 'snr' in result.quality_metrics:
                        print(f"   📊 SNR: {result.quality_metrics['snr']:.1f} dB")
                    if 'correlation' in result.quality_metrics:
# PART 4
                        print(f"   🔗 Correlation: {result.quality_metrics['correlation']:.3f}")
                if result.fallback_methods_tried:
                    print(f"   🔄 Fallbacks tried: {', '.join(result.fallback_methods_tried)}")
                return 0
            else:
                print(f"❌ Processing failed: {result.error_message}")
                if result.fallback_methods_tried:
                    print(f"   🔄 Methods tried: {', '.join(result.fallback_methods_tried)}")
                return 1
                
        except Exception as e:
            print(f"❌ Processing error: {e}")
            return 1

# PART 5
    def _build_config_from_args(self, args: argparse.Namespace) -> AutotuneConfig:
        """Build configuration from arguments."""
        config = AutotuneConfig()
        
        config.quality_priority = args.quality_priority
        config.fallback_enabled = not args.disable_fallback
        config.enable_preprocessing = not args.disable_preprocessing
        config.enable_postprocessing = not args.disable_postprocessing
        config.quality_threshold = args.quality_threshold
        
        return config
    
    def _save_benchmark_report(self, benchmark_data: Dict, output_path: str):
        """Save benchmark report."""
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(benchmark_data, f, indent=2, default=str)
            print(f"💾 Report saved to: {output_path}")
        except Exception as e:
            print(f"❌ Error saving report: {e}")

# ===== MAIN FUNCTION =====

def main():
    """Main CLI function."""
    cli = MultiAutotuneCLI()
    return cli.run()

if __name__ == "__main__":
    sys.exit(main())