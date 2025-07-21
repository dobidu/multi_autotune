#!/usr/bin/env python3
"""
Complete Example of the Multi-Library Autotune System
====================================================

This example demonstrates how to use the multi-library autotune system
with all its advanced features.

Execute: python multi_autotune_example.py
"""

import os
import sys
import time
import json
from pathlib import Path
import numpy as np

# Add system directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Import multi-library system
try:
    from multi_autotune import (
        multi_library_autotune, 
        MultiLibraryAutotuneProcessor,
        AutotuneConfig,
        validate_system_setup,
        VALID_METHODS
    )
    print("✅ Multi-library system imported successfully")
except ImportError as e:
    print(f"❌ Error importing system: {e}")
    print("💡 Make sure all files are in the same directory")
    sys.exit(1)

def create_demo_files():
    """Create demo files."""
    print("📁 Creating demo files...")
    
    try:
        import soundfile as sf
        import pretty_midi
        
        # Create test audio with detuning
        sr = 44100
        duration = 3.0
        t = np.linspace(0, duration, int(sr * duration))
        
        # Sequence of detuned notes
        notes_freqs = [
            (440 * 1.03, 0, 0.8),      # A4 +50 cents
            (493.88 * 0.97, 0.8, 1.6), # B4 -50 cents  
            (523.25 * 1.05, 1.6, 2.4), # C5 +85 cents
            (440 * 0.95, 2.4, 3.0)     # A4 -85 cents
        ]
        
        audio = np.zeros(len(t))
        for freq, start_time, end_time in notes_freqs:
            start_idx = int(start_time * sr)
            end_idx = int(end_time * sr)
            segment_t = t[start_idx:end_idx] - t[start_idx]
            
            # Tone with harmonics and envelope
            tone = (0.6 * np.sin(2 * np.pi * freq * segment_t) +
                   0.3 * np.sin(2 * np.pi * freq * 2 * segment_t) +
                   0.1 * np.sin(2 * np.pi * freq * 3 * segment_t))
            
            # ADSR envelope
            envelope = np.ones_like(tone)
            fade_samples = min(1000, len(tone) // 10)
            if fade_samples > 0:
                envelope[:fade_samples] = np.linspace(0, 1, fade_samples)
                envelope[-fade_samples:] = np.linspace(1, 0, fade_samples)
            
            tone *= envelope
            audio[start_idx:end_idx] = tone
        
        # Add subtle noise
        noise = np.random.normal(0, 0.02, len(audio))
        audio += noise
        
        # Save audio
        sf.write("demo_detuned_vocal.wav", audio, sr)
        print("   ✅ demo_detuned_vocal.wav created")
        
        # Create corresponding MIDI (in tune)
        midi_obj = pretty_midi.PrettyMIDI()
        instrument = pretty_midi.Instrument(program=0)  # Piano
        
        midi_notes = [
            (69, 0.0, 0.8),   # A4
            (71, 0.8, 1.6),   # B4
            (72, 1.6, 2.4),   # C5
            (69, 2.4, 3.0)    # A4
        ]
        
        for pitch, start_time, end_time in midi_notes:
            note = pretty_midi.Note(
                velocity=80,
                pitch=pitch,
                start=start_time,
                end=end_time
            )
            instrument.notes.append(note)
        
        midi_obj.instruments.append(instrument)
        midi_obj.write("demo_perfect_melody.mid")
        print("   ✅ demo_perfect_melody.mid created")
        
        return True
        
    except ImportError as e:
        print(f"   ❌ Error: {e}")
        print("   💡 Run: pip install soundfile pretty_midi")
        return False
    except Exception as e:
        print(f"   ❌ Error creating files: {e}")
        return False

def demonstrate_system_validation():
    """Demonstrate system validation."""
    print("\n🔍 SYSTEM VALIDATION")
    print("=" * 50)
    
    validation = validate_system_setup()
    
    print(f"Overall status: {'✅ OK' if validation['overall_status'] else '❌ PROBLEMS'}")
    print(f"Available methods: {len(validation['available_methods'])}")
    
    if validation['available_methods']:
        print("\nFunctional methods:")
        for method in validation['available_methods']:
            print(f"  • {method['name']:25} Q:{method['quality_score']:.1f} S:{method['speed_score']:.1f}")
    
    if validation['missing_dependencies']:
        print(f"\nMissing dependencies:")
        for dep in validation['missing_dependencies']:
            print(f"  • {dep['name']}: {dep['installation_notes']}")
    
    return validation['overall_status']

def demonstrate_basic_usage():
    """Demonstrate basic system usage."""
    print("\n🎵 BASIC USAGE DEMONSTRATION")
    print("=" * 50)
    
    try:
        # Simplest usage - automatic selection
        print("1. Automatic method selection...")
        
        result = multi_library_autotune(
            audio_path="demo_detuned_vocal.wav",
            midi_path="demo_perfect_melody.mid",
            force=0.8,
            pitch_shift_method="auto"
        )
        
        if result.success:
            print(f"   ✅ Success with {result.method_used}")
            print(f"   ⏱️  Time: {result.processing_time:.2f}s")
            if 'snr' in result.quality_metrics:
                print(f"   📊 SNR: {result.quality_metrics['snr']:.1f} dB")
        else:
            print(f"   ❌ Failed: {result.error_message}")
            return False
        
        return True
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

def demonstrate_method_comparison():
    """Demonstrate method comparison."""
    print("\n⚖️  METHOD COMPARISON")
    print("=" * 50)
    
    # Detect available methods
    processor = MultiLibraryAutotuneProcessor()
    available_methods = processor.get_available_methods()
    
    if not available_methods:
        print("❌ No methods available for comparison")
        return False
    
    print(f"Testing {len(available_methods)} available methods...")
    
    results = {}
    
    # Test each available method
    for method in available_methods[:3]:  # Limit to 3 methods for the example
        print(f"\n🔄 Testing {method}...")
        
        try:
            start_time = time.time()
            
            result = multi_library_autotune(
                audio_path="demo_detuned_vocal.wav",
                midi_path="demo_perfect_melody.mid",
                force=0.8,
                pitch_shift_method=method
            )
            
            processing_time = time.time() - start_time
            
            if result.success:
                results[method] = {
                    'success': True,
                    'time': processing_time,
                    'snr': result.quality_metrics.get('snr', 0),
                    'output': result.output_path
                }
                print(f"   ✅ Success in {processing_time:.2f}s")
                
            else:
                results[method] = {
                    'success': False,
                    'error': result.error_message
                }
                print(f"   ❌ Failed: {result.error_message}")
                
        except Exception as e:
            results[method] = {
                'success': False,
                'error': str(e)
            }
            print(f"   ❌ Error: {e}")
    
    # Comparative report
    print(f"\n📊 COMPARATIVE REPORT")
    print("-" * 30)
    
    successful = [(method, data) for method, data in results.items() if data['success']]
    
    if successful:
        # Sort by speed
        successful.sort(key=lambda x: x[1]['time'])
        
        print("🏃 Speed ranking:")
        for i, (method, data) in enumerate(successful, 1):
            print(f"{i}. {method:20} {data['time']:.2f}s  SNR:{data['snr']:.1f}dB")
        
        # Sort by quality
        successful.sort(key=lambda x: x[1]['snr'], reverse=True)
        
        print("\n🎵 Quality ranking (SNR):")
        for i, (method, data) in enumerate(successful, 1):
            print(f"{i}. {method:20} SNR:{data['snr']:5.1f}dB  {data['time']:.2f}s")
    
    failed = [(method, data) for method, data in results.items() if not data['success']]
    if failed:
        print(f"\n❌ Methods that failed:")
        for method, data in failed:
            print(f"  • {method}: {data['error']}")
    
    return len(successful) > 0

def demonstrate_profiles():
    """Demonstrate use of predefined profiles."""
    print("\n📋 PROFILE DEMONSTRATION")
    print("=" * 50)
    
    profiles_to_test = ["realtime", "production", "research"]
    
    for profile in profiles_to_test:
        print(f"\n🔧 Testing profile '{profile}'...")
        
        try:
            result = multi_library_autotune(
                audio_path="demo_detuned_vocal.wav",
                midi_path="demo_perfect_melody.mid",
                force=0.8,
                profile=profile
            )
            
            if result.success:
                print(f"   ✅ {profile}: {result.method_used} em {result.processing_time:.2f}s")
            else:
                print(f"   ❌ {profile}: {result.error_message}")
                
        except Exception as e:
            print(f"   ❌ {profile}: {e}")

def demonstrate_fallback_system():
    """Demonstrate fallback system."""
    print("\n🔄 FALLBACK SYSTEM DEMONSTRATION")
    print("=" * 50)
    
    # Test with method that may not be available
    print("Testing fallback with unavailable method...")
    
    try:
        # Tenta método que pode não existir
        result = multi_library_autotune(
            audio_path="demo_detuned_vocal.wav",
            midi_path="demo_perfect_melody.mid",
            force=0.8,
            pitch_shift_method="method_that_doesnt_exist"
        )
        
        if result.success:
            print(f"   ✅ Fallback worked: {result.method_used}")
            if result.fallback_methods_tried:
                print(f"   🔄 Methods tried: {', '.join(result.fallback_methods_tried)}")
        else:
            print(f"   ❌ Fallback failed: {result.error_message}")
            
    except Exception as e:
        print(f"   ❌ Error in fallback test: {e}")

def demonstrate_configuration():
    """Demonstrate custom configuration."""
    print("\n⚙️  CUSTOM CONFIGURATION DEMONSTRATION")
    print("=" * 50)
    
    # Custom configuration
    custom_config = AutotuneConfig(
        quality_priority=0.9,  # Prioritize quality
        enable_preprocessing=True,
        enable_postprocessing=True,
        fallback_enabled=True,
        quality_threshold=0.85
    )
    
    print("Testing with custom configuration...")
    
    try:
        result = multi_library_autotune(
            audio_path="demo_detuned_vocal.wav",
            midi_path="demo_perfect_melody.mid",
            force=0.9,
            pitch_shift_method="auto",
            config=custom_config
        )
        
        if result.success:
            print(f"   ✅ Configuração personalizada: {result.method_used}")
            print(f"   ⏱️  Tempo: {result.processing_time:.2f}s")
            print(f"   📊 Qualidade: {result.quality_metrics.get('snr', 0):.1f} dB")
        else:
            print(f"   ❌ Falhou: {result.error_message}")
            
    except Exception as e:
        print(f"   ❌ Erro: {e}")

def demonstrate_benchmarking():
    """Demonstrate benchmarking system."""
    print("\n🏃 BENCHMARKING DEMONSTRATION")
    print("=" * 50)
    
    print("Running automatic benchmark...")
    
    try:
        result = multi_library_autotune(
            audio_path="demo_detuned_vocal.wav",
            midi_path="demo_perfect_melody.mid",
            force=0.8,
            enable_benchmarking=True
        )
        
        if result.success and 'benchmark_results' in result.quality_metrics:
            benchmark_data = result.quality_metrics['benchmark_results']
            
            print(f"\n📊 Benchmark results:")
            print("-" * 40)
            
            # Sort results by score
            sorted_results = []
            for method, data in benchmark_data.items():
                if data['success']:
                    score = data.get('method_info', 0)
                    time_ms = data['processing_time'] * 1000
                    snr = data['quality_metrics'].get('snr', 0)
                    sorted_results.append((score, method, time_ms, snr))
            
            sorted_results.sort(reverse=True)
            
            for i, (score, method, time_ms, snr) in enumerate(sorted_results, 1):
                print(f"{i}. {method:20} Score:{score:.2f} {time_ms:6.1f}ms SNR:{snr:5.1f}dB")
            
            # Save report
            with open("benchmark_report.json", "w") as f:
                json.dump(benchmark_data, f, indent=2, default=str)
            print(f"\n💾 Detailed report saved to: benchmark_report.json")
            
        else:
            print("❌ Benchmark não executado ou falhou")
            
    except Exception as e:
        print(f"❌ Erro no benchmark: {e}")

def demonstrate_advanced_features():
    """Demonstra funcionalidades avançadas."""
    print("\n🚀 FUNCIONALIDADES AVANÇADAS")
    print("=" * 50)
    
    # 1. Processamento com múltiplos métodos de fallback
    print("1. Fallback chain personalizada...")
    
    try:
        result = multi_library_autotune(
            audio_path="demo_detuned_vocal.wav",
            midi_path="demo_perfect_melody.mid",
            force=0.8,
            pitch_shift_method=["librosa_hifi", "librosa_standard", "scipy_manual"]
        )
        
        if result.success:
            print(f"   ✅ Chain executada: {result.method_used}")
        else:
            print(f"   ❌ Chain falhou: {result.error_message}")
    except Exception as e:
        print(f"   ❌ Erro: {e}")
    
            # 2. Detailed analysis of audio characteristics
        print("\n2. Audio characteristics analysis...")
    
    try:
        from multi_autotune import AudioCharacteristics
        import librosa
        
        audio, sr = librosa.load("demo_detuned_vocal.wav", sr=44100)
        characteristics = AudioCharacteristics.analyze_audio(audio, sr)
        
        print(f"   📊 Duration: {characteristics.duration:.2f}s")
        print(f"   📈 Complexity: {characteristics.complexity_score:.3f}")
        print(f"   🔊 Noise level: {characteristics.noise_level:.3f}")
        print(f"   📐 Dynamic range: {characteristics.dynamic_range:.1f} dB")
        print(f"   🎼 Harmonic content: {characteristics.harmonic_content:.3f}")
        
    except Exception as e:
        print(f"   ❌ Analysis error: {e}")

def cleanup_demo_files():
    """Remove demonstration files."""
    demo_files = [
        "demo_detuned_vocal.wav",
        "demo_perfect_melody.mid",
        "benchmark_report.json"
    ]
    
    # Also remove generated output files
    import glob
    output_files = glob.glob("*_autotuned*.wav")
    demo_files.extend(output_files)
    
    print(f"\n🧹 Cleaning up {len(demo_files)} demonstration files...")
    
    removed_count = 0
    for file in demo_files:
        try:
            if os.path.exists(file):
                os.remove(file)
                removed_count += 1
        except Exception:
            pass
    
    print(f"   ✅ {removed_count} files removed")

def print_usage_guide():
    """Print usage guide."""
    print("\n📖 SYSTEM USAGE GUIDE")
    print("=" * 50)
    
    print("""
🎯 BASIC USAGE:
  from multi_autotune import multi_library_autotune
  
  result = multi_library_autotune(
      audio_path="vocal.wav",
      midi_path="melody.mid",
      force=0.85
  )

🔧 METHOD SELECTION:
  # Automatic (recommended)
  method="auto"
  
  # Specific method
  method="librosa_hifi"
  
  # Fallback chain
  method=["pyrubberband_shift", "librosa_hifi", "librosa_standard"]

📋 PREDEFINED PROFILES:
  profile="production"    # High quality
  profile="realtime"      # Maximum speed
  profile="broadcast"     # Broadcast quality
  profile="research"      # For analysis
  profile="creative"      # Creative effects

⚙️  CUSTOM CONFIGURATION:
  config = AutotuneConfig(
      quality_priority=0.9,
      enable_preprocessing=True,
      fallback_enabled=True
  )

🏃 BENCHMARKING:
  result = multi_library_autotune(
      audio_path="vocal.wav",
      midi_path="melody.mid",
      enable_benchmarking=True
  )

💻 COMMAND LINE:
  python multi_autotune_cli.py vocal.wav melody.mid
  python multi_autotune_cli.py --list-methods
  python multi_autotune_cli.py --validate-setup
  python multi_autotune_cli.py vocal.wav melody.mid --benchmark
    """)

def main():
    """Main example function."""
    print("🎵" + "=" * 68 + "🎵")
    print("🎵 COMPLETE DEMONSTRATION OF THE MULTI-LIBRARY AUTOTUNE SYSTEM 🎵")
    print("🎵" + "=" * 68 + "🎵")
    
    print("\nThis example demonstrates all system features:")
    print("• Intelligent automatic method selection")
    print("• Comparison between different libraries")
    print("• Robust fallback system")
    print("• Predefined profiles for different use cases")
    print("• Advanced custom configuration")
    print("• Automatic benchmarking and performance analysis")
    
    try:
        # 1. Validação do sistema
        if not demonstrate_system_validation():
            print("❌ Sistema não está configurado adequadamente!")
            print("💡 Execute: python multi_autotune_cli.py --check-dependencies")
            return 1
        
        # 2. Criação de arquivos de demonstração
        if not create_demo_files():
            print("❌ Could not create demonstration files!")
            return 1
        
        # 3. Demonstrações principais
        demonstrations = [
            ("Uso Básico", demonstrate_basic_usage),
            ("Comparação de Métodos", demonstrate_method_comparison),
            ("Perfis Predefinidos", demonstrate_profiles),
            ("Sistema de Fallback", demonstrate_fallback_system),
            ("Configuração Personalizada", demonstrate_configuration),
            ("Benchmarking", demonstrate_benchmarking),
            ("Funcionalidades Avançadas", demonstrate_advanced_features)
        ]
        
        success_count = 0
        
        for demo_name, demo_func in demonstrations:
            try:
                print(f"\n{'='*20} {demo_name} {'='*20}")
                if demo_func():
                    success_count += 1
                    print(f"✅ {demo_name} concluído com sucesso")
                else:
                    print(f"⚠️  {demo_name} teve problemas")
            except Exception as e:
                print(f"❌ Erro em {demo_name}: {e}")
        
        # 4. Relatório final
        print(f"\n{'='*70}")
        print("📊 RELATÓRIO FINAL DA DEMONSTRAÇÃO")
        print(f"{'='*70}")
        
        print(f"Demonstrações executadas: {len(demonstrations)}")
        print(f"Sucessos: {success_count}")
        print(f"Taxa de sucesso: {success_count/len(demonstrations)*100:.1f}%")
        
        if success_count == len(demonstrations):
            print("\n🎉 TODAS AS DEMONSTRAÇÕES CONCLUÍDAS COM SUCESSO! 🎉")
            print("\n🚀 O sistema está funcionando perfeitamente!")
        elif success_count >= len(demonstrations) * 0.7:
            print("\n✅ Sistema funcionando adequadamente")
            print("⚠️  Algumas funcionalidades podem ter limitações")
        else:
            print("\n⚠️  Sistema com problemas significativos")
            print("🔧 Verifique as dependências e configuração")
        
        # 5. Guia de uso
        print_usage_guide()
        
        # 6. Limpeza (opcional)
        try:
            response = input("\n❓ Remover arquivos de demonstração? (s/N): ").lower()
            if response in ['s', 'sim', 'y', 'yes']:
                cleanup_demo_files()
            else:
                print("📁 Arquivos de demonstração mantidos")
        except (EOFError, KeyboardInterrupt):
            print("\n📁 Arquivos de demonstração mantidos")
        
        print(f"\n🎯 PRÓXIMOS PASSOS:")
        print(f"1. Use python multi_autotune_cli.py --help para ver todas as opções")
        print(f"2. Test with your own audio and MIDI files")
        print(f"3. Experimente diferentes métodos e perfis")
        print(f"4. Execute benchmarks para otimizar sua configuração")
        
        return 0 if success_count >= len(demonstrations) * 0.7 else 1
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Demonstração cancelada pelo usuário")
        return 130
    except Exception as e:
        print(f"\n❌ Erro inesperado: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())