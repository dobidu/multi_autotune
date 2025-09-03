#!/usr/bin/env python3
"""
Complete Example of the Multi-Library Autotune System
====================================================

This example demonstrates how to use the multi-library autotune system
with all its advanced features.

Execute: python multi_autotune_example.py
"""

import sys
import time
import json

import pretty_midi
import wavfile
import numpy as np

from pathlib import Path
from os.path import join, isfile, abspath, isdir

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

def build_mid_file_from_vptracker(input_folder: str):
    json_path = join(input_folder, "vptracker.json")
    midi_path = join(input_folder, "vptracker.mid")
    if isfile(midi_path):
        return

    midi_obj = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=0)
    with open(json_path) as f:
        voice_pitch = json.load(f)
        for note in voice_pitch:
            instrument.notes.append(pretty_midi.Note(
                velocity = 80,
                pitch = note["midi_note"],
                start = note["start"],
                end = note["end"]
            ))
        midi_obj.instruments.append(instrument)
        midi_obj.write(midi_path)

def autotune_vocal_track(input_folder: str):
    audio_path = join(input_folder, "vocal.wav")
    midi_path = join(input_folder, "vptracker.mid")
    tuned_vocal_path = join(input_folder, "vocal_tuned.wav")
    if isfile(tuned_vocal_path):
        return

    result = multi_library_autotune(
        audio_path,
        midi_path,
        force=0.7,
        pitch_shift_method="pyrubberband_shift",
        fallback_enabled=False,
        output_path=tuned_vocal_path
    )
    
    if result.success:
        print(f"✅ Success with {result.method_used}")
        print(f"⏱️ Time: {result.processing_time:.2f}s")
        if 'snr' in result.quality_metrics:
            print(f"📊 SNR: {result.quality_metrics['snr']:.1f} dB")
    else:
        raise Exception(result.error_message)

def mix_tuned_vocal_with_instrumental(input_folder: str):
    tuned_vocal_path = join(input_folder, "vocal_tuned.wav")
    instrumental_path = join(input_folder, "instrumental.wav")
    output_path = join(input_folder, "tuned_output.wav")

    with wavfile.open(tuned_vocal_path, 'r') as vocal_f:
        with wavfile.open(instrumental_path, 'r') as instrumental_f:
            output_audio = []
            vocal_audio = vocal_f.read_float()
            instrumental_audio = instrumental_f.read_float()

            for frame in range(instrumental_f.num_frames):
                frame_data = []
                for channel in range(instrumental_f.num_channels):
                    frame_data.append((vocal_audio[frame][0] / 1.4 + instrumental_audio[frame][channel]) / 2)
                output_audio.append(frame_data)

            wavfile.write(
                output_path,
                output_audio,
                sample_rate=44100,
                bits_per_sample=16,
                fmt=wavfile.chunk.WavFormat.PCM,
                metadata=None
            )

def process_folder(input_folder: str):
    try:
        build_mid_file_from_vptracker(input_folder)
        autotune_vocal_track(input_folder)
        mix_tuned_vocal_with_instrumental(input_folder)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def main():
    try:
        target_folder = abspath(sys.argv[1])
        print(target_folder)
        if not isdir(target_folder):
            raise Exception("Target directory doesn't exist.")
        process_folder(target_folder)
        return 0
        
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