# methods/vst_plugin_shifter.py
#!/usr/bin/env python3
"""
VST plugin implementation for pitch shifting and autotuning.
"""
import numpy as np
from .base_shifter import BasePitchShifter
import json
from pathlib import Path

class VSTPluginShifter(BasePitchShifter):
    """
    Pitch shifting using a VST3 plugin via Pedalboard.
    """
    def __init__(self, config, vst_path: str = None):
        """
        Initialize with the path to the VST3 plugin, or read from vst_config.json if not provided.
        """
        super().__init__(config)
        if vst_path is None:
            config_path = Path("vst_config.json")
            if config_path.exists():
                with open(config_path, "r", encoding="utf-8") as f:
                    vst_path = json.load(f).get("vst_plugin_path")
        if not vst_path or not vst_path.endswith('.vst3'):
            raise ValueError("A valid path to a .vst3 plugin is required (see vst_config.json).")
        self.vst_path = vst_path

    def shift_pitch(self, audio: np.ndarray, sr: int, semitones: float) -> np.ndarray:
        """Implementation with a VST3 plugin."""
        try:
            import pedalboard
            
            vst_plugin = pedalboard.load_plugin(self.vst_path)
            
            # Tenta configurar o parâmetro de pitch/semitons
            if hasattr(vst_plugin, 'pitch'):
                vst_plugin.pitch = semitones
            elif hasattr(vst_plugin, 'semitones'):
                vst_plugin.semitones = semitones
            else:
                self.logger.warning(f"Could not find a standard pitch/semitones parameter in {self.vst_path}. The VST might use its internal settings.")

            return vst_plugin(audio, sample_rate=sr)
            
        except ImportError:
            raise ImportError("Pedalboard library not found. Please install it with: pip install pedalboard")
        except Exception as e:
            raise RuntimeError(f"VST plugin processing failed: {e}")
