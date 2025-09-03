# methods/soundtouch_shifter.py
#!/usr/bin/env python3
"""
SoundTouch implementation using the official 'soundstretch' command-line tool.
This is the most robust method, avoiding Python wrapper issues.
"""
import numpy as np
import soundfile as sf
import subprocess
import tempfile
import os
from pathlib import Path
from .base_shifter import BasePitchShifter

class SoundTouchShifter(BasePitchShifter):
    """
    Pitch shifting by calling the 'soundstretch' command-line executable.
    """
    def __init__(self, config):
        super().__init__(config)
        # Verifica se o 'soundstretch' está disponível no sistema
        self.soundstretch_path = self._find_soundstretch()
        if not self.soundstretch_path:
            raise FileNotFoundError(
                "The 'soundstretch' executable was not found in your system's PATH. "
                "Please install the SoundTouch library command-line tools."
            )

    def _find_soundstretch(self) -> str:
        """Finds the path to the soundstretch executable."""
        # Tenta encontrar no PATH do sistema
        for path in os.environ.get("PATH", "").split(os.pathsep):
            for exe in ["soundstretch", "soundstretch.exe"]:
                exe_path = Path(path) / exe
                if exe_path.is_file():
                    return str(exe_path)
        return None

    def shift_pitch(self, audio: np.ndarray, sr: int, semitones: float) -> np.ndarray:
        """
        Implementation by calling the soundstretch command-line tool.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir_path = Path(temp_dir)
            
            # 1. Criar arquivos temporários para entrada e saída
            input_path = temp_dir_path / "input.wav"
            output_path = temp_dir_path / "output.wav"
            
            # Salva o áudio de entrada
            sf.write(input_path, audio, sr)
            
            # 2. Monta o comando para o soundstretch
            # -pitch=<semitones> : Aplica o pitch shift
            # -quick : Usa um modo mais rápido (bom para vocais)
            # -naa : Desativa o anti-aliasing para um som menos "filtrado"
            command = [
                self.soundstretch_path,
                str(input_path),
                str(output_path),
                f"-pitch={semitones:.2f}",
                "-quick",
                "-naa"
            ]
            
            try:
                # 3. Executa o comando
                result = subprocess.run(
                    command,
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=60  # Timeout de 60 segundos
                )
                
                # 4. Carrega o áudio processado
                if output_path.exists():
                    processed_audio, _ = sf.read(output_path)
                    return processed_audio
                else:
                    raise RuntimeError(f"soundstretch execution succeeded but no output file was created. Stderr: {result.stderr}")

            except FileNotFoundError:
                raise RuntimeError("The 'soundstretch' command is not available. Please install it.")
            except subprocess.TimeoutExpired:
                raise RuntimeError("soundstretch processing timed out.")
            except subprocess.CalledProcessError as e:
                # O erro do soundstretch vai para o stderr
                raise RuntimeError(f"soundstretch failed with error: {e.stderr}")
