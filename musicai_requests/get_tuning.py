import re
from collections.abc import Collection

import numpy as np
import torch
import torchaudio
from scipy.interpolate import interp1d


class DetuningEstimation(torch.nn.Module):
    """
    Estimates the de-tuning (respective to concert pitch 440Hz) of an audio file containing music.

    Args:
        sample_rate (int): The sample rate of the audio files.
        n_fft (int, optional): The size of the FFT. Defaults to 16384.
        low_midi_pitch (int, optional): The lowest MIDI pitch to consider. Defaults to 24.
        high_midi_pitch (int, optional): The highest MIDI pitch to consider. Defaults to 108.
        spec_filter_size (int, optional): The size of the mean filter applied to the spectrogram. Defaults to 101.
        spec_log_add (float, optional): A constant added to the spectrogram before taking the log. Defaults to 1.
        spec_log_mul (float, optional): A constant multiplied to the spectrogram before taking the log. Defaults to 100.
    """

    def __init__(
        self,
        sample_rate: int,
        n_fft=16384,
        low_midi_pitch=24,
        high_midi_pitch=108,
        spec_filter_size=101,
        spec_log_add=1,
        spec_log_mul=100,
    ):
        super().__init__()
        self.spec = torchaudio.transforms.Spectrogram(n_fft=n_fft, hop_length=n_fft // 2)
        self.spec_log_add = spec_log_add
        self.spec_log_mul = spec_log_mul

        # These will be used to interpolate a cent-level spectrum from the FFT.
        self.fft_freq = fft_frequencies(n_fft, sample_rate)
        self.cent_freq = midi_to_hz(np.arange(low_midi_pitch, high_midi_pitch, 0.01))

        # This is basically a running average filter.
        self.smooth_spec = torch.nn.Conv1d(1, 1, spec_filter_size, bias=False, padding="same")
        self.smooth_spec.weight.data = torch.ones((1, 1, spec_filter_size)) / spec_filter_size

        # This is basically a comb filter with ones every 100 elements, zeros elsewhere.
        self.detect_tuning = torch.nn.Conv1d(1, 1, len(self.cent_freq) // 100, bias=False, padding=50, dilation=100)
        self.detect_tuning.weight.data = torch.ones((1, 1, len(self.cent_freq) // 100))

    def forward(self, audio: torch.Tensor) -> int:
        fft_spec = torch.log(self.spec_log_mul * self.spec(audio) + self.spec_log_add).sum(1)[:-1]

        cent_spec = torch.from_numpy(
            interp1d(self.fft_freq, fft_spec, kind="cubic", fill_value="extrapolate")(self.cent_freq).astype("float32")
        )

        smoothed_cent_spec = self.smooth_spec(cent_spec[None, ...]).squeeze()
        filtered_cent_spec = torch.nn.functional.relu(cent_spec - smoothed_cent_spec)
        tuning_scores = self.detect_tuning(filtered_cent_spec[None, ...]).squeeze()[:100]
        return int(tuning_scores.argmax() - 50)


def fft_frequencies(n_fft: int, sample_rate: int) -> np.ndarray:
    return np.fft.fftfreq(n_fft, 1.0 / sample_rate)[: n_fft // 2]


def logarithmic_frequencies(bins_per_octave: int, f_min: float, f_max: float, f_ref: float = 440.0) -> np.ndarray:
    left = np.floor(np.log2(float(f_min) / f_ref) * bins_per_octave)
    right = np.ceil(np.log2(float(f_max) / f_ref) * bins_per_octave)
    frequencies = f_ref * 2.0 ** (np.arange(left, right) / float(bins_per_octave))
    return frequencies[np.searchsorted(frequencies, f_min) : np.searchsorted(frequencies, f_max, "right")]


def midi_to_hz(notes: int | Collection[int]) -> float | np.ndarray:
    return 440.0 * (2.0 ** ((np.asanyarray(notes) - 69.0) / 12.0))


def hz_to_midi(frequencies: float | Collection[float], f_ref: float = 440.0) -> float | np.ndarray:
    return 12 * (np.log2(np.asanyarray(frequencies)) - np.log2(f_ref)) + 69


def note_to_midi(note: str, round_semitone: bool = True) -> int | float | np.ndarray:
    note_regex = re.compile(r"^(?P<note>[A-Ga-g])(?P<accidental>[#♯𝄪b!♭𝄫♮]*)(?P<octave>[+-]?\d+)?(?P<cents>[+-]\d+)?$")

    if not isinstance(note, str):
        return np.array([note_to_midi(n, round_semitone=round_semitone) for n in note])

    pitch_map = {
        "C": 0,
        "D": 2,
        "E": 4,
        "F": 5,
        "G": 7,
        "A": 9,
        "B": 11,
    }
    acc_map = {
        "#": 1,
        "": 0,
        "b": -1,
        "!": -1,
        "♯": 1,
        "𝄪": 2,
        "♭": -1,
        "𝄫": -2,
        "♮": 0,
    }

    match = note_regex.match(note)

    if not match:
        raise ValueError(f"Improper note format: {note:s}")

    pitch = match.group("note").upper()
    offset = np.sum([acc_map[o] for o in match.group("accidental")])
    octave = match.group("octave")
    cents = match.group("cents")

    octave = int(octave) if octave else 0
    cents = int(cents) * 0.01 if cents else 0
    note_value: float = 12 * (octave + 1) + pitch_map[pitch] + offset + cents

    if round_semitone:
        return int(np.round(note_value))

    return note_value


def note_to_hz(note: str, round_semitone: bool = True) -> float | np.ndarray:
    return midi_to_hz(note_to_midi(note, round_semitone=round_semitone))