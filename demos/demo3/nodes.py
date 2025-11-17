from pathlib import Path
import threading
import time

import cv2
import numpy as np

from f32nodes.core import Node


def _resolve_frame_size(config):
    frame_size = int(config["frame_size"])
    if frame_size <= 0:
        raise ValueError("frame_size must be > 0")
    return frame_size


class AudioFileReader(Node):
    """Loops a mono WAV buffer and optionally plays it through sounddevice."""
    def define_config(self):
        return {
            "file_path": "spectral_fx.wav",
            "frame_size": 1600,
            "enable_playback": True,
        }

    def setup(self):
        import soundfile as sf

        file_path = Path(__file__).resolve().parent / self.config["file_path"]
        if not file_path.exists():
            raise FileNotFoundError(f"Audio file not found: {file_path}")

        self._audio_data, self._sample_rate = sf.read(str(file_path), dtype="float32")

        if len(self._audio_data.shape) > 1:
            self._audio_data = np.mean(self._audio_data, axis=1).astype(np.float32)

        self._frame_size = _resolve_frame_size(self.config)
        self._loop_length = len(self._audio_data)
        self._frame_offsets = np.arange(self._frame_size, dtype=np.int64)

        self._graph_position = 0
        self._playback_position = 0

        self._playback_enabled = bool(self.config.get("enable_playback", True))
        self._playback_thread = None
        self._playback_stop = threading.Event()
        self._chunk_lock = threading.Lock()
        self._last_chunk = None
        self._stream = None
        self._sd = None

        if self._playback_enabled:
            self._start_playback_thread()

    def define_ports(self):
        self.add_output_port(
            "audio_out", shape=(self._frame_size,), min_val=-1.0, max_val=1.0
        )

    def compute(self):
        chunk = None
        if self._playback_enabled:
            with self._chunk_lock:
                if self._last_chunk is not None:
                    chunk = self._last_chunk.copy()

        if chunk is None:
            chunk = self._frame_from_graph_position()

        self.output_ports["audio_out"].set_value(chunk)

    def _frame_from_graph_position(self):
        start = self._graph_position
        frame = np.take(
            self._audio_data,
            self._frame_offsets + start,
            mode="wrap",
        )
        self._graph_position = (start + self._frame_size) % self._loop_length
        return frame

    def _start_playback_thread(self):
        if not self._playback_enabled or self._playback_thread is not None:
            return

        try:
            import sounddevice as sd
        except ImportError:
            print(
                "Audio playback disabled: 'sounddevice' not installed. "
                "Set enable_playback=False to silence this message."
            )
            self._playback_enabled = False
            return

        self._sd = sd

        def playback_loop():
            while not self._playback_stop.is_set():
                if self._stream is None:
                    if not self._open_stream():
                        if self._playback_stop.wait(1.0):
                            break
                        continue

                chunk = self._next_playback_chunk()
                if chunk is None:
                    time.sleep(0.01)
                    continue

                try:
                    self._stream.write(chunk[:, None])
                    with self._chunk_lock:
                        self._last_chunk = chunk.copy()
                except Exception as exc:
                    print(f"Audio playback error: {exc}. Retrying...")
                    self._close_stream()
                    time.sleep(0.1)

        self._playback_thread = threading.Thread(
            target=playback_loop, daemon=True
        )
        self._playback_thread.start()
        print(
            f"Audio playback worker running ({self._sample_rate}Hz, blocksize={self._frame_size})"
        )

    def _open_stream(self):
        if self._sd is None:
            return False

        try:
            stream = self._sd.OutputStream(
                channels=1,
                samplerate=self._sample_rate,
                blocksize=self._frame_size,
                dtype=np.float32,
            )
            stream.start()
            self._stream = stream
            print("Audio playback stream started")
            return True
        except Exception as exc:
            print(
                f"Audio playback unavailable ({exc}). "
                "Will retry; set enable_playback=False in graph.yaml to skip audio."
            )
            self._stream = None
            return False

    def _close_stream(self):
        if self._stream is not None:
            try:
                if self._stream.active:
                    self._stream.stop()
            except Exception:
                pass
            finally:
                self._stream.close()
                self._stream = None
                with self._chunk_lock:
                    self._last_chunk = None

    def _next_playback_chunk(self):
        start = self._playback_position
        chunk = np.take(
            self._audio_data,
            self._frame_offsets + start,
            mode="wrap",
        ).astype(np.float32, copy=False)
        self._playback_position = (start + self._frame_size) % self._loop_length
        return chunk

    def teardown(self):
        if self._playback_thread is not None:
            self._playback_stop.set()
            self._playback_thread.join(timeout=0.5)

        self._close_stream()
        if self._playback_enabled:
            print("Audio playback stopped")


class FFTAnalysis(Node):
    """Transforms audio frames into smoothed octave-band magnitudes."""
    def define_config(self):
        return {
            "fft_size": 4096,
            "sample_rate": 48000,
            "frame_size": 1600,
            "overlap": 0.75,
            "num_bands": 31,
        }

    def setup(self):
        self._fft_size = self.config["fft_size"]
        self._sample_rate = self.config["sample_rate"]
        self._overlap = self.config["overlap"]
        self._num_bands = self.config["num_bands"]

        self._window = np.blackman(self._fft_size).astype(np.float32)
        self._window_correction = 1.0 / np.sqrt(np.mean(self._window**2))

        self._hop_size = _resolve_frame_size(self.config)
        self._buffer = np.zeros(self._fft_size, dtype=np.float32)
        self._buffer_pos = 0

        self._create_octave_bands()

        self._smoothed_bands = np.zeros(self._num_bands, dtype=np.float32)
        self._smooth_factor = 0.3

        print(
            f"FFTAnalysis: {self._sample_rate}Hz -> {self._hop_size} samples/frame (hop size)"
        )

    def _create_octave_bands(self):
        octave_centers = []
        f = 25.0
        for _ in range(self._num_bands):
            octave_centers.append(f)
            f *= 2 ** (1 / 3)

        freq_per_bin = self._sample_rate / self._fft_size
        nyquist_bin = self._fft_size // 2
        self._band_bins = []

        for fc in octave_centers:
            f_lower = fc / (2 ** (1 / 6))
            f_upper = fc * (2 ** (1 / 6))
            bin_lower = max(1, int(np.floor(f_lower / freq_per_bin)))
            bin_upper = min(nyquist_bin, int(np.ceil(f_upper / freq_per_bin)))
            if bin_upper <= bin_lower:
                bin_upper = min(nyquist_bin, bin_lower + 1)
            self._band_bins.append((bin_lower, bin_upper))

    def define_ports(self):
        # Hop size must match the AudioFileReader frame_size configured in graph.yaml.
        self.add_input_port(
            "audio_in", shape=(self._hop_size,), min_val=-1.0, max_val=1.0
        )
        self.add_output_port(
            "spectrum", shape=(self._num_bands,), min_val=-60.0, max_val=10.0
        )

    def compute(self):
        audio_input = self.input_ports["audio_in"].get_value()

        if audio_input is None or len(audio_input) == 0:
            self.output_ports["spectrum"].set_value(self._smoothed_bands.copy())
            return

        input_len = len(audio_input)
        samples_needed = self._fft_size - self._buffer_pos

        if input_len < samples_needed:
            self._buffer[self._buffer_pos : self._buffer_pos + input_len] = audio_input
            self._buffer_pos += input_len
            self.output_ports["spectrum"].set_value(self._smoothed_bands.copy())
            return

        self._buffer[self._buffer_pos :] = audio_input[:samples_needed]

        windowed = self._buffer * self._window
        fft = np.fft.rfft(windowed)
        magnitude = np.abs(fft).astype(np.float32)
        magnitude *= (2.0 / self._fft_size) * self._window_correction

        band_energies = np.zeros(self._num_bands, dtype=np.float32)
        for i, (bin_lower, bin_upper) in enumerate(self._band_bins):
            if bin_upper > bin_lower:
                power = magnitude[bin_lower:bin_upper] ** 2
                band_power = np.mean(power)
                db_value = 10 * np.log10(band_power + 1e-10)
                band_energies[i] = np.clip(db_value, -60.0, 10.0)
            else:
                band_energies[i] = -60.0

        self._smoothed_bands = (
            self._smooth_factor * band_energies
            + (1 - self._smooth_factor) * self._smoothed_bands
        )

        overlap_samples = int(self._fft_size * self._overlap)
        self._buffer[:overlap_samples] = self._buffer[-overlap_samples:]
        self._buffer_pos = overlap_samples

        remaining = input_len - samples_needed
        if remaining > 0:
            copy_amount = min(remaining, self._fft_size - overlap_samples)
            self._buffer[overlap_samples : overlap_samples + copy_amount] = audio_input[
                samples_needed : samples_needed + copy_amount
            ]
            self._buffer_pos += copy_amount

        self.output_ports["spectrum"].set_value(self._smoothed_bands)


class RmsPeakMeter(Node):
    """Tracks RMS and peak envelopes for a streaming audio signal."""
    def define_config(self):
        return {
            "frame_size": 1600,
            "rms_smoothing": 0.2,
            "peak_decay": 0.9,
        }

    def setup(self):
        self._frame_size = _resolve_frame_size(self.config)
        self._alpha = float(self.config.get("rms_smoothing", 0.2))
        self._alpha = np.clip(self._alpha, 0.0, 1.0)
        self._peak_decay = float(self.config.get("peak_decay", 0.9))
        self._peak_decay = np.clip(self._peak_decay, 0.0, 1.0)
        self._rms = np.float32(0.0)
        self._peak = np.float32(0.0)

    def define_ports(self):
        # Shares the same frame size as the audio reader so chunks line up sample-for-sample.
        self.add_input_port(
            "audio_in", shape=(self._frame_size,), min_val=-1.0, max_val=1.0
        )
        self.add_output_port("rms", shape=(), min_val=0.0, max_val=1.0)
        self.add_output_port("peak", shape=(), min_val=0.0, max_val=1.0)

    def compute(self):
        audio = self.input_ports["audio_in"].get_value()

        if audio is None:
            self.output_ports["rms"].set_value(self._rms)
            self.output_ports["peak"].set_value(self._peak)
            return

        audio = audio.astype(np.float32, copy=False)
        rms_value = float(np.sqrt(np.mean(audio * audio)))
        peak_value = float(np.max(np.abs(audio)))

        self._rms = np.float32(self._alpha * rms_value + (1.0 - self._alpha) * self._rms)

        if peak_value >= self._peak:
            self._peak = np.float32(peak_value)
        else:
            self._peak = np.float32(self._peak * self._peak_decay)

        self.output_ports["rms"].set_value(self._rms)
        self.output_ports["peak"].set_value(self._peak)


class SpectrumMapper(Node):
    """Maps FFT band windows into normalized low/mid control scalars."""
    def define_config(self):
        return {
            "num_bands": 31,
            "low_range": (0, 10),
            "mid_range": (10, 20),
            "low_db_min": -60.0,
            "low_db_max": -10.0,
            "mid_db_min": -60.0,
            "mid_db_max": -10.0,
        }

    def setup(self):
        self._num_bands = int(self.config["num_bands"])
        if self._num_bands <= 0:
            raise ValueError("num_bands must be > 0")

        self._low_range = self._validate_band_range(self.config["low_range"])
        self._mid_range = self._validate_band_range(self.config["mid_range"])

        self._low_db_min = float(self.config["low_db_min"])
        self._low_db_max = float(self.config["low_db_max"])
        self._mid_db_min = float(self.config["mid_db_min"])
        self._mid_db_max = float(self.config["mid_db_max"])

        if self._low_db_max <= self._low_db_min:
            raise ValueError("low_db_max must be greater than low_db_min")
        if self._mid_db_max <= self._mid_db_min:
            raise ValueError("mid_db_max must be greater than mid_db_min")

    def _validate_band_range(self, band_range):
        if len(band_range) != 2:
            raise ValueError("Band range must contain exactly two values")
        start, end = int(band_range[0]), int(band_range[1])
        if start < 0 or end < 0:
            raise ValueError("Band range values must be >= 0")
        if end <= start:
            raise ValueError("Band range end must be greater than start")
        if end > self._num_bands:
            raise ValueError("Band range end exceeds available bands")
        return (start, end)

    def define_ports(self):
        self.add_input_port(
            "spectrum", shape=(self._num_bands,), min_val=-60.0, max_val=10.0
        )
        self.add_output_port("low_energy", shape=(), min_val=0.0, max_val=1.0)
        self.add_output_port("mid_energy", shape=(), min_val=0.0, max_val=1.0)

    def compute(self):
        spectrum = self.input_ports["spectrum"].get_value()

        if spectrum is None:
            self.output_ports["low_energy"].set_value(np.float32(0.0))
            self.output_ports["mid_energy"].set_value(np.float32(0.0))
            return

        low_range = self._low_range
        mid_range = self._mid_range
        low_db_min = self._low_db_min
        low_db_max = self._low_db_max
        mid_db_min = self._mid_db_min
        mid_db_max = self._mid_db_max

        low_db = np.mean(spectrum[low_range[0] : low_range[1]])
        mid_db = np.mean(spectrum[mid_range[0] : mid_range[1]])

        low_energy = np.clip((low_db - low_db_min) / (low_db_max - low_db_min), 0.0, 1.0)
        mid_energy = np.clip((mid_db - mid_db_min) / (mid_db_max - mid_db_min), 0.0, 1.0)

        self.output_ports["low_energy"].set_value(np.float32(low_energy))
        self.output_ports["mid_energy"].set_value(np.float32(mid_energy))


class AudioVisualizer(Node):
    """Generates an RGBA texture by mixing background, rectangle, and feedback layers."""
    def define_config(self):
        return {
            "width": 426,
            "height": 240,
            "center_x": 213.0,
            "center_y": 120.0,
            "rect_width": 60.0,
            "rect_height": 60.0,
            "stroke_width": 2.0,
            "stroke_width_min_factor": 0.6,
            "stroke_width_max_factor": 1.4,
            "rect_r": 1.0,
            "rect_g": 0.0,
            "rect_b": 0.0,
            "rect_a": 1.0,
            "feather": 1.0,
            "scale_min": 0.3,
            "scale_max": 2.2,
            "rotation_speed": 0.06,
            "bg_alpha": 1.0,
            "bg_base_r": 0.0,
            "bg_base_g": 0.0,
            "bg_base_b": 0.0,
            "bg_scale_r": 0.35,
            "bg_scale_g": 0.15,
            "bg_scale_b": 0.1,
            "feedback_decay_min": 0.3,
            "feedback_decay_max": 0.98,
            "feedback_energy_power": 1.5,
            "feedback_energy_scale": 3.0,
        }

    def setup(self):
        self._width = int(self.config["width"])
        self._height = int(self.config["height"])
        y, x = np.mgrid[0 : self._height, 0 : self._width]
        self._x = x.astype(np.float32) - float(self.config["center_x"])
        self._y = y.astype(np.float32) - float(self.config["center_y"])

        self._rect_buffer = np.zeros((self._height, self._width, 4), dtype=np.float32)
        self._background = np.zeros_like(self._rect_buffer)
        self._feedback_history = np.zeros_like(self._rect_buffer)
        self._feedback_buffer = np.zeros_like(self._rect_buffer)
        self._composite = np.zeros_like(self._rect_buffer)
        self._rect_canvas = np.zeros_like(self._rect_buffer)
        self._rect_mask = np.zeros((self._height, self._width), dtype=np.float32)

        self._rect_color = np.array(
            [self.config["rect_r"], self.config["rect_g"], self.config["rect_b"]],
            dtype=np.float32,
        )
        self._rect_alpha = float(self.config["rect_a"])
        self._rotation = 0.0
        self._stroke_width = float(self.config["stroke_width"])
        self._stroke_min_factor = float(self.config["stroke_width_min_factor"])
        self._stroke_max_factor = float(self.config["stroke_width_max_factor"])

    def define_ports(self):
        # Three independent scalar controls: background hue, rectangle intensity, echo amount.
        self.add_input_port(
            "background_change",
            shape=(),
            min_val=0.0,
            max_val=1.0,
            default_value=np.float32(0.0),
        )
        self.add_input_port(
            "rectangle_intensity",
            shape=(),
            min_val=0.0,
            max_val=1.0,
            default_value=np.float32(0.0),
        )
        self.add_input_port(
            "delay_effect",
            shape=(),
            min_val=0.0,
            max_val=1.0,
            default_value=np.float32(0.0),
        )
        self.add_output_port(
            "rgba_out",
            shape=(self._height, self._width, 4),
            min_val=0.0,
            max_val=1.0,
        )

    def compute(self):
        bg_change = self.input_ports["background_change"].get_value()
        rect_intensity = self.input_ports["rectangle_intensity"].get_value()
        delay_effect = self.input_ports["delay_effect"].get_value()

        bg_change = np.clip(float(bg_change if bg_change is not None else 0.0), 0.0, 1.0)
        rect_intensity = np.clip(
            float(rect_intensity if rect_intensity is not None else 0.0), 0.0, 1.0
        )
        delay_effect = np.clip(
            float(delay_effect if delay_effect is not None else 0.0), 0.0, 1.0
        )

        self._update_background(bg_change)
        self._render_rectangle(rect_intensity, bg_change)
        self._apply_feedback(delay_effect)
        self._composite_layers()
        self.output_ports["rgba_out"].set_value(self._composite.copy())

    def _update_background(self, amount):
        r = np.clip(
            self.config["bg_base_r"] + amount * self.config["bg_scale_r"], 0.0, 1.0
        )
        g = np.clip(
            self.config["bg_base_g"] + amount * self.config["bg_scale_g"], 0.0, 1.0
        )
        b = np.clip(
            self.config["bg_base_b"] + amount * self.config["bg_scale_b"], 0.0, 1.0
        )
        self._background[..., 0] = r
        self._background[..., 1] = g
        self._background[..., 2] = b
        self._background[..., 3] = self.config["bg_alpha"]

    def _render_rectangle(self, intensity, rotation_control):
        scale = self.config["scale_min"] + intensity * (
            self.config["scale_max"] - self.config["scale_min"]
        )
        scaled_width = max(2.0, self.config["rect_width"] * scale)
        scaled_height = max(2.0, self.config["rect_height"] * scale)

        stroke = max(
            self._stroke_width
            * (
                self._stroke_min_factor
                + intensity * (self._stroke_max_factor - self._stroke_min_factor)
            ),
            0.0,
        )

        rotation_delta = (rotation_control - 0.5) * self.config["rotation_speed"]
        self._rotation = (self._rotation + rotation_delta + np.pi) % (2 * np.pi) - np.pi

        canvas = self._rect_canvas
        mask = self._rect_mask
        canvas.fill(0.0)
        mask.fill(0.0)

        half_w = scaled_width * 0.5
        half_h = scaled_height * 0.5
        cx = float(self.config["center_x"])
        cy = float(self.config["center_y"])
        top_left = (int(round(cx - half_w)), int(round(cy - half_h)))
        bottom_right = (int(round(cx + half_w)), int(round(cy + half_h)))
        cv2.rectangle(mask, top_left, bottom_right, color=1.0, thickness=-1)

        if stroke > 0.0:
            shrink = max(1, int(round(stroke)))
            inner_top = (
                int(round(cx - half_w + shrink)),
                int(round(cy - half_h + shrink)),
            )
            inner_bottom = (
                int(round(cx + half_w - shrink)),
                int(round(cy + half_h - shrink)),
            )
            if inner_bottom[0] > inner_top[0] and inner_bottom[1] > inner_top[1]:
                cv2.rectangle(mask, inner_top, inner_bottom, color=0.0, thickness=-1)

        feather = max(self.config["feather"], 1e-3)
        cv2.GaussianBlur(mask, (0, 0), sigmaX=feather, sigmaY=feather, dst=mask)

        canvas[..., :3] = self._rect_color
        canvas[..., 3] = np.clip(mask * self._rect_alpha, 0.0, 1.0)

        angle_deg = np.degrees(self._rotation)
        rotation_matrix = cv2.getRotationMatrix2D((cx, cy), angle_deg, 1.0)
        cv2.warpAffine(
            canvas,
            rotation_matrix,
            (self._width, self._height),
            dst=self._rect_buffer,
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0.0, 0.0, 0.0, 0.0),
        )

    def _apply_feedback(self, amount):
        energy_shaped = np.power(amount, self.config["feedback_energy_power"])
        energy_shaped *= self.config["feedback_energy_scale"]
        energy_shaped = np.clip(energy_shaped, 0.0, 1.0)

        decay = self.config["feedback_decay_min"] + energy_shaped * (
            self.config["feedback_decay_max"] - self.config["feedback_decay_min"]
        )
        self._feedback_history *= decay

        np.maximum(self._feedback_history, self._rect_buffer, out=self._feedback_buffer)
        self._feedback_history[:] = self._feedback_buffer

    def _composite_layers(self):
        fg = self._feedback_buffer
        bg = self._background
        fg_a = fg[..., 3:4]
        bg_a = bg[..., 3:4]
        inv_fg = 1.0 - fg_a
        self._composite[..., :3] = fg[..., :3] * fg_a + bg[..., :3] * inv_fg
        self._composite[..., 3:4] = np.clip(fg_a + bg_a * inv_fg, 0.0, 1.0)
