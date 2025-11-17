import math
import time
import numpy as np

from f32nodes.core import Node


class ModulationSignal(Node):
    """Low-frequency scalar sine used as the amplitude control signal."""

    def define_config(self):
        # Defaults can be overridden per instance from graph.yaml.
        return {
            "frequency_hz": 0.6,  # how fast the modulation oscillates
            "amplitude": 0.8,  # cap the modulation range inside [-1, 1]
        }

    def setup(self):
        # setup() runs once before compute(); grab timestamps or spin up helper threads here.
        self._start_time = time.perf_counter()

    def define_ports(self):
        # shape == () marks this as a scalar port; GUI renders it as a gauge.
        self.add_output_port(
            "modulation",
            shape=(),
            min_val=-1.0,
            max_val=1.0,
        )

    def compute(self):
        # compute() is called every frame; emit a normalized sine sample.
        elapsed = time.perf_counter() - self._start_time
        phase = elapsed * self.config["frequency_hz"] * math.tau
        value = math.sin(phase) * self.config["amplitude"]
        # Ports enforce float32 dtype, so wrap plain Python floats before writing.
        self.write_output("modulation", np.float32(value))


class ModulatedWave(Node):
    """Generate a moving waveform whose amplitude follows the modulation input."""

    def define_config(self):
        return {
            "carrier_frequency_hz": 1.6,
            "sample_count": 240,  # length of the 1D trace
            "base_amplitude": 1.0,
        }

    def setup(self):
        self._start_time = time.perf_counter()
        sample_count = self.config["sample_count"]
        # Cache the static phase ramp so compute() does not reallocate every frame.
        self._phase_offset = np.linspace(
            0.0, math.tau, sample_count, endpoint=False, dtype=np.float32
        )

    def define_ports(self):
        # Optional scalar input; default_value serves as the fallback when unconnected.
        self.add_input_port(
            "amplitude",
            shape=(),
            min_val=-1.0,
            max_val=1.0,
            default_value=np.float32(self.config["base_amplitude"]),
        )
        # 1D output so the GUI renders it as an oscilloscope trace.
        self.add_output_port(
            "wave",
            shape=(self.config["sample_count"],),
            min_val=-1.0,
            max_val=1.0,
        )

    def compute(self):
        elapsed = time.perf_counter() - self._start_time
        freq = self.config["carrier_frequency_hz"]
        # read_input() returns numpy scalars; cast to float for math helpers.
        amp_input = float(self.read_input("amplitude"))
        # Map the control input from [-1, 1] to [0, 1] to keep the envelope positive.
        normalized = np.clip((amp_input + 1.0) * 0.5, 0.0, 1.0)
        envelope = np.float32(normalized * self.config["base_amplitude"])

        # Build a static phase ramp across the buffer so the wave scrolls smoothly.
        base_phase = np.float32(elapsed * freq * math.tau)
        phases = base_phase + self._phase_offset

        wave = np.sin(phases) * envelope
        self.write_output("wave", wave.astype(np.float32, copy=False))
