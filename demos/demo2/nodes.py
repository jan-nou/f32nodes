import time
import numpy as np

from f32nodes.core import Node


def clamp_signed(array):
    """Clamp any intermediate buffer to [-1, 1] before shipping to ports."""
    return np.clip(array, -1.0, 1.0).astype(np.float32)


def clamp_unit(array):
    """Clamp to [0, 1] for density/alpha style data."""
    return np.clip(array, 0.0, 1.0).astype(np.float32)


class LatticeDriver(Node):
    def define_config(self):
        return {"phase_rate": 0.29, "drift_rate": 0.12}

    def setup(self):
        self.start_time = time.perf_counter()
        # Pre-build a normalized sample ramp so we can blend multiple sine bands each frame.
        self.samples = np.linspace(0.0, 1.0, 256, dtype=np.float32)

    def define_ports(self):
        self.add_output_port("phase_wave", shape=(256,), min_val=-1.0, max_val=1.0)
        self.add_output_port("drift_scalar", shape=(), min_val=0.0, max_val=1.0)

    def compute(self):
        elapsed = time.perf_counter() - self.start_time
        phase = (elapsed * self.config["phase_rate"]) % 1.0
        drift = (elapsed * self.config["drift_rate"]) % 1.0

        base = np.sin((self.samples * 2.0 + phase) * 2.0 * np.pi)
        overlay = np.cos((self.samples * 6.0 + drift) * 2.0 * np.pi) * 0.5
        phase_wave = base * 0.7 + overlay

        drift_scalar = np.float32(0.5 + 0.5 * np.sin(drift * 2.0 * np.pi))

        self.write_output("phase_wave", clamp_signed(phase_wave))
        self.write_output("drift_scalar", drift_scalar)


class MirageField(Node):
    def define_config(self):
        return {"resolution": 200}

    def setup(self):
        res = self.config["resolution"]
        coords = np.linspace(-1.0, 1.0, res, dtype=np.float32)
        # Cache a square lattice in [-1, 1] so we can reuse it for every frame.
        self.grid_x, self.grid_y = np.meshgrid(coords, coords)

    def define_ports(self):
        res = self.config["resolution"]
        self.add_input_port("drift_scalar", shape=(), min_val=0.0, max_val=1.0)
        self.add_input_port("phase_wave", shape=(256,), min_val=-1.0, max_val=1.0)

        self.add_output_port("mirage_map", shape=(res, res), min_val=-1.0, max_val=1.0)
        self.add_output_port("density_map", shape=(res, res), min_val=0.0, max_val=1.0)

    def compute(self):
        phase_wave = self.read_input("phase_wave")
        drift_scalar = float(self.read_input("drift_scalar"))

        # Sample the 1D phase wave twice (offset + rolled) to create two orthogonal fields.
        wave_x = np.interp(
            ((self.grid_x + 1.0) * 0.5) * (phase_wave.size - 1),
            np.arange(phase_wave.size, dtype=np.float32),
            phase_wave,
        )
        wave_y = np.interp(
            ((self.grid_y + 1.0) * 0.5) * (phase_wave.size - 1),
            np.arange(phase_wave.size, dtype=np.float32),
            np.roll(phase_wave, 64),
        )

        checker = np.sin((self.grid_x + drift_scalar) * 6.0 * np.pi) * np.sin(
            (self.grid_y - drift_scalar) * 6.0 * np.pi
        )
        mirage = clamp_signed(wave_x * 0.5 + wave_y * 0.3 + checker * 0.2)
        density = clamp_unit(np.abs(mirage) * 0.7 + (wave_x * 0.5 + 0.5) * 0.3)

        self.write_output("mirage_map", mirage)
        self.write_output("density_map", density)


class MirageDisplay(Node):
    def define_config(self):
        return {"resolution": 200}

    def define_ports(self):
        res = self.config["resolution"]
        self.add_input_port("density_map", shape=(res, res), min_val=0.0, max_val=1.0)
        self.add_input_port("mirage_map", shape=(res, res), min_val=-1.0, max_val=1.0)

        self.add_output_port(
            "mirage_rgba", shape=(res, res, 4), min_val=0.0, max_val=1.0
        )

    def compute(self):
        mirage_map = self.read_input("mirage_map")
        density_map = self.read_input("density_map")

        # Simple three-tone palette to highlight high/low areas and a soft alpha ramp.
        height_norm = (mirage_map + 1.0) * 0.5
        highlight = clamp_unit(np.abs(mirage_map) * 0.4 + density_map * 0.3)
        midtone = clamp_unit(height_norm * 0.5 + density_map * 0.5)
        shadow = clamp_unit((1.0 - height_norm) * 0.6 + (1.0 - density_map) * 0.3)
        alpha = np.float32(np.clip(0.45 + np.mean(density_map) * 0.35, 0.0, 1.0))

        rgba = np.stack(
            [
                highlight,
                midtone,
                shadow,
                np.full_like(mirage_map, alpha, dtype=np.float32),
            ],
            axis=-1,
        )

        self.write_output("mirage_rgba", clamp_unit(rgba))
