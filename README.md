# f32nodes

[![PyPI](https://img.shields.io/pypi/v/f32nodes.svg)](https://pypi.org/project/f32nodes/)
[![Python Versions](https://img.shields.io/pypi/pyversions/f32nodes.svg)](https://pypi.org/project/f32nodes/)
[![License](https://img.shields.io/badge/license-Apache_2.0-blue.svg)](LICENSE)
[![CI](https://github.com/jan-nou/f32nodes/actions/workflows/ci.yml/badge.svg)](https://github.com/jan-nou/f32nodes/actions/workflows/ci.yml)

## Overview

f32nodes integrates a visual programming workflow with plain Python modules into a live-computation environment tuned for multi-modal real-time data streams.

### Core Features
- **Python-native nodes**: each node is a plain Python subclass representing one step of computation. Nodes declare typed ports and exchange `numpy.float32` tensors.
- **YAML graph spec**: node instances, configs, and connections sit in a YAML file; the loader resolves import prefixes relative to that file and wires ports automatically.
- **Strict graph & port contracts**: preflight rejects unconnected nodes plus shape/range/cycle errors, enforces a single connected DAG, and port writes continue to enforce dtype, shape, and limits during execution.
- **Zero-setup visuals**: the strict port contracts drive automatic gauges, waveforms, heatmaps, or RGBA renderers per output shape—with renderer-only normalization and no UI setup required.
- **Performance-minded UI**: the PyQt graph view avoids per-frame allocation churn so visualizations stay responsive alongside the runtime.
- **Runtime instrumentation**: the runner targets a configurable FPS, records backend/GUI/frame timings, samples per-node costs, and flags when the schedule drifts.
- **Persistent layout**: the PyQt6 view stores node positions and viewport state per graph file so the workspace survives restarts.

### Visualization Semantics
Renderer-only normalization: visual widgets resample or scale data for display, but the numpy buffers on ports stay untouched.

- Scalar ports (`shape == ()`) render as bar gauges between declared bounds.
- 1D ports (`shape == (N,)`) render as waveforms after resampling to viewport width.
- 2D ports (`shape == (H, W)`) appear as grayscale heatmaps.
- RGBA ports (`shape == (H, W, 4)`) display as color textures.

### Scope Boundaries
- No bundled node library; domain-specific nodes live alongside each project.
- Graph execution is single-threaded within one Python process. Nodes may spawn their own threads, but the runtime does not orchestrate multi-threaded or distributed scheduling.
- Ports standardize on `numpy.float32` scalars and tensors for consistent interoperability; other dtypes require conversion at graph edges.
- The runner provides best-effort real-time: it reports drift and timings but cannot enforce hard deadlines, and overall throughput is ultimately limited by Python/PyQt rather than native C++ speeds.

## Getting Started

### Install
Requirements: Python `>=3.12`, pip 23+, and a working Qt/OpenGL stack (installed automatically through `PyQt6` on most platforms).

Install from PyPI:

```bash
pip install f32nodes
```

## Demo Gallery

Every demo uses the same setup workflow; from the repo root:

```bash
cd demos/<demo_name>
python3 -m venv .venv; source .venv/bin/activate
pip install -r requirements.txt
python3 main.py
```

### Demo 1 – Amplitude Modulation

A low-frequency scalar modulates a waveform generator. The entry point for learning the node API. All files in this demo are heavily commented to illustrate the full flow.

- `nodes.py` – defines the modulator and carrier nodes.
- `graph.yaml` – initializes & configures nodes, then connects the graph.
- `main.py` – loads and runs the graph.

![Demo 1 screenshot](demos/demo1/demo1.gif)

### Demo 2 – Wavefields

Coupled oscillators drive a drifting 2D height map, while the display node remixes the field into density, height, and a stylized RGBA texture.

![Demo 2 screenshot](demos/demo2/demo2.gif)

### Demo 3 – Audio Nodes

Streams the bundled `spectral_fx.wav`, computes octave-band energy plus RMS, and drives an `AudioVisualizer` node whose three abstract controls (“background change”, “rectangle intensity”, “delay effect”) accept any scalar signal you patch into them. `soundfile` is required for decoding; `sounddevice` enables optional playback (toggle via `enable_playback`). 

![Demo 3 screenshot](demos/demo3/demo3.gif)

## Development
- **Tests**: run `pytest`.
- **Linting**: run `ruff check .`.
- **Packaging**: the project ships with a Poetry `pyproject.toml`; use `poetry install` if you prefer Poetry over raw pip.
- **Editable install**:

  ```bash
  python -m venv .venv
  source .venv/bin/activate
  pip install -U pip
  pip install -e .
  ```

## License & Support
Licensed under the [Apache License 2.0](LICENSE).
Support is best effort; please open an issue on GitHub and I’ll respond as time allows.

## Related Python Node Frameworks
- [Ryven](https://github.com/leon-thomm/Ryven) — visual scripting environment for Python that emphasises interactive, dynamically typed nodes within a Qt interface.
- [PyFlow](https://github.com/wonderworks-software/PyFlow) — general-purpose visual scripting system modelled after Unreal’s Blueprints, featuring a plug-in architecture and broad node catalog.
- [Nodezator](https://github.com/IndiePython/nodezator) — node-based Pygame application that streamlines prototyping workflows for games and creative coding projects.
