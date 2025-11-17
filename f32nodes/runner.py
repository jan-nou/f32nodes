import time
import yaml
import importlib
import os
import sys
import inspect
import numpy as np
from collections import defaultdict, deque
from PyQt6.QtCore import QThread, pyqtSignal, Qt
from f32nodes.gui import GUI
from f32nodes.core import Graph


class Runner(QThread):
    data_ready = pyqtSignal(list)

    def __init__(self, yaml_path):
        super().__init__()
        self.fps = 30

        # Resolve yaml_path relative to caller's script location
        caller_frame = inspect.currentframe().f_back
        caller_file = caller_frame.f_globals.get("__file__")
        if caller_file:
            caller_dir = os.path.dirname(os.path.abspath(caller_file))
            yaml_path = os.path.join(caller_dir, yaml_path)

        # Store YAML directory for relative imports
        self.yaml_dir = os.path.dirname(os.path.abspath(yaml_path))

        # Construct layout file path
        layout_path = yaml_path.replace(".yaml", ".layout.yaml")

        self.graph = self.build_graph_from_yaml(yaml_path)

        self.gui = GUI(self.graph, layout_path)
        # Connect computation directly to GUI updates (blocking to measure time)
        self.data_ready.connect(
            self.gui.update_visualizations, Qt.ConnectionType.BlockingQueuedConnection
        )

        self.currently_behind = False

        self.stats_print_interval = 1000
        self.stats_window_size = 1000
        self.backend_times = deque(maxlen=self.stats_window_size)
        self.gui_times = deque(maxlen=self.stats_window_size)
        self.total_times = deque(maxlen=self.stats_window_size)
        self.node_times = defaultdict(lambda: deque(maxlen=self.stats_window_size))
        self.frames_behind = deque(maxlen=self.stats_window_size)

    def build_graph_from_yaml(self, yaml_path):
        with open(yaml_path, "r") as f:
            config = yaml.safe_load(f)

        # Add YAML directory to sys.path for relative imports
        if self.yaml_dir not in sys.path:
            sys.path.insert(0, self.yaml_dir)

        def resolve_node_type(node_type):
            """Resolve node type string to actual class using paths relative to YAML location"""
            path_prefix, class_name = node_type.split(".", 1)

            paths = config["paths"]

            if path_prefix not in paths:
                raise ValueError(
                    f"Path prefix '{path_prefix}' not found in paths section. Available: {list(paths.keys())}"
                )

            # Get base path (relative to YAML location)
            base_path = paths[path_prefix]

            # Handle nested module paths like "sources.one_d.Wave1D"
            if "." in class_name:
                # Split into module path and actual class name
                module_parts = class_name.rsplit(".", 1)
                module_path = module_parts[0]
                actual_class_name = module_parts[1]

                # Build full import path: base_path + module_path
                full_import_path = f"{base_path}.{module_path}"
                module = importlib.import_module(full_import_path)
                return getattr(module, actual_class_name)
            else:
                # Simple case - class is direct attribute of base module
                import_path = base_path
                module = importlib.import_module(import_path)
                return getattr(module, class_name)

        graph = Graph()

        nodes_config = config.get("nodes", {})
        node_lookup = {}
        for node_name, node_config in nodes_config.items():
            node_type = node_config["type"]

            # Parse type like "demo.Constant"
            if "." not in node_type:
                raise ValueError(
                    f"Node type '{node_type}' must specify module (e.g. 'demo.Constant')"
                )

            node_class = resolve_node_type(node_type)

            # Create node instance with config
            node_cfg = node_config.get("config", {})
            node = node_class(node_name, node_cfg)
            graph.add(node)
            node_lookup[node_name] = node

        connections = config.get("connections", [])
        for connection in connections:
            from_node_name, from_port = connection["from"]
            to_node_name, to_port = connection["to"]

            from_node = node_lookup.get(from_node_name)
            to_node = node_lookup.get(to_node_name)

            if from_node is None:
                raise ValueError(f"Source node '{from_node_name}' not found")
            if to_node is None:
                raise ValueError(f"Target node '{to_node_name}' not found")

            graph.connect(from_node, from_port, to_node, to_port)

        graph.finalize_graph()

        return graph

    def print_statistics(self, frame):
        """Print computation statistics"""
        # Get sample count (same for all measurements)
        n = len(self.backend_times) if len(self.backend_times) > 0 else 0

        print(f"\n{'=' * 60}")
        print(f"STATISTICS @ Frame {frame} (n={n})")
        print(f"{'=' * 60}")

        # Backend computation statistics
        if len(self.backend_times) > 0:
            backend_array = np.array(self.backend_times)
            backend_mean = np.mean(backend_array) * 1000  # Convert to ms
            backend_std = np.std(backend_array) * 1000
            backend_min = np.min(backend_array) * 1000
            backend_max = np.max(backend_array) * 1000
            print(
                f"\nBackend: {backend_mean:.3f}ms ± {backend_std:.3f}ms (min: {backend_min:.3f}ms, max: {backend_max:.3f}ms)"
            )

        # GUI update statistics
        if len(self.gui_times) > 0:
            gui_array = np.array(self.gui_times)
            gui_mean = np.mean(gui_array) * 1000  # Convert to ms
            gui_std = np.std(gui_array) * 1000
            gui_min = np.min(gui_array) * 1000
            gui_max = np.max(gui_array) * 1000
            print(
                f"Gui:     {gui_mean:.3f}ms ± {gui_std:.3f}ms (min: {gui_min:.3f}ms, max: {gui_max:.3f}ms)"
            )

        # Total frame time statistics
        if len(self.total_times) > 0:
            total_array = np.array(self.total_times)
            total_mean = np.mean(total_array) * 1000  # Convert to ms
            total_std = np.std(total_array) * 1000
            total_min = np.min(total_array) * 1000
            total_max = np.max(total_array) * 1000
            print(
                f"Total:   {total_mean:.3f}ms ± {total_std:.3f}ms (min: {total_min:.3f}ms, max: {total_max:.3f}ms)"
            )

        # Frames behind statistics
        if len(self.frames_behind) > 0:
            frames_behind_array = np.array(self.frames_behind)
            times_behind = np.sum(frames_behind_array > 0)
            behind_percentage = (times_behind / len(self.frames_behind)) * 100
            print(f"\n{round(behind_percentage, 2)}% of frames behind schedule")

        # Per-node computation statistics
        if self.node_times:
            print("\nPer-Node Backend Computation:")
            # Sort nodes by mean computation time (descending)
            node_stats = []
            for node_name, times in self.node_times.items():
                if len(times) > 0:
                    times_array = np.array(times)
                    mean_time = np.mean(times_array) * 1000  # Convert to ms
                    std_time = np.std(times_array) * 1000
                    min_time = np.min(times_array) * 1000
                    max_time = np.max(times_array) * 1000
                    node_stats.append(
                        (node_name, mean_time, std_time, min_time, max_time)
                    )

            node_stats.sort(key=lambda x: x[1], reverse=True)

            for node_name, mean_time, std_time, min_time, max_time in node_stats:
                print(
                    f"  {node_name:30s}: {mean_time:8.3f}ms ± {std_time:6.3f}ms (min: {min_time:6.3f}ms, max: {max_time:6.3f}ms)"
                )

        print(f"{'=' * 60}\n")

    def shutdown(self):
        """Stop computation thread gracefully"""
        self.gui.save_layout()
        self.requestInterruption()
        self.wait(1000)  # Wait up to 1 second

    def run(self):
        """Background computation loop - runs in separate thread"""
        # Walltime-based timing to prevent drift
        start_walltime = time.time()
        frame = 0
        frame_interval = 1.0 / self.fps

        while not self.isInterruptionRequested():
            frame_start = time.time()

            try:
                backend_start = time.time()
                port_results, debug_info = self.graph.compute()
                backend_time = time.time() - backend_start

                gui_start = time.time()
                self.data_ready.emit(port_results)
                gui_time = time.time() - gui_start
            except Exception as e:
                print(f"Computation error: {e}")
                break

            total_frame_time = time.time() - frame_start

            # Collect statistics
            self.backend_times.append(backend_time)
            self.gui_times.append(gui_time)
            self.total_times.append(total_frame_time)
            if "node_computation_times" in debug_info:
                for node_name, node_time in debug_info[
                    "node_computation_times"
                ].items():
                    self.node_times[node_name].append(node_time)

            # Print statistics every N frames
            if (frame + 1) % self.stats_print_interval == 0:
                self.print_statistics(frame + 1)

            # Calculate expected frame based on wall time
            current_time = time.time()
            expected_frame = int((current_time - start_walltime) / frame_interval)
            frames_behind_now = expected_frame - frame

            # Track frames behind in rolling window
            self.frames_behind.append(frames_behind_now)

            if frames_behind_now >= 1:
                # Behind schedule - warn on state change, skip sleep to catch up
                if not self.currently_behind:
                    self.currently_behind = True
                    print(
                        f"WARNING: Falling behind schedule ({frames_behind_now} frame(s) behind)"
                    )
                frame += 1
            else:
                # On time - print caught up message on state change, then sleep
                if self.currently_behind:
                    self.currently_behind = False
                    print("Caught up to schedule")
                frame += 1
                target_time = start_walltime + (frame * frame_interval)
                sleep_time = target_time - time.time()
                if sleep_time > 0:
                    time.sleep(sleep_time)

    def start_app(self):
        """Start the application with background computation and responsive GUI"""
        from PyQt6.QtWidgets import QApplication

        # Initialize GUI window (must happen before warmup)
        self.gui.run(start_event_loop=False)

        # Ensure window is actually visible before warmup
        # Process events multiple times to ensure window is shown and rendered
        for _ in range(10):
            QApplication.processEvents()
            time.sleep(0.01)  # Give Qt time to actually show the window

        # Warmup: compute first frame and initialize all GUI widgets before timed loop
        print("Warming up GUI...")
        warmup_start = time.time()
        initial_results, _ = self.graph.compute()
        self.gui.update_visualizations(initial_results)

        # Force multiple event processing cycles to ensure complete painting
        for _ in range(100):
            QApplication.processEvents()
            time.sleep(0.001)

        warmup_time = time.time() - warmup_start
        print(f"GUI warmup completed in {warmup_time:.3f}s")

        # Connect Qt quit signal to graceful shutdown
        self.gui.app.aboutToQuit.connect(self.shutdown)

        # Start background computation (this Runner thread)
        self.start()

        # Run GUI event loop - caller can decide how to handle exit code
        return self.gui.app.exec()
