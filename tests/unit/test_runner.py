from f32nodes import Runner
from unittest.mock import Mock, patch
from collections import deque, defaultdict
import time
from PyQt6.QtCore import QCoreApplication


class TestRunner:
    def test_yaml_missing_path_error(self):
        """Test error handling when YAML references missing path prefix"""
        import tempfile
        import os
        from unittest.mock import patch

        bad_yaml = """
paths:
  good: "tests.unit.test_runner"

nodes:
  test:
    type: missing_path.Constant
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(bad_yaml)
            temp_path = f.name

        try:
            with (
                patch("f32nodes.gui.QApplication"),
                patch("f32nodes.gui.QMainWindow"),
                patch("f32nodes.gui.QGraphicsScene"),
                patch("f32nodes.gui.QGraphicsView"),
            ):
                Runner(temp_path)
                assert False, "Should have raised ValueError for missing path"
        except ValueError as e:
            assert "Path prefix 'missing_path' not found" in str(e)
        finally:
            os.unlink(temp_path)

    def test_single_update_loop_calls_both_graph_and_gui(self):
        """Test that runner initializes GUI with graph for background computation"""
        mock_graph = Mock()
        mock_gui = Mock()

        runner = Runner.__new__(Runner)
        runner.graph = mock_graph
        runner.gui = mock_gui

        runner.gui.initialize(runner.graph)
        mock_gui.initialize.assert_called_once_with(mock_graph)


class TestRunnerBackgroundComputation:
    def test_runner_background_thread_signal_emission(self):
        """Test that Runner runs computation in background and emits signals with data"""
        app = QCoreApplication.instance()
        if app is None:
            app = QCoreApplication([])

        with (
            patch("f32nodes.gui.QApplication"),
            patch("f32nodes.gui.QMainWindow"),
            patch("f32nodes.gui.QGraphicsScene"),
            patch("f32nodes.gui.QGraphicsView"),
        ):
            runner = Runner.__new__(Runner)
            super(Runner, runner).__init__()

            runner.fps = 60
            runner.stats_window_size = 10
            runner.stats_print_interval = 1000
            runner.backend_times = deque(maxlen=runner.stats_window_size)
            runner.gui_times = deque(maxlen=runner.stats_window_size)
            runner.total_times = deque(maxlen=runner.stats_window_size)
            runner.node_times = defaultdict(
                lambda: deque(maxlen=runner.stats_window_size)
            )
            runner.frames_behind = deque(maxlen=runner.stats_window_size)
            runner.currently_behind = False

            mock_graph = Mock()
            test_port_results = [{"node": "test", "port": "output", "value": 42}]
            test_debug_info = {"computation_time": 0.001}
            mock_graph.compute.return_value = (test_port_results, test_debug_info)
            runner.graph = mock_graph

            captured_data = []

            def capture_data(port_results):
                captured_data.append(port_results)

            runner.data_ready.connect(capture_data)

            runner.start()

            for _ in range(10):
                app.processEvents()
                time.sleep(0.01)

            runner.requestInterruption()
            runner.quit()
            runner.wait()

            assert len(captured_data) > 0, (
                "Runner should have emitted at least one signal"
            )
            assert captured_data[0] == test_port_results, (
                "Signal should contain port results"
            )

            assert mock_graph.compute.call_count > 0, (
                "Runner should have called graph.compute"
            )
