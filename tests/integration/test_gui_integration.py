from unittest.mock import patch, Mock
from f32nodes import Runner
from f32nodes.core import Node
from PyQt6.QtCore import QThread


class TestGUIAutoStart:
    def test_runner_defaults_to_gui_true(self, tmp_path):
        simple_yaml = """
paths:
  test: "tests.integration.test_gui_integration"

nodes:
  input_node:
    type: test.MockInputNode
  output_node:
    type: test.MockProcessNode
    
connections:
  - from: [input_node, "data"]
    to: [output_node, "input_data"]
"""
        yaml_path = tmp_path / "auto_start.yaml"
        yaml_path.write_text(simple_yaml)

        with (
            patch("f32nodes.gui.QApplication") as mock_app,
            patch("f32nodes.gui.QMainWindow") as mock_window,
            patch("f32nodes.gui.QGraphicsScene"),
            patch("f32nodes.gui.QGraphicsView"),
            patch("f32nodes.gui.QGraphicsRectItem"),
            patch("f32nodes.gui.QGraphicsTextItem"),
            patch("f32nodes.gui.QGraphicsPathItem"),
        ):
            mock_app.instance.return_value = None
            runner = Runner(str(yaml_path))
            assert runner.gui is not None
            assert hasattr(runner.gui, "graph")
            assert runner.gui.graph is not None
            runner.gui.run()
            mock_app.assert_called_once()
            mock_window.assert_called_once()
            window_instance = mock_window.return_value
            window_instance.setWindowTitle.assert_called_with(
                "Adaptive Node Graph Visualizer"
            )
            window_instance.show.assert_called_once()

            expected_layout = str(yaml_path).replace(".yaml", ".layout.yaml")
            assert runner.gui.layout_path == expected_layout


class TestNodePopulation:
    def test_gui_populates_visual_nodes_from_graph(self, tmp_path):
        multi_node_yaml = """
paths:
  test: "tests.integration.test_gui_integration"

nodes:
  input_node:
    type: test.MockInputNode
  process_node:
    type: test.MockProcessNode
      
connections:
  - from: [input_node, "data"]
    to: [process_node, "input_data"]
"""
        yaml_path = tmp_path / "node_population.yaml"
        yaml_path.write_text(multi_node_yaml)

        with patch("f32nodes.gui.AdaptiveNodeGraphics") as mock_graphics:
            mock_item = Mock()
            mock_item.name = "mock"
            mock_item.input_ports = {}
            mock_item.output_ports = {}
            mock_item.output_port_positions = {}
            mock_item.input_port_positions = {}
            mock_item.connection_lines = []
            mock_item.pos.return_value = Mock(x=lambda: 0.0, y=lambda: 0.0)
            mock_graphics.return_value = mock_item

            runner = Runner(str(yaml_path))
            gui = runner.gui

            gui.scene = Mock()
            gui.scene.clear = Mock()
            gui.scene.addItem = Mock()
            assert len(gui.graph.nodes) == 2
            gui.populate_visual_nodes()
            assert hasattr(gui, "visual_nodes")
            assert len(gui.visual_nodes) == 2
            assert mock_graphics.call_count == len(gui.graph.nodes)
            for node_name in ["input_node", "process_node"]:
                assert node_name in gui.visual_nodes
            assert hasattr(gui, "visual_connections")
            assert len(gui.visual_connections) == 1
            connection = gui.visual_connections[0]
            assert connection["source_node"].name == "input_node"
            assert connection["target_node"].name == "process_node"
            assert connection["source_port"] == "data"
            assert connection["target_port"] == "input_data"


class TestSingleUpdateLoopIntegration:
    def test_runner_single_loop_drives_gui_updates_synchronously(self, tmp_path):
        simple_yaml = """
paths:
  test: "tests.integration.test_gui_integration"

nodes:
  input_node:
    type: test.MockInputNode
  output_node:
    type: test.MockProcessNode

connections:
  - from: [input_node, "data"]
    to: [output_node, "input_data"]
"""
        yaml_path = tmp_path / "single_loop.yaml"
        yaml_path.write_text(simple_yaml)

        with (
            patch("f32nodes.gui.QApplication"),
            patch("f32nodes.gui.QMainWindow"),
            patch("f32nodes.gui.QGraphicsScene"),
            patch("f32nodes.gui.QGraphicsView"),
            patch("f32nodes.gui.QGraphicsRectItem"),
            patch("f32nodes.gui.QGraphicsTextItem"),
            patch("f32nodes.gui.QGraphicsPathItem"),
        ):
            from f32nodes.gui import QApplication as QtApp

            QtApp.instance.return_value = None
            runner = Runner(str(yaml_path))

            assert isinstance(runner, QThread)
            port_results, debug_info = runner.graph.compute()
            runner.gui.update_visualizations(port_results)
            assert hasattr(runner.gui, "update_visualizations")
            assert len(port_results) > 0
            assert runner.graph is not None


class MockNode(Node):
    def define_config(self):
        return {}

    def define_ports(self):
        pass

    def compute(self):
        pass


class MockInputNode(Node):
    def define_config(self):
        return {}

    def define_ports(self):
        self.add_output_port("data", shape=(5,), min_val=0.0, max_val=10.0)

    def compute(self):
        import numpy as np

        self.output_ports["data"].set_value(
            np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
        )


class MockProcessNode(Node):
    def define_config(self):
        return {}

    def define_ports(self):
        self.add_input_port("input_data", shape=(5,), min_val=0.0, max_val=10.0)
        self.add_output_port("result", shape=(5,), min_val=0.0, max_val=20.0)

    def compute(self):
        input_data = self.input_ports["input_data"].get_value()
        if input_data is not None:
            result = input_data * 2
            self.output_ports["result"].set_value(result)
