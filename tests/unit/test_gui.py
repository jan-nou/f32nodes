from unittest.mock import Mock
from f32nodes.gui import GUI


class TestGUI:
    """Test GUI basic functionality without brittle PyQt6 mocking"""

    def test_gui_initialization_with_graph(self, tmp_path):
        """GUI should initialize with a graph reference"""
        mock_graph = Mock()
        mock_graph.nodes = []
        mock_graph.connections = []

        layout_path = tmp_path / "dummy.layout.yaml"
        gui = GUI(mock_graph, str(layout_path))

        # Verify basic initialization
        assert gui.graph is mock_graph
        assert hasattr(gui, "graph")

    def test_gui_has_basic_interface_methods(self, tmp_path):
        """GUI should have expected interface methods"""
        mock_graph = Mock()
        mock_graph.nodes = []
        mock_graph.connections = []

        layout_path = tmp_path / "dummy.layout.yaml"
        gui = GUI(mock_graph, str(layout_path))

        # Check basic interface exists
        assert hasattr(gui, "run")
        assert callable(gui.run)

    def test_run_reuses_existing_qapplication(self, tmp_path, monkeypatch):
        """GUI.run should reuse a pre-existing QApplication instance"""
        mock_graph = Mock()
        mock_graph.nodes = []
        layout_path = tmp_path / "dummy.layout.yaml"

        mock_app = Mock()
        qt_constructor = Mock()
        qt_constructor.instance = Mock(return_value=mock_app)
        monkeypatch.setattr("f32nodes.gui.QApplication", qt_constructor)

        gui = GUI(mock_graph, str(layout_path))

        mock_main_window = Mock()
        mock_main_window.setWindowTitle = Mock()
        mock_main_window.resize = Mock()
        mock_main_window.show = Mock()
        mock_main_window.setCentralWidget = Mock()
        monkeypatch.setattr(
            "f32nodes.gui.QMainWindow", Mock(return_value=mock_main_window)
        )

        mock_scene = Mock()
        mock_scene.setSceneRect = Mock()
        mock_scene.setBackgroundBrush = Mock()
        mock_scene.addItem = Mock()
        mock_scene.itemsBoundingRect = Mock(
            return_value=Mock(isEmpty=Mock(return_value=True))
        )
        monkeypatch.setattr(
            "f32nodes.gui.QGraphicsScene", Mock(return_value=mock_scene)
        )

        mock_view = Mock()
        monkeypatch.setattr(
            "f32nodes.gui.ZoomableGraphicsView", Mock(return_value=mock_view)
        )

        monkeypatch.setattr(
            "f32nodes.gui.AdaptiveNodeGraphics", Mock(return_value=Mock())
        )
        monkeypatch.setattr("f32nodes.gui.GUI.populate_visual_nodes", Mock())
        monkeypatch.setattr("f32nodes.gui.GUI.load_layout", Mock(return_value=True))

        result = gui.run(start_event_loop=False)

        assert result == 0
        qt_constructor.assert_not_called()

    def test_run_creates_qapplication_and_execs_when_requested(
        self, tmp_path, monkeypatch
    ):
        """GUI.run should create QApplication and forward exec result when requested"""

        mock_graph = Mock()
        mock_graph.nodes = []
        layout_path = tmp_path / "dummy.layout.yaml"

        mock_app = Mock()
        mock_app.exec.return_value = 123

        qt_constructor = Mock(return_value=mock_app)
        qt_constructor.instance = Mock(return_value=None)
        monkeypatch.setattr("f32nodes.gui.QApplication", qt_constructor)
        mock_main_window = Mock()
        mock_main_window.setWindowTitle = Mock()
        mock_main_window.resize = Mock()
        mock_main_window.show = Mock()
        mock_main_window.setCentralWidget = Mock()
        monkeypatch.setattr(
            "f32nodes.gui.QMainWindow", Mock(return_value=mock_main_window)
        )

        mock_scene = Mock()
        mock_scene.setSceneRect = Mock()
        mock_scene.setBackgroundBrush = Mock()
        mock_scene.addItem = Mock()
        mock_scene.itemsBoundingRect = Mock(
            return_value=Mock(isEmpty=Mock(return_value=True))
        )
        monkeypatch.setattr(
            "f32nodes.gui.QGraphicsScene", Mock(return_value=mock_scene)
        )

        mock_view = Mock()
        mock_view.pos = Mock()
        monkeypatch.setattr(
            "f32nodes.gui.ZoomableGraphicsView", Mock(return_value=mock_view)
        )

        dummy_graphics_item = Mock()
        dummy_graphics_item.output_ports = {}
        dummy_graphics_item.input_ports = {}
        dummy_graphics_item.output_port_positions = {}
        dummy_graphics_item.input_port_positions = {}
        dummy_graphics_item.connection_lines = []
        dummy_graphics_item.pos.return_value.x.return_value = 0.0
        dummy_graphics_item.pos.return_value.y.return_value = 0.0

        monkeypatch.setattr(
            "f32nodes.gui.AdaptiveNodeGraphics",
            Mock(return_value=dummy_graphics_item),
        )
        monkeypatch.setattr("f32nodes.gui.GUI.populate_visual_nodes", Mock())
        monkeypatch.setattr("f32nodes.gui.GUI.load_layout", Mock(return_value=True))

        gui = GUI(mock_graph, str(layout_path))

        result = gui.run(start_event_loop=True)

        assert result == 123
        mock_app.exec.assert_called_once()
        qt_constructor.assert_called_once()
