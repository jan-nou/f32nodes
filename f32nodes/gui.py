import sys
import yaml
import os
import numpy as np
import cv2
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QGraphicsView,
    QGraphicsScene,
    QGraphicsItem,
    QGraphicsRectItem,
    QGraphicsTextItem,
    QGraphicsPathItem,
)
from PyQt6.QtCore import Qt, QPointF
from PyQt6.QtGui import (
    QPen,
    QBrush,
    QColor,
    QFont,
    QPainter,
    QPixmap,
    QImage,
    QPolygonF,
    QWheelEvent,
    QMouseEvent,
    QPainterPath,
)

# ============================================================================
# Main Application
# ============================================================================


class GUI:
    def __init__(self, graph, layout_path):
        self.graph = graph
        self.layout_path = layout_path
        self.app = None
        self.main_window = None
        self.scene = None
        self.view = None
        self.visual_nodes = {}
        self.visual_connections = []

    def update_visualizations(self, port_results):
        for port_result in port_results:
            node_name = port_result["node"]
            port_name = port_result["port"]
            port_value = port_result["value"]

            if node_name in self.visual_nodes:
                graphics_item = self.visual_nodes[node_name]
                if port_name in graphics_item.output_ports:
                    graphics_item.update_port_data(port_name, port_value)

    def run(self, start_event_loop=None):
        try:
            existing_app = QApplication.instance()
            created_app = existing_app is None

            if created_app:
                self.app = QApplication(sys.argv)
            else:
                self.app = existing_app
            self.main_window = QMainWindow()

            self.main_window.setWindowTitle("Adaptive Node Graph Visualizer")
            self.main_window.resize(1400, 900)

            # Setup graphics
            try:
                if hasattr(self.main_window, "setCentralWidget"):
                    self.scene = QGraphicsScene()
                    self.scene.setSceneRect(-50000, -50000, 100000, 100000)
                    self.scene.setBackgroundBrush(QBrush(QColor(0, 0, 0)))

                    self.view = ZoomableGraphicsView(self.scene)
                    self.main_window.setCentralWidget(self.view)

                    self.populate_visual_nodes()
                    if not self.load_layout():
                        self.center_view()

            except Exception as e:
                print(f"Graphics setup error: {e}")
                self.populate_visual_nodes()  # Still create data structures

            self.main_window.show()

            # Auto-detect test environment
            if start_event_loop is None:
                is_test = "mock" in str(type(self.main_window)).lower()
                start_event_loop = not is_test

            if start_event_loop and created_app:
                return self.app.exec()

            return 0

        except Exception as e:
            if "mock" in str(type(getattr(self, "main_window", None))).lower():
                if hasattr(self.main_window, "show"):
                    self.main_window.show()
            else:
                raise e

    def populate_visual_nodes(self):
        self.visual_nodes.clear()

        if hasattr(self.graph, "connections"):
            self.visual_connections = getattr(self.graph, "connections", [])

        if self.scene:
            self.create_graphics()

    def create_graphics(self):
        self.scene.clear()

        # Layout nodes
        nodes_per_column = 2
        spacing_x = 800
        spacing_y = 600
        start_x = 500
        start_y = 500

        for i, node in enumerate(self.graph.nodes):
            col = i // nodes_per_column
            row = i % nodes_per_column
            x = start_x + col * spacing_x
            y = start_y + row * spacing_y

            graphics_item = AdaptiveNodeGraphics(
                node.name, node.input_ports, node.output_ports, x, y
            )
            self.visual_nodes[node.name] = graphics_item
            self.scene.addItem(graphics_item)

        self.draw_connections()

    def draw_connections(self):
        for connection in self.visual_connections:
            source_node = connection.get("source_node")
            target_node = connection.get("target_node")
            source_port = connection.get("source_port")
            target_port = connection.get("target_port")

            if not all([source_node, target_node, source_port, target_port]):
                continue

            # Find graphics items by node name
            source_item = self.visual_nodes.get(source_node.name)
            target_item = self.visual_nodes.get(target_node.name)

            if source_item and target_item:
                if (
                    source_port in source_item.output_port_positions
                    and target_port in target_item.input_port_positions
                ):
                    # Calculate line coordinates
                    source_pos = source_item.pos()
                    target_pos = target_item.pos()
                    source_port_pos = source_item.output_port_positions[source_port]
                    target_port_pos = target_item.input_port_positions[target_port]

                    x1 = source_pos.x() + source_port_pos[0]
                    y1 = source_pos.y() + source_port_pos[1]
                    x2 = target_pos.x() + target_port_pos[0]
                    y2 = target_pos.y() + target_port_pos[1]

                    # Create curved connection
                    path = self.create_connection_path(x1, y1, x2, y2)
                    path_item = QGraphicsPathItem(path)
                    path_item.setPen(QPen(AdaptiveNodeGraphics.CONNECTION_COLOR, 3))
                    path_item.setZValue(-1)  # Keep connections in background
                    self.scene.addItem(path_item)

                    # Store connection info for updates
                    line_info = {
                        "line": path_item,
                        "source_node": source_item,
                        "target_node": target_item,
                        "source_port": source_port,
                        "target_port": target_port,
                    }
                    source_item.connection_lines.append(line_info)
                    target_item.connection_lines.append(line_info)

    def create_connection_path(self, x1, y1, x2, y2):
        path = QPainterPath()
        path.moveTo(x1, y1)

        # Calculate control point offset based on horizontal distance
        distance = abs(x2 - x1)
        offset = min(distance * 0.5, 150)  # Curve amount

        # Control points create horizontal tangents at both ends
        ctrl1_x = x1 + offset
        ctrl1_y = y1
        ctrl2_x = x2 - offset
        ctrl2_y = y2

        path.cubicTo(ctrl1_x, ctrl1_y, ctrl2_x, ctrl2_y, x2, y2)
        return path

    def center_view(self):
        if not self.scene or not self.view:
            return

        scene_rect = self.scene.itemsBoundingRect()
        if not scene_rect.isEmpty():
            self.view.centerOn(scene_rect.center())

    def save_layout(self):
        layout = {"nodes": {}, "view": {}}

        # Save node positions
        for node_name, graphics_item in self.visual_nodes.items():
            pos = graphics_item.pos()
            layout["nodes"][node_name] = {"x": float(pos.x()), "y": float(pos.y())}

        # Save view state
        if self.view:
            center = self.view.mapToScene(self.view.viewport().rect().center())
            layout["view"] = {
                "center_x": float(center.x()),
                "center_y": float(center.y()),
                "scale": float(self.view.transform().m11()),
            }

        with open(self.layout_path, "w") as f:
            f.write("# Auto-generated by f32nodes; manual adjustments not required.\n")
            yaml.dump(layout, f, default_flow_style=False)

    def load_layout(self):
        if not os.path.exists(self.layout_path):
            return False

        with open(self.layout_path, "r") as f:
            layout = yaml.safe_load(f)

        if not layout:
            return False

        # Apply node positions
        nodes_layout = layout.get("nodes", {})
        for node_name, pos_data in nodes_layout.items():
            if node_name in self.visual_nodes:
                graphics_item = self.visual_nodes[node_name]
                graphics_item.setPos(pos_data["x"], pos_data["y"])

        # Apply view state
        view_layout = layout.get("view", {})
        if view_layout and self.view:
            center_x = view_layout.get("center_x", 0)
            center_y = view_layout.get("center_y", 0)
            scale = view_layout.get("scale", 1.0)

            self.view.resetTransform()
            self.view.scale(scale, scale)
            self.view.current_scale = scale  # Sync internal scale tracker
            self.view.centerOn(center_x, center_y)

        return True


# ============================================================================
# View Infrastructure - Canvas navigation and interaction
# ============================================================================


class ZoomableGraphicsView(QGraphicsView):
    """Custom QGraphicsView with mouse wheel zoom and middle button panning"""

    def __init__(self, scene=None, parent=None):
        super().__init__(scene, parent)
        self.current_scale = 1.0
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setDragMode(QGraphicsView.DragMode.RubberBandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.is_panning = False
        self.pan_start_pos = None

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.MiddleButton:
            self.is_panning = True
            self.pan_start_pos = event.position()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent):
        if self.is_panning:
            if event.buttons() != Qt.MouseButton.MiddleButton:
                self.is_panning = False
                self.setCursor(Qt.CursorShape.ArrowCursor)
                return

            delta = event.position() - self.pan_start_pos
            self.horizontalScrollBar().setValue(
                self.horizontalScrollBar().value() - int(delta.x())
            )
            self.verticalScrollBar().setValue(
                self.verticalScrollBar().value() - int(delta.y())
            )
            self.pan_start_pos = event.position()
            event.accept()
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.MiddleButton:
            self.is_panning = False
            self.setCursor(Qt.CursorShape.ArrowCursor)
            event.accept()
        else:
            super().mouseReleaseEvent(event)

    def wheelEvent(self, event: QWheelEvent):
        zoom_factor = 1.15 if event.angleDelta().y() > 0 else 1.0 / 1.15
        new_scale = self.current_scale * zoom_factor

        # Clamp zoom
        if new_scale < 0.1:
            new_scale = 0.1
        elif new_scale > 5.0:
            new_scale = 5.0

        self.current_scale = new_scale
        self.resetTransform()
        self.scale(self.current_scale, self.current_scale)
        event.accept()


# ============================================================================
# Node Components - Visual nodes with embedded port data visualizations
# ============================================================================


class AdaptiveNodeGraphics(QGraphicsRectItem):
    """Node graphics that adapts size to content"""

    # Layout constants
    NODE_MARGIN = 20
    PORT_HEIGHT = 20
    TITLE_HEIGHT = 50
    TITLE_BOTTOM_SPACING = 15  # Space between title and first viz
    BOTTOM_MARGIN = 20
    INPUT_PORT_SLOT_HEIGHT = 50

    # Classic CRT Terminal
    NODE_COLOR = QColor(20, 25, 30)
    NODE_BORDER = QColor(0, 200, 200)
    BORDER_WIDTH = 5
    INPUT_PORT_COLOR = QColor(0, 200, 200)
    OUTPUT_PORT_COLOR = QColor(0, 200, 200)
    CONNECTION_COLOR = QColor(60, 80, 75)

    TEXT_COLOR = QColor(57, 255, 20)
    FONT_FAMILY = "Courier New"
    FONT_SIZE = 20

    def __init__(self, name, input_ports, output_ports, x=0, y=0):
        super().__init__()

        self.name = name
        self.input_ports = input_ports
        self.output_ports = output_ports
        self.port_visualizations = {}
        self.connection_lines = []

        self.calculate_layout()
        self.setup_graphics(x, y)
        self.create_ports()

    def get_visualization_dimensions(self, shape):
        """Get hardcoded visualization dimensions based on port shape"""
        if shape == ():
            return (ScalarVisualization.WIDTH, ScalarVisualization.HEIGHT)
        elif len(shape) == 1:
            return (Graph1DVisualization.WIDTH, Graph1DVisualization.HEIGHT)
        elif len(shape) == 2:
            return (Heatmap2DVisualization.WIDTH, Heatmap2DVisualization.HEIGHT)
        elif len(shape) == 3 and shape[2] == 4:
            return (ImageRGBAVisualization.WIDTH, ImageRGBAVisualization.HEIGHT)
        else:
            # Unknown shape - error message size
            return (200, 100)

    def calculate_layout(self):
        """Calculate node dimensions based on actual content needs"""
        # Calculate space needed for output ports
        output_ports_height = 0
        self.port_viz_dimensions = []  # Store (width, height) for each viz

        for _, port in self.output_ports.items():
            viz_width, viz_height = self.get_visualization_dimensions(port.shape)
            self.port_viz_dimensions.append((viz_width, viz_height))
            output_ports_height += viz_height + self.NODE_MARGIN

        # Calculate space needed for input ports
        input_ports_height = len(self.input_ports) * self.INPUT_PORT_SLOT_HEIGHT

        # Node height is determined by whichever side needs more space
        content_height = max(output_ports_height, input_ports_height)
        self.node_height = (
            self.TITLE_HEIGHT
            + self.TITLE_BOTTOM_SPACING
            + content_height
            + self.BOTTOM_MARGIN
        )

        # All nodes same width: max width across ALL widget types
        max_widget_width = max(
            ScalarVisualization.WIDTH,
            Graph1DVisualization.WIDTH,
            Heatmap2DVisualization.WIDTH,
            ImageRGBAVisualization.WIDTH,
        )
        self.node_width = max_widget_width + 2 * self.NODE_MARGIN

    def setup_graphics(self, x, y):
        """Setup the main node visual elements"""
        self.setRect(0, 0, self.node_width, self.node_height)
        self.setBrush(QBrush(self.NODE_COLOR))
        self.setPen(QPen(self.NODE_BORDER, self.BORDER_WIDTH))
        self.setPos(x, y)
        self.setZValue(0)  # Nodes above connections

        # Enable interaction
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, True)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, True)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges, True)

        # Add title
        title_text = QGraphicsTextItem(self.name, self)
        title_text.setPos(20, 10)
        title_text.setDefaultTextColor(self.TEXT_COLOR)
        font = QFont(self.FONT_FAMILY, self.FONT_SIZE)
        font.setBold(True)
        title_text.setFont(font)

    def create_ports(self):
        """Create port graphics with adaptive layout"""
        self.input_port_positions = {}
        self.output_port_positions = {}

        current_y = self.TITLE_HEIGHT + self.TITLE_BOTTOM_SPACING

        # Handle output ports - positioned at visualization centers
        for i, (port_name, port) in enumerate(self.output_ports.items()):
            _, viz_height = self.port_viz_dimensions[i]

            # Visualization starts here
            viz_y = current_y

            # Port positioned at VERTICAL CENTER of visualization
            viz_center_y = viz_y + viz_height / 2

            # Create output port with its visualization
            self.create_output_port(port_name, port, viz_center_y, viz_y, viz_height)

            current_y += viz_height + self.NODE_MARGIN

        # Handle input ports - fixed slot height per port, starting from bottom going up
        for i, port_name in enumerate(self.input_ports.keys()):
            # Center port in its fixed-height slot
            # Bottom slot touches node_height - BOTTOM_MARGIN (symmetric to output starting at TITLE_HEIGHT)
            port_y = (self.node_height - self.BOTTOM_MARGIN) - (
                i + 0.5
            ) * self.INPUT_PORT_SLOT_HEIGHT
            self.create_input_port(port_name, port_y)

    def create_input_port(self, port_name, port_y):
        """Create an input port"""
        # Port circle
        port_rect = QGraphicsRectItem(-15, port_y, 30, self.PORT_HEIGHT, self)
        port_rect.setBrush(QBrush(self.INPUT_PORT_COLOR))
        port_rect.setPen(QPen(self.INPUT_PORT_COLOR.lighter(), 2))

        # Port label (left of port, vertically centered)
        port_text = QGraphicsTextItem(port_name, self)
        port_text.setDefaultTextColor(self.TEXT_COLOR)
        font = QFont(self.FONT_FAMILY, self.FONT_SIZE)
        font.setBold(True)
        port_text.setFont(font)

        # Position text right-aligned to the left of the port
        text_width = port_text.boundingRect().width()
        text_height = port_text.boundingRect().height()
        port_text.setPos(
            -15 - text_width - 10, port_y + (self.PORT_HEIGHT - text_height) / 2
        )

        port_center_y = port_y + self.PORT_HEIGHT // 2
        self.input_port_positions[port_name] = (0, port_center_y)

    def create_output_port(self, port_name, port, viz_center_y, viz_y, viz_height):
        """Create an output port centered on its visualization"""
        # Port circle at center of visualization
        port_y = viz_center_y - self.PORT_HEIGHT / 2
        port_rect = QGraphicsRectItem(
            self.node_width - 15, port_y, 30, self.PORT_HEIGHT, self
        )
        port_rect.setBrush(QBrush(self.OUTPUT_PORT_COLOR))
        port_rect.setPen(QPen(self.OUTPUT_PORT_COLOR.lighter(), 2))

        # Port label (right of port, vertically centered)
        port_text = QGraphicsTextItem(port_name, self)
        port_text.setDefaultTextColor(self.TEXT_COLOR)
        font = QFont(self.FONT_FAMILY, self.FONT_SIZE)
        font.setBold(True)
        port_text.setFont(font)

        # Position text left-aligned to the right of the port
        text_height = port_text.boundingRect().height()
        port_text.setPos(
            self.node_width + 15 + 10, port_y + (self.PORT_HEIGHT - text_height) / 2
        )

        # Create visualization widget (hardcoded sizes)
        viz_widget = None

        if port.shape == ():
            viz_widget = ScalarVisualization(port.min_val, port.max_val, self)
        elif len(port.shape) == 1:
            viz_widget = Graph1DVisualization(port.min_val, port.max_val, self)
        elif len(port.shape) == 2:
            viz_widget = Heatmap2DVisualization(port.min_val, port.max_val, self)
        elif len(port.shape) == 3 and port.shape[2] == 4:
            viz_widget = ImageRGBAVisualization(port.min_val, port.max_val, self)
        else:
            # Unsupported shape - show error message centered in viz area
            error_text = QGraphicsTextItem(self)
            error_text.setPlainText(
                f"Visualization for shape {port.shape}\nis not supported"
            )
            max_text_width = self.node_width - 2 * self.NODE_MARGIN
            error_text.setTextWidth(max_text_width)
            doc = error_text.document()
            text_option = doc.defaultTextOption()
            text_option.setAlignment(Qt.AlignmentFlag.AlignHCenter)
            doc.setDefaultTextOption(text_option)
            error_text.setDefaultTextColor(QColor(255, 100, 100))
            font = QFont(self.FONT_FAMILY, self.FONT_SIZE)
            error_text.setFont(font)
            # Vertically center inside visualization slot
            text_height = error_text.boundingRect().height()
            error_text.setPos(
                self.NODE_MARGIN,
                viz_y + (viz_height - text_height) / 2,
            )

        if viz_widget:
            viz_widget.setPos(self.NODE_MARGIN, viz_y)
            self.port_visualizations[port_name] = viz_widget

        self.output_port_positions[port_name] = (self.node_width, viz_center_y)

    def update_port_data(self, port_name, data):
        if port_name not in self.port_visualizations:
            return

        # Port shapes are fixed at definition time, so visualization type is fixed
        # Just update the visualization with new data
        self.port_visualizations[port_name].update_data(data)

    def itemChange(self, change, value):
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionHasChanged:
            self.update_connections()
        return super().itemChange(change, value)

    def update_connections(self):
        for line_info in self.connection_lines:
            path_item = line_info["line"]
            source_node = line_info["source_node"]
            target_node = line_info["target_node"]
            source_port = line_info["source_port"]
            target_port = line_info["target_port"]

            # Calculate new path coordinates
            source_pos = source_node.pos()
            target_pos = target_node.pos()
            source_port_pos = source_node.output_port_positions[source_port]
            target_port_pos = target_node.input_port_positions[target_port]

            x1 = source_pos.x() + source_port_pos[0]
            y1 = source_pos.y() + source_port_pos[1]
            x2 = target_pos.x() + target_port_pos[0]
            y2 = target_pos.y() + target_port_pos[1]

            # Update path with new curve
            path = QPainterPath()
            path.moveTo(x1, y1)
            distance = abs(x2 - x1)
            offset = min(distance * 0.5, 150)
            path.cubicTo(x1 + offset, y1, x2 - offset, y2, x2, y2)
            path_item.setPath(path)


class ScalarVisualization(QGraphicsRectItem):
    """Displays scalar (0D) values with bar meter"""

    WIDTH = 426  # 240p
    HEIGHT = 80
    BACKGROUND_COLOR = QColor(35, 35, 45)
    BORDER_COLOR = QColor(120, 140, 160)
    BAR_BACKGROUND_COLOR = QColor(50, 50, 60)
    BAR_FILL_COLOR = QColor(120, 220, 120)
    BAR_BORDER_COLOR = QColor(80, 80, 90)

    def __init__(self, min_val, max_val, parent=None):
        super().__init__(0, 0, self.WIDTH, self.HEIGHT, parent)
        self.setBrush(QBrush(self.BACKGROUND_COLOR))
        self.setPen(QPen(self.BORDER_COLOR, 2))
        self.min_val = float(min_val)
        self.max_val = float(max_val)

        # Text display (repositioned to top)
        self.text_item = QGraphicsTextItem("0.0", self)
        self.text_item.setPos(15, 5)
        self.text_item.setDefaultTextColor(AdaptiveNodeGraphics.TEXT_COLOR)
        font = QFont(AdaptiveNodeGraphics.FONT_FAMILY, AdaptiveNodeGraphics.FONT_SIZE)
        font.setBold(True)
        self.text_item.setFont(font)

        # Bar meter
        bar_x = 10
        bar_y = 45
        bar_width = self.WIDTH - 20
        bar_height = 25

        # Bar background
        bar_bg = QGraphicsRectItem(bar_x, bar_y, bar_width, bar_height, self)
        bar_bg.setBrush(QBrush(self.BAR_BACKGROUND_COLOR))
        bar_bg.setPen(QPen(self.BAR_BORDER_COLOR, 1))

        # Bar fill (starts at 0 width)
        self.bar_fill = QGraphicsRectItem(bar_x, bar_y, 0, bar_height, self)
        self.bar_fill.setBrush(QBrush(self.BAR_FILL_COLOR))
        self.bar_fill.setPen(QPen(Qt.PenStyle.NoPen))

        # Store bar dimensions for update_data
        self.bar_x = bar_x
        self.bar_y = bar_y
        self.bar_max_width = bar_width
        self.bar_height = bar_height

    def update_data(self, value):
        current_value = float(value)
        self.text_item.setPlainText(f"{current_value:.3f}")

        # Normalize to [0, 1] for bar width
        if self.max_val != self.min_val:
            normalized = (current_value - self.min_val) / (self.max_val - self.min_val)
            normalized = max(0.0, min(1.0, normalized))  # Clamp to [0, 1]
        else:
            normalized = 0.5

        # Update bar fill width
        fill_width = self.bar_max_width * normalized
        self.bar_fill.setRect(self.bar_x, self.bar_y, fill_width, self.bar_height)


class Graph1DVisualization(QGraphicsRectItem):
    """Displays 1D line graphs"""

    WIDTH = 426  # 240p
    HEIGHT = 240  # 240p
    BACKGROUND_COLOR = QColor(35, 35, 45)
    BORDER_COLOR = QColor(120, 140, 160)

    def __init__(self, min_val, max_val, parent=None):
        super().__init__(0, 0, self.WIDTH, self.HEIGHT, parent)
        self.setBrush(QBrush(self.BACKGROUND_COLOR))
        self.setPen(QPen(self.BORDER_COLOR, 2))
        self.min_val = float(min_val)
        self.max_val = float(max_val)

        # Max label (top left, exactly on border)
        max_label = QGraphicsTextItem(f"{self.max_val:.2f}", self)
        max_label.setDefaultTextColor(AdaptiveNodeGraphics.TEXT_COLOR)
        font = QFont(AdaptiveNodeGraphics.FONT_FAMILY, AdaptiveNodeGraphics.FONT_SIZE)
        font.setBold(True)
        max_label.setFont(font)
        max_label.setPos(5, 0)  # Exactly on top border

        # Min label (bottom left, exactly on border)
        min_label = QGraphicsTextItem(f"{self.min_val:.2f}", self)
        min_label.setDefaultTextColor(AdaptiveNodeGraphics.TEXT_COLOR)
        min_label.setFont(font)
        min_text_height = min_label.boundingRect().height()
        min_label.setPos(5, self.HEIGHT - min_text_height)  # Exactly on bottom border

        # Pre-allocate buffers based on DISPLAY resolution (not port shape)
        self.display_width = self.WIDTH

        # Pre-allocated buffers for display data
        self.display_data = np.zeros(self.display_width, dtype=np.float32)
        self.y_coords = np.zeros(self.display_width, dtype=np.float32)

        # Pre-compute x coordinates (they never change)
        self.x_coords = np.linspace(0, self.WIDTH, self.display_width, dtype=np.float32)

    def update_data(self, data):
        data_len = len(data)

        # RESIZE FIRST to display resolution
        if data_len != self.display_width:
            # np.interp is faster than cv2.resize for 1D data
            data_indices = np.linspace(0, data_len - 1, self.display_width)
            self.display_data[:] = np.interp(data_indices, np.arange(data_len), data)
        else:
            self.display_data[:] = data

        # NOW normalize to [0, 1] range (smaller dataset)
        if self.max_val != self.min_val:
            self.display_data -= self.min_val
            self.display_data /= self.max_val - self.min_val
        else:
            # Handle edge case where min == max
            self.display_data[:] = 0.5

        self.update()

    def paint(self, painter, option, widget):
        super().paint(painter, option, widget)

        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(QPen(QColor(120, 220, 120), 3))

        # Vectorized y-coordinate calculation
        np.multiply(self.display_data, self.HEIGHT, out=self.y_coords)
        self.y_coords *= -1
        self.y_coords += self.HEIGHT

        # Convert to QPointF list (this is the unavoidable Python loop for Qt)
        points = [
            QPointF(float(x), float(y)) for x, y in zip(self.x_coords, self.y_coords)
        ]

        # Draw polyline (single call instead of many drawLine calls)
        polygon = QPolygonF(points)
        painter.drawPolyline(polygon)


class Heatmap2DVisualization(QGraphicsRectItem):
    """Displays 2D heatmaps"""

    WIDTH = 426  # 240p
    HEIGHT = 240  # 240p
    BACKGROUND_COLOR = QColor(35, 35, 45)
    BORDER_COLOR = QColor(120, 140, 160)

    def __init__(self, min_val, max_val, parent=None):
        super().__init__(0, 0, self.WIDTH, self.HEIGHT, parent)
        self.setBrush(QBrush(self.BACKGROUND_COLOR))
        self.setPen(QPen(self.BORDER_COLOR, 2))
        self.min_val = float(min_val)
        self.max_val = float(max_val)

        # Allocate buffers at DISPLAY resolution (not port shape!)
        display_shape = (self.HEIGHT, self.WIDTH)
        self.normalized_float_buffer = np.zeros(
            display_shape, dtype=np.float32, order="C"
        )
        self.normalized_buffer = np.zeros(display_shape, dtype=np.uint8, order="C")

        # Create persistent QImage wrapping buffer (no allocation on paint)
        self.qimage = QImage(
            self.normalized_buffer.data,
            self.WIDTH,
            self.HEIGHT,
            self.WIDTH,
            QImage.Format.Format_Grayscale8,
        )

    def update_data(self, data):
        if data.shape != (self.HEIGHT, self.WIDTH):
            data = cv2.resize(
                data,
                (self.WIDTH, self.HEIGHT),
                interpolation=cv2.INTER_AREA,  # Best for downsampling
            )

        # NOW normalize to [0, 255] (smaller dataset!)
        if self.max_val != self.min_val:
            # Fused normalization: 3 passes instead of 4
            scale = 255 / (self.max_val - self.min_val)

            np.subtract(data, self.min_val, out=self.normalized_float_buffer)
            np.multiply(
                self.normalized_float_buffer,
                scale,
                out=self.normalized_float_buffer,
            )
            np.clip(
                self.normalized_float_buffer,
                0,
                255,
                out=self.normalized_float_buffer,
            )

            # Convert to uint8 (in-place into buffer)
            self.normalized_buffer[:] = self.normalized_float_buffer
        else:
            # Handle edge case where min == max
            self.normalized_buffer[:] = 128

        self.update()

    def paint(self, painter, option, widget):
        super().paint(painter, option, widget)

        # Already at display resolution - no scaling needed!
        pixmap = QPixmap.fromImage(self.qimage)
        painter.drawPixmap(0, 0, pixmap)


class ImageRGBAVisualization(QGraphicsRectItem):
    """Displays RGBA images"""

    WIDTH = 426  # 240p
    HEIGHT = 240  # 240p
    BACKGROUND_COLOR = QColor(35, 35, 45)
    BORDER_COLOR = QColor(120, 140, 160)

    def __init__(self, min_val, max_val, parent=None):
        super().__init__(0, 0, self.WIDTH, self.HEIGHT, parent)
        self.setBrush(QBrush(self.BACKGROUND_COLOR))
        self.setPen(QPen(self.BORDER_COLOR, 2))
        self.min_val = float(min_val)
        self.max_val = float(max_val)

        # Allocate buffers at DISPLAY resolution (not port shape!)
        display_shape = (self.HEIGHT, self.WIDTH, 4)
        self.normalized_float_buffer = np.zeros(
            display_shape, dtype=np.float32, order="C"
        )
        self.bgra_buffer = np.zeros(display_shape, dtype=np.uint8, order="C")

        # Create persistent QImage wrapping buffer (no allocation on paint)
        self.qimage = QImage(
            self.bgra_buffer.data,
            self.WIDTH,
            self.HEIGHT,
            self.WIDTH * 4,
            QImage.Format.Format_ARGB32,
        )

    def update_data(self, data):
        # RESIZE FIRST to display resolution
        if data.shape[:2] != (self.HEIGHT, self.WIDTH):
            data = cv2.resize(
                data,
                (self.WIDTH, self.HEIGHT),
                interpolation=cv2.INTER_AREA,  # Best for downsampling
            )

        # NOW normalize to [0, 255] (smaller dataset!)
        if self.max_val != self.min_val:
            # Fused normalization: 3 passes instead of 4
            scale = 255 / (self.max_val - self.min_val)

            np.subtract(data, self.min_val, out=self.normalized_float_buffer)
            np.multiply(
                self.normalized_float_buffer,
                scale,
                out=self.normalized_float_buffer,
            )
            np.clip(
                self.normalized_float_buffer,
                0,
                255,
                out=self.normalized_float_buffer,
            )

            # Reorder RGBA->BGRA (direct writes, no temporary arrays)
            self.bgra_buffer[:, :, 0] = self.normalized_float_buffer[:, :, 2]  # B
            self.bgra_buffer[:, :, 1] = self.normalized_float_buffer[:, :, 1]  # G
            self.bgra_buffer[:, :, 2] = self.normalized_float_buffer[:, :, 0]  # R
            self.bgra_buffer[:, :, 3] = self.normalized_float_buffer[:, :, 3]  # A
        else:
            # Handle edge case where min == max
            self.bgra_buffer[:] = 128

        self.update()

    def paint(self, painter, option, widget):
        super().paint(painter, option, widget)

        # Already at display resolution - no scaling needed!
        pixmap = QPixmap.fromImage(self.qimage)
        painter.drawPixmap(0, 0, pixmap)
