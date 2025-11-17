import pytest
import numpy as np
from f32nodes.core import Node, Port


class TestPort:
    def test_port_accepts_numpy_values(self):
        port_scalar = Port("scalar", shape=(), min_val=0.0, max_val=10.0)
        port_scalar.set_value(np.float32(3.14))
        assert float(port_scalar.get_value()) == pytest.approx(3.14, rel=1e-6)

        port_array = Port("array", shape=(3,), min_val=0.0, max_val=10.0)
        test_array = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        port_array.set_value(test_array)
        np.testing.assert_array_equal(port_array.get_value(), test_array)

    def test_port_rejects_non_numpy_types(self):
        port = Port("test_port", shape=(), min_val=0.0, max_val=10.0)
        with pytest.raises(TypeError):
            port.set_value("not a numpy value")
        with pytest.raises(TypeError):
            port.set_value(3.14)  # Python float, not np.float32

    def test_port_with_default_value(self):
        port = Port(
            "test_port",
            shape=(),
            min_val=0.0,
            max_val=10.0,
            default_value=np.float32(3.14),
        )
        assert float(port.get_value()) == pytest.approx(3.14, rel=1e-6)


class TestNode:
    def test_node_writes_to_port_objects(self):
        class TestNode(Node):
            def define_config(self):
                return {}

            def define_ports(self):
                self.add_output_port("output", shape=(), min_val=0.0, max_val=100.0)

            def compute(self):
                self.write_output("output", np.float32(42.0))

        node = TestNode("test")
        node.compute()

        assert isinstance(node.output_ports["output"], Port)
        assert node.output_ports["output"].get_value() == 42.0

    def test_node_config_merging(self):
        class TestNode(Node):
            def define_config(self):
                return {"param1": "default", "param2": 42}

            def define_ports(self):
                self.add_output_port("output", shape=(), min_val=0.0, max_val=100.0)

            def compute(self):
                pass

        node1 = TestNode("test1")
        assert node1.config["param1"] == "default"
        assert node1.config["param2"] == 42

        node2 = TestNode("test2", {"param1": "override"})
        assert node2.config["param1"] == "override"
        assert node2.config["param2"] == 42  # Should keep default

    def test_node_port_access_validation(self):
        class TestNode(Node):
            def define_config(self):
                return {}

            def define_ports(self):
                self.add_input_port("input", shape=(), min_val=0.0, max_val=100.0)
                self.add_output_port("output", shape=(), min_val=0.0, max_val=100.0)

            def compute(self):
                pass

        node = TestNode("test")

        with pytest.raises(KeyError, match="Input port 'nonexistent' not found"):
            node.read_input("nonexistent")

        with pytest.raises(KeyError, match="Output port 'nonexistent' not found"):
            node.write_output("nonexistent", np.float32(1.0))

    def test_optional_ports_use_defaults(self):
        class TestNode(Node):
            def define_config(self):
                return {}

            def define_ports(self):
                self.add_input_port(
                    "gain",
                    shape=(),
                    min_val=0.0,
                    max_val=10.0,
                    default_value=np.float32(1.5),
                )
                self.add_output_port("output", shape=(), min_val=0.0, max_val=100.0)

            def compute(self):
                gain = self.read_input("gain")
                self.write_output(
                    "output", np.float32(gain) * np.float32(10.0)
                )

        node1 = TestNode("test1")
        node1.compute()
        assert node1.output_ports["output"].get_value() == 15.0  # 1.5 * 10

        node2 = TestNode("test2")
        node2.input_ports["gain"].set_value(np.float32(3.0))
        node2.compute()
        assert node2.output_ports["output"].get_value() == 30.0  # 3.0 * 10

    def test_required_ports_validation(self):
        class TestNode(Node):
            def define_config(self):
                return {}

            def define_ports(self):
                self.add_input_port("required", shape=(), min_val=0.0, max_val=100.0)
                self.add_output_port("output", shape=(), min_val=0.0, max_val=100.0)

            def compute(self):
                value = self.read_input("required")
                self.write_output("output", np.float32(value))

        node1 = TestNode("test1")
        with pytest.raises(
            ValueError, match="Input port 'required' has no value and no default"
        ):
            node1.compute()

        node2 = TestNode("test2")
        node2.input_ports["required"].set_value(np.float32(7.0))
        node2.compute()
        assert node2.output_ports["output"].get_value() == 7.0

    def test_config_based_defaults(self):
        class TestNode(Node):
            def define_config(self):
                return {"scale_default": 2.5}

            def define_ports(self):
                self.add_input_port(
                    "scale",
                    shape=(),
                    min_val=0.0,
                    max_val=10.0,
                    default_value=np.float32(self.config["scale_default"]),
                )
                self.add_output_port("output", shape=(), min_val=0.0, max_val=1000.0)

            def compute(self):
                scale = self.read_input("scale")
                self.write_output("output", np.float32(scale * 100.0))

        node1 = TestNode("test1")
        node1.compute()
        assert node1.output_ports["output"].get_value() == 250.0  # 2.5 * 100

        node2 = TestNode("test2", {"scale_default": 1.5})
        node2.compute()
        assert node2.output_ports["output"].get_value() == 150.0  # 1.5 * 100


class TestPortShapeRangeValidation:
    def test_port_requires_exact_shape_tuple(self):
        port_scalar = Port("scalar", shape=(), min_val=0.0, max_val=1.0)
        assert port_scalar.shape == ()
        assert port_scalar.min_val == 0.0
        assert port_scalar.max_val == 1.0

        port_1d = Port("audio", shape=(1470,), min_val=-1.0, max_val=1.0)
        assert port_1d.shape == (1470,)

        port_2d = Port("image", shape=(480, 640, 3), min_val=0.0, max_val=255.0)
        assert port_2d.shape == (480, 640, 3)

    def test_port_validates_shape_exactly_on_set_value(self):
        port = Port("audio", shape=(1470,), min_val=-1.0, max_val=1.0)

        correct_array = np.array([0.5] * 1470, dtype=np.float32)
        port.set_value(correct_array)

        wrong_shape = np.array([0.5] * 1000, dtype=np.float32)
        with pytest.raises(TypeError, match="shape"):
            port.set_value(wrong_shape)

    def test_port_validates_range_on_set_value(self):
        port = Port("normalized", shape=(100,), min_val=0.0, max_val=1.0)

        # Values in range should work
        valid_array = np.array([0.5] * 100, dtype=np.float32)
        port.set_value(valid_array)

        # Values outside range should fail with ValueError
        invalid_min = np.array([-0.1] + [0.5] * 99, dtype=np.float32)
        with pytest.raises(ValueError, match="range"):
            port.set_value(invalid_min)

        invalid_max = np.array([1.1] + [0.5] * 99, dtype=np.float32)
        with pytest.raises(ValueError, match="range"):
            port.set_value(invalid_max)

    def test_port_scalar_uses_empty_tuple_shape(self):
        """Scalar ports should use shape=() (empty tuple)"""
        port = Port("gain", shape=(), min_val=0.0, max_val=10.0)

        # Scalar should work
        port.set_value(np.float32(5.0))
        assert port.get_value() == 5.0

        # Array should fail
        with pytest.raises(TypeError, match="shape"):
            port.set_value(np.array([5.0], dtype=np.float32))


class TestGraphConnectionCompatibility:
    """Test Graph.connect() with exact shape and range matching"""

    def test_connection_requires_exact_shape_and_range_match(self):
        """Graph.connect should only allow connections with identical shapes and ranges"""
        from f32nodes.core import Graph

        class SourceNode(Node):
            def define_config(self):
                return {}

            def define_ports(self):
                self.add_output_port("out", shape=(100,), min_val=0.0, max_val=1.0)

            def compute(self):
                pass

        class CompatibleSink(Node):
            def define_config(self):
                return {}

            def define_ports(self):
                self.add_input_port("in", shape=(100,), min_val=0.0, max_val=1.0)

            def compute(self):
                pass

        class WrongShapeSink(Node):
            def define_config(self):
                return {}

            def define_ports(self):
                self.add_input_port("in", shape=(200,), min_val=0.0, max_val=1.0)

            def compute(self):
                pass

        class WrongRangeSink(Node):
            def define_config(self):
                return {}

            def define_ports(self):
                self.add_input_port("in", shape=(100,), min_val=-1.0, max_val=1.0)

            def compute(self):
                pass

        graph = Graph()
        source = SourceNode("source")
        compatible = CompatibleSink("compatible")
        wrong_shape = WrongShapeSink("wrong_shape")
        wrong_range = WrongRangeSink("wrong_range")

        graph.add(source)
        graph.add(compatible)
        graph.add(wrong_shape)
        graph.add(wrong_range)

        # Compatible should work
        graph.connect(source, "out", compatible, "in")

        # Incompatible shape should fail
        with pytest.raises(TypeError):
            graph.connect(source, "out", wrong_shape, "in")

        # Incompatible range should fail
        with pytest.raises(TypeError):
            graph.connect(source, "out", wrong_range, "in")


class TestGraphConnectivity:
    """Graph.finalize_graph should reject disconnected components"""

    def _make_passthrough_node(self, name):
        class Passthrough(Node):
            def define_config(self):
                return {}

            def define_ports(self):
                self.add_input_port("in", shape=(), min_val=-1.0, max_val=1.0)
                self.add_output_port("out", shape=(), min_val=-1.0, max_val=1.0)

            def compute(self):
                self.write_output("out", self.read_input("in"))

        return Passthrough(name)

    def _make_source_node(self, name):
        class Source(Node):
            def define_config(self):
                return {}

            def define_ports(self):
                self.add_output_port("out", shape=(), min_val=-1.0, max_val=1.0)

            def compute(self):
                self.write_output("out", np.float32(0.5))

        return Source(name)

    def _make_sink_node(self, name):
        class Sink(Node):
            def define_config(self):
                return {}

            def define_ports(self):
                self.add_input_port("in", shape=(), min_val=-1.0, max_val=1.0)

            def compute(self):
                self.read_input("in")

        return Sink(name)

    def test_disconnected_components_raise(self):
        from f32nodes.core import Graph

        graph = Graph()
        # First component: source -> sink
        source = self._make_source_node("source")
        sink = self._make_sink_node("sink")
        graph.add(source)
        graph.add(sink)
        graph.connect(source, "out", sink, "in")

        # Second component: passthrough alone (still connected internally)
        passthrough_source = self._make_source_node("orphan_source")
        passthrough_sink = self._make_sink_node("orphan_sink")
        graph.add(passthrough_source)
        graph.add(passthrough_sink)
        graph.connect(passthrough_source, "out", passthrough_sink, "in")

        with pytest.raises(
            ValueError, match="Graph must form a single connected component"
        ):
            graph.finalize_graph()

    def test_single_component_passes(self):
        from f32nodes.core import Graph

        graph = Graph()
        source = self._make_source_node("source")
        mid = self._make_passthrough_node("mid")
        sink = self._make_sink_node("sink")

        graph.add(source)
        graph.add(mid)
        graph.add(sink)

        graph.connect(source, "out", mid, "in")
        graph.connect(mid, "out", sink, "in")

        # Should not raise once we enforce a single connected component.
        graph.finalize_graph()
