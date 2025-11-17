import tempfile
import os
import numpy as np
from f32nodes.core import Node, Graph
from f32nodes import Runner


class Source(Node):
    def define_config(self):
        return {"value": 1.0}

    def setup(self):
        self.compute_count = 0

    def define_ports(self):
        self.add_output_port("out", shape=(), min_val=-1000.0, max_val=1000.0)

    def compute(self):
        self.compute_count += 1
        self.write_output("out", np.float32(self.config["value"]))


class Process(Node):
    def define_config(self):
        return {"multiplier": 1.0}

    def setup(self):
        self.compute_count = 0

    def define_ports(self):
        self.add_input_port("in", shape=(), min_val=-1000.0, max_val=1000.0)
        self.add_output_port("out", shape=(), min_val=-1000.0, max_val=1000.0)

    def compute(self):
        self.compute_count += 1
        val = self.read_input("in")
        result = np.float32(val) * np.float32(self.config["multiplier"])
        self.write_output("out", result)


class Combine(Node):
    def define_config(self):
        return {}

    def setup(self):
        self.compute_count = 0

    def define_ports(self):
        self.add_input_port("a", shape=(), min_val=-1000.0, max_val=1000.0)
        self.add_input_port("b", shape=(), min_val=-1000.0, max_val=1000.0)
        self.add_output_port("out", shape=(), min_val=-1000.0, max_val=1000.0)

    def compute(self):
        self.compute_count += 1
        a = self.read_input("a")
        b = self.read_input("b")
        self.write_output("out", np.float32(a) + np.float32(b))


class Sink(Node):
    def define_config(self):
        return {}

    def setup(self):
        self.compute_count = 0
        self.result = None

    def define_ports(self):
        self.add_input_port("in", shape=(), min_val=-1000.0, max_val=1000.0)

    def compute(self):
        self.compute_count += 1
        self.result = self.read_input("in")


class TestGraphComputation:
    def test_pull_based_evaluation_from_sinks(self):
        graph = Graph()
        source = Source("source", {"value": 5.0})
        sink = Sink("sink")

        graph.add(source)
        graph.add(sink)
        graph.connect(source, "out", sink, "in")
        graph.finalize_graph()

        graph.compute()

        assert source.compute_count == 1
        assert sink.compute_count == 1
        assert sink.result == 5.0

    def test_diamond_dependency_caching(self):
        graph = Graph()
        source = Source("source", {"value": 10.0})
        proc1 = Process("proc1", {"multiplier": 2.0})
        proc2 = Process("proc2", {"multiplier": 3.0})
        combiner = Combine("combiner")

        for node in [source, proc1, proc2, combiner]:
            graph.add(node)

        graph.connect(source, "out", proc1, "in")
        graph.connect(source, "out", proc2, "in")
        graph.connect(proc1, "out", combiner, "a")
        graph.connect(proc2, "out", combiner, "b")
        graph.finalize_graph()

        graph.compute()

        assert source.compute_count == 1
        assert proc1.compute_count == 1
        assert proc2.compute_count == 1
        assert combiner.compute_count == 1

        assert combiner.output_ports["out"].get_value() == 50.0

    def test_complex_dag_branching_and_merging(self):
        graph = Graph()
        s1 = Source("s1", {"value": 2.0})
        s2 = Source("s2", {"value": 3.0})
        p1 = Process("p1", {"multiplier": 4.0})
        p2 = Process("p2", {"multiplier": 5.0})
        c1 = Combine("c1")
        sink = Sink("sink")

        for node in [s1, s2, p1, p2, c1, sink]:
            graph.add(node)

        graph.connect(s1, "out", p1, "in")
        graph.connect(s2, "out", p2, "in")
        graph.connect(p1, "out", c1, "a")
        graph.connect(p2, "out", c1, "b")
        graph.connect(c1, "out", sink, "in")
        graph.finalize_graph()

        graph.compute()

        assert sink.result == 23.0


class TestYAMLLoading:
    def test_yaml_to_graph_integration(self):
        yaml_content = """
paths:
  test: "tests.integration.test_backend"

nodes:
  source1:
    type: test.Source
    config:
      value: 7.0

  processor1:
    type: test.Process
    config:
      multiplier: 2.0

  sink1:
    type: test.Sink

connections:
  - from: [source1, "out"]
    to: [processor1, "in"]
  - from: [processor1, "out"]
    to: [sink1, "in"]
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            temp_path = f.name

        try:
            from unittest.mock import patch

            with (
                patch("f32nodes.gui.QApplication"),
                patch("f32nodes.gui.QMainWindow"),
                patch("f32nodes.gui.QGraphicsScene"),
                patch("f32nodes.gui.QGraphicsView"),
            ):
                runner = Runner(temp_path)

                assert len(runner.graph.nodes) == 3

                source_node = next(n for n in runner.graph.nodes if n.name == "source1")
                proc_node = next(
                    n for n in runner.graph.nodes if n.name == "processor1"
                )
                sink_node = next(n for n in runner.graph.nodes if n.name == "sink1")

                assert isinstance(source_node, Source)
                assert source_node.config["value"] == 7.0
                assert isinstance(proc_node, Process)
                assert proc_node.config["multiplier"] == 2.0
                assert isinstance(sink_node, Sink)

                port_results, debug_info = runner.graph.compute()

                assert sink_node.result == 14.0

        finally:
            os.unlink(temp_path)

    def test_comprehensive_yaml_loading_features(self):
        fixture_path = "test_data/comprehensive_yaml_test.yaml"
        from unittest.mock import patch

        with (
            patch("f32nodes.gui.QApplication"),
            patch("f32nodes.gui.QMainWindow"),
            patch("f32nodes.gui.QGraphicsScene"),
            patch("f32nodes.gui.QGraphicsView"),
        ):
            runner = Runner(fixture_path)

            assert len(runner.graph.nodes) == 6
            assert len(runner.graph.connections) == 5

            nodes = {node.name: node for node in runner.graph.nodes}

            assert "source1" in nodes
            assert "source2" in nodes
            assert "processor1" in nodes
            assert "processor2" in nodes
            assert "combiner1" in nodes
            assert "sink1" in nodes

            assert nodes["source1"].config["value"] == 10.0
            assert nodes["source2"].config["value"] == 5.0
            assert nodes["processor1"].config["multiplier"] == 2.0

            assert nodes["processor2"].config["multiplier"] == 1.0

            assert isinstance(nodes["source1"], Source)
            assert isinstance(nodes["processor1"], Process)
            assert isinstance(nodes["combiner1"], Combine)
            assert isinstance(nodes["sink1"], Sink)

            port_results, debug_info = runner.graph.compute()

            # Verify complex computation:
            # source1(10.0) -> processor1(*2.0) = 20.0
            # source2(5.0) -> processor2(*1.0) = 5.0
            # combiner1: 20.0 + 5.0 = 25.0
            # sink1: 25.0
            assert nodes["sink1"].result == 25.0


class TestShapeRangeValidationIntegration:
    def test_shape_range_validation_in_graph_computation(self):
        from f32nodes.core import Graph

        class AudioSource(Node):
            def define_config(self):
                return {}

            def define_ports(self):
                # Audio buffer: 1470 samples, normalized to [-1.0, 1.0]
                self.add_output_port(
                    "audio_out", shape=(1470,), min_val=-1.0, max_val=1.0
                )

            def compute(self):
                audio_data = np.linspace(-0.5, 0.5, 1470, dtype=np.float32)
                self.write_output("audio_out", audio_data)

        class AudioProcessor(Node):
            def define_config(self):
                return {}

            def define_ports(self):
                # Input: audio buffer
                self.add_input_port(
                    "audio_in", shape=(1470,), min_val=-1.0, max_val=1.0
                )
                # Output: spectrum (31 frequency bins, dB scale)
                self.add_output_port(
                    "spectrum_out", shape=(31,), min_val=-60.0, max_val=0.0
                )

            def compute(self):
                _ = self.read_input("audio_in")  # Read input (would be used for FFT)
                # Simulate FFT: output 31 frequency bins in dB scale
                spectrum = np.linspace(-40.0, -10.0, 31, dtype=np.float32)
                self.write_output("spectrum_out", spectrum)

        class SpectrumSink(Node):
            def define_config(self):
                return {}

            def setup(self):
                self.result = None

            def define_ports(self):
                self.add_input_port(
                    "spectrum_in", shape=(31,), min_val=-60.0, max_val=0.0
                )

            def compute(self):
                self.result = self.read_input("spectrum_in")

        # Build graph
        graph = Graph()
        source = AudioSource("audio_source")
        processor = AudioProcessor("processor")
        sink = SpectrumSink("sink")

        graph.add(source)
        graph.add(processor)
        graph.add(sink)

        # Connect with exact shape and range matching
        graph.connect(source, "audio_out", processor, "audio_in")
        graph.connect(processor, "spectrum_out", sink, "spectrum_in")
        graph.finalize_graph()

        # Compute graph
        port_results, debug_info = graph.compute()

        # Verify computation succeeded
        assert sink.result is not None
        assert sink.result.shape == (31,)
        assert np.all(sink.result >= -60.0)
        assert np.all(sink.result <= 0.0)
