import sys
from pathlib import Path

import pytest

from f32nodes import Runner

DEMOS_DIR = Path(__file__).resolve().parents[2] / "demos"


@pytest.fixture
def isolated_demo_imports():
    """Every demo has its own `nodes` module; keep them from leaking between tests."""
    saved_path = list(sys.path)
    sys.modules.pop("nodes", None)
    yield
    sys.path[:] = saved_path
    sys.modules.pop("nodes", None)


@pytest.mark.parametrize("demo", ["demo1", "demo2"])
def test_demo_graph_builds_and_computes(demo, isolated_demo_imports):
    yaml_path = DEMOS_DIR / demo / "graph.yaml"

    runner = Runner.__new__(Runner)
    runner.yaml_dir = str(yaml_path.parent)
    graph = runner.build_graph_from_yaml(str(yaml_path))

    for _ in range(3):
        port_results, _ = graph.compute()

    assert port_results
