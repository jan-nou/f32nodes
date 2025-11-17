from abc import ABC, abstractmethod
import numpy as np
import time


class Port:
    def __init__(self, name, shape, min_val, max_val, default_value=None):
        self.name = name
        if not isinstance(shape, tuple):
            raise TypeError(f"shape must be a tuple, got {type(shape)}")
        if not all(isinstance(dim, int) for dim in shape):
            raise TypeError("All shape dimensions must be integers")
        self.shape = shape
        self.min_val = float(min_val)
        self.max_val = float(max_val)
        self.default_value = default_value
        self.value = None

    def set_value(self, value):
        if not isinstance(value, (np.float32, np.ndarray)):
            raise TypeError(f"Port accepts only numpy float32, got {type(value)}")

        if isinstance(value, np.ndarray):
            if value.dtype != np.float32:
                raise TypeError(f"Port arrays must be float32 dtype, got {value.dtype}")

            # Check shape match
            if value.shape != self.shape:
                raise TypeError(f"Port requires shape {self.shape}, got {value.shape}")

            # Check range
            if np.any(value < self.min_val) or np.any(value > self.max_val):
                raise ValueError(
                    f"Port values must be in range [{self.min_val}, {self.max_val}]"
                )

        elif isinstance(value, np.float32):
            # Scalar value must have empty tuple shape
            if self.shape != ():
                raise TypeError(f"Port requires shape {self.shape}, got scalar")

            # Check range for scalar
            if value < self.min_val or value > self.max_val:
                raise ValueError(
                    f"Port values must be in range [{self.min_val}, {self.max_val}]"
                )

        self.value = value

    def get_value(self):
        if self.value is None and self.default_value is not None:
            return self.default_value
        elif self.value is None and self.default_value is None:
            raise ValueError(f"Input port '{self.name}' has no value and no default")
        return self.value


class Node(ABC):
    def __init__(self, name, config=None):
        self.name = name
        self.input_ports = {}
        self.output_ports = {}

        default_config = self.define_config()
        self.config = default_config.copy()
        if config:
            for key in config:
                if key not in default_config:
                    print(
                        f"Warning: Unknown config key '{key}' for {self.__class__.__name__}"
                    )
            self.config.update(config)
        self.setup()
        self.define_ports()

    def add_input_port(self, name, shape, min_val, max_val, default_value=None):
        self.input_ports[name] = Port(name, shape, min_val, max_val, default_value)

    def add_output_port(self, name, shape, min_val, max_val):
        self.output_ports[name] = Port(name, shape, min_val, max_val)

    def read_input(self, port_name):
        if port_name not in self.input_ports:
            raise KeyError(f"Input port '{port_name}' not found")
        return self.input_ports[port_name].get_value()

    def write_output(self, port_name, value):
        if port_name not in self.output_ports:
            raise KeyError(f"Output port '{port_name}' not found")
        self.output_ports[port_name].set_value(value)

    @abstractmethod
    def define_config(self):
        pass

    def setup(self):
        pass

    @abstractmethod
    def define_ports(self):
        pass

    @abstractmethod
    def compute(self):
        pass


class Graph:
    def __init__(self):
        self.nodes = []
        self.connections = []
        self.sink_nodes = None
        self.is_finalized = False

    def add(self, node):
        self.nodes.append(node)

    def connect(self, source_node, source_port, target_node, target_port):
        if source_port not in source_node.output_ports:
            available_ports = list(source_node.output_ports.keys())
            raise KeyError(
                f"Source node '{source_node.name}' has no output port '{source_port}'. Available output ports: {available_ports}"
            )
        if target_port not in target_node.input_ports:
            available_ports = list(target_node.input_ports.keys())
            raise KeyError(
                f"Target node '{target_node.name}' has no input port '{target_port}'. Available input ports: {available_ports}"
            )

        source_shape = source_node.output_ports[source_port].shape
        target_shape = target_node.input_ports[target_port].shape
        source_min = source_node.output_ports[source_port].min_val
        target_min = target_node.input_ports[target_port].min_val
        source_max = source_node.output_ports[source_port].max_val
        target_max = target_node.input_ports[target_port].max_val

        if source_shape != target_shape:
            raise TypeError(
                f"Cannot connect port '{source_node.name}.{source_port}' with shape {source_shape} to port '{target_node.name}.{target_port}' with shape {target_shape}"
            )

        if source_min != target_min or source_max != target_max:
            raise TypeError(
                f"Cannot connect port '{source_node.name}.{source_port}' with range [{source_min}, {source_max}] to port '{target_node.name}.{target_port}' with range [{target_min}, {target_max}]"
            )

        # Check if this input port already has a connection
        for existing_connection in self.connections:
            if (
                existing_connection["target_node"] == target_node
                and existing_connection["target_port"] == target_port
            ):
                raise ValueError(
                    f"Input port '{target_node.name}.{target_port}' already has a connection. Multiple connections to the same input port are not allowed."
                )

        connection = {
            "source_node": source_node,
            "source_port": source_port,
            "target_node": target_node,
            "target_port": target_port,
        }
        self.connections.append(connection)

    def finalize_graph(self):
        """Validate topology, detect cycles, and prepare the graph for execution."""

        def get_downstream_nodes(node):
            return [
                conn["target_node"]
                for conn in self.connections
                if conn["source_node"] == node
            ]

        def ensure_all_nodes_connected():
            connected = {conn["source_node"] for conn in self.connections} | {
                conn["target_node"] for conn in self.connections
            }
            isolated = [node.name for node in self.nodes if node not in connected]
            if isolated:
                raise ValueError(
                    "Graph contains unconnected nodes: "
                    + ", ".join(isolated)
                    + ". Isolated nodes are not allowed."
                )

        def ensure_dag():
            state = {node: 0 for node in self.nodes}
            path = []

            def dfs(node):
                if state[node] == 1:
                    cycle_start = path.index(node)
                    cycle_nodes = path[cycle_start:] + [node]
                    cycle_desc = " -> ".join(n.name for n in cycle_nodes)
                    raise ValueError(
                        f"Graph contains a cycle: {cycle_desc}. Cycles are not allowed in DAGs."
                    )
                if state[node] == 2:
                    return
                state[node] = 1
                path.append(node)
                for downstream in get_downstream_nodes(node):
                    dfs(downstream)
                path.pop()
                state[node] = 2

            for node in self.nodes:
                if state[node] == 0:
                    dfs(node)

        def ensure_single_component():
            if not self.nodes:
                return
            adjacency = {node: set() for node in self.nodes}
            for conn in self.connections:
                src = conn["source_node"]
                dst = conn["target_node"]
                adjacency[src].add(dst)
                adjacency[dst].add(src)

            visited = set()
            stack = [self.nodes[0]]
            while stack:
                current = stack.pop()
                if current in visited:
                    continue
                visited.add(current)
                stack.extend(adjacency[current] - visited)

            if len(visited) != len(self.nodes):
                disconnected = [node.name for node in self.nodes if node not in visited]
                raise ValueError(
                    "Graph must form a single connected component. "
                    f"Disconnected nodes: {', '.join(disconnected)}"
                )

        ensure_all_nodes_connected()
        ensure_dag()
        ensure_single_component()

        # Find sink nodes (nodes with no outgoing connections)
        self.sink_nodes = [
            node for node in self.nodes if not get_downstream_nodes(node)
        ]

        # Report optional port defaults
        for node in self.nodes:
            for port_name, port in node.input_ports.items():
                if port.default_value is not None and port.value is None:
                    print(
                        f"Node '{node.name}': Optional port '{port_name}' not connected, using default value {port.default_value}"
                    )

        self.is_finalized = True

    def compute(self):
        if not self.is_finalized:
            raise RuntimeError(
                "Graph must be finalized before computation. Call finalize_graph() first."
            )

        def compute_node_recursive(node, computed_nodes, node_computation_times):
            if node in computed_nodes:
                return

            for connection in self.connections:
                if connection["target_node"] == node:
                    source_node = connection["source_node"]
                    compute_node_recursive(
                        source_node, computed_nodes, node_computation_times
                    )

                    source_port = connection["source_port"]
                    target_port = connection["target_port"]
                    source_value = source_node.output_ports[source_port].get_value()
                    node.input_ports[target_port].set_value(source_value)

            node_start_time = time.time()
            node.compute()
            node_computation_time = time.time() - node_start_time
            node_computation_times[node.name] = node_computation_time
            computed_nodes.add(node)

        node_computation_times = {}
        computed_nodes = set()

        for sink_node in self.sink_nodes:
            compute_node_recursive(sink_node, computed_nodes, node_computation_times)

        port_results = []
        for node in self.nodes:
            for port_name, port in node.output_ports.items():
                port_results.append(
                    {"node": node.name, "port": port_name, "value": port.get_value()}
                )

        debug_info = {"node_computation_times": node_computation_times}
        return (port_results, debug_info)
