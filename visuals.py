from graphviz import Digraph
import numpy as np

np.set_printoptions(precision=3, suppress=True)

# Color mapping based on gradient norm
def get_color_for_gradient(norm, max_norm=1.0):
    clipped = min(norm / max_norm, 1.0)
    red = int(255 * clipped)
    green = int(255 * (1 - clipped))
    return f"#{red:02x}{green:02x}00"

def summarize_array(arr):
    if arr.ndim == 0:
        return f"{arr.item():.3f}"
    return (
        f"mean: {arr.mean():.3f}, std: {arr.std():.3f}\\n"
        f"min: {arr.min():.3f}, max: {arr.max():.3f}"
    )

def trace(root):
    nodes, edges = set(), set()
    def build(v):
        if v not in nodes:
            nodes.add(v)
            for child in v._prev:
                edges.add((child, v))
                build(child)
    build(root)
    return nodes, edges

def draw_dot(root, format='svg', rankdir='LR', max_grad_norm=1.0):
    nodes, edges = trace(root)
    dot = Digraph(format=format, graph_attr={'rankdir': rankdir})

    op_color_map = {
        '+': 'lightblue', '*': 'lightgreen', '@': 'plum',
        'relu': 'orange', 'tanh': 'yellow', 'sigmoid': 'pink', 'exp': 'lightgoldenrod',
        'sum': 'lightcoral', 'mean': 'lightcoral', 'max': 'lightcoral',
        'reshape': 'lightcyan', 'transpose': 'lightcyan'
    }

    for n in nodes:
        shape_str = str(n.data.shape)
        data_summary = summarize_array(n.data)
        grad_summary = summarize_array(n.grad)
        grad_norm = np.linalg.norm(n.grad)
        color = get_color_for_gradient(grad_norm, max_norm=max_grad_norm)

        op_label = n._op if n._op else "Input"
        label = (
            f"{op_label}\\n"
            f"Shape: {shape_str}\\n"
            f"| Data: {data_summary}\\n"
            f"| Grad: {grad_summary}"
        )

        fill = op_color_map.get(n._op, color if n._op else "white")
        dot.node(str(id(n)), label=label, shape='record', style='filled', fillcolor=fill)

        if n._op:
            op_node = str(id(n)) + n._op
            dot.node(op_node, label=n._op, shape='oval', style='filled', fillcolor=fill)
            dot.edge(op_node, str(id(n)))

    for n1, n2 in edges:
        op_node = str(id(n2)) + n2._op if n2._op else str(id(n2))
        dot.edge(str(id(n1)), op_node)

    return dot
