from graphviz import Digraph
import numpy as np
np.set_printoptions(precision=3, suppress=True)

def summarize_array(arr, max_elements=5):
    """Return a summary string for an array with basic stats."""
    if arr.ndim == 0:
        return f"{arr.item():.3f}"
    else:
        mean_val = arr.mean()
        std_val = arr.std()
        min_val = arr.min()
        max_val = arr.max()
        return f"mean: {mean_val:.3f}, std: {std_val:.3f}\\nmin: {min_val:.3f}, max: {max_val:.3f}"

# Define a mapping for operations to colors.
op_color_map = {
    "+": "lightblue",
    "*": "lightgreen",
    "@": "plum",
    "relu": "orange",
    "tanh": "yellow",
    "sigmoid": "pink",
    "exp": "lightgoldenrod",
    "sum": "lightcoral",
    "mean": "lightcoral",
    "max": "lightcoral",
    "reshape": "lightcyan",
    "transpose": "lightcyan",
    # For power, you can add a specific color:
    "**2": "violet",  # Example for specific exponent values
    "**3": "violet",  # Or leave generic
}

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

def draw_dot(root, format='svg', rankdir='LR'):
    """
    Visualizes the computational graph for tensor operations.
    Uses colors to distinguish between different operations and input nodes.
    """
    assert rankdir in ['LR', 'TB']
    nodes, edges = trace(root)
    dot = Digraph(format=format, graph_attr={'rankdir': rankdir})
    
    for n in nodes:
        shape_info = str(n.data.shape)
        data_summary = summarize_array(n.data)
        grad_summary = summarize_array(n.grad)
        
        # Determine the base label. If there's no op, label it as Input.
        op_label = n._op if n._op else "Input"
        
        # Determine fillcolor based on op type.
        if n._op:
            # Look up a color using the mapping; if not found, use lightgray.
            fillcolor = op_color_map.get(n._op, "lightgray")
        else:
            fillcolor = "white"
        
        # Build the label including summary info.
        label = f"{op_label}\\nShape: {shape_info}\\nData: {data_summary}\\nGrad: {grad_summary}"
        
        # Set different styles for operations vs. inputs:
        node_style = "filled"
        font_color = "black" if fillcolor != "black" else "white"
        
        dot.node(str(id(n)), label=label, shape='record',
                 fillcolor=fillcolor, style=node_style, fontcolor=font_color)
        
        if n._op:
            # Create a separate node for the operation (optionally, or merge with value).
            op_node = str(id(n)) + n._op
            dot.node(op_node, label=n._op, shape='oval', fillcolor=op_color_map.get(n._op, "lightgray"),
                     style="filled", fontcolor="black")
            dot.edge(op_node, str(id(n)))
    
    for n1, n2 in edges:
        op_node = str(id(n2)) + n2._op if n2._op else str(id(n2))
        dot.edge(str(id(n1)), op_node)
    
    return dot
