
# dashboard.py
from graphviz import Digraph
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris

from engine import ValueTensor
from model import MLP
from optim import SGD, SGDMomentum, Adam
from objective import cross_entropy

# — Page Setup —
st.set_page_config(layout="wide")
st.sidebar.title("Nanograd Dashboard")
page = st.sidebar.radio("Go to", ["Architecture", "Training", "Loss Landscape"])

# … Architecture page code here …

if page == "Architecture":
    st.header("🔍 Network Architecture")

    # 1. Load a dataset to infer dims
    data = load_iris()
    X, y = data.data, data.target
    input_dim = X.shape[1]        # 4 features
    output_dim = len(np.unique(y))  # 3 classes

    # 2. Define a model with two hidden layers
    hidden1, hidden2 = 16, 8
    layer_sizes = [input_dim, hidden1, hidden2, output_dim]
    activation_name = "relu"
    mlp = MLP(layer_sizes, activation=activation_name)

    # 3. Build a Graphviz diagram with circle nodes
    dot = Digraph()
    dot.graph_attr.update(rankdir="LR")  # left → right

    # color map
    colors = {
        "input": "lightblue",
        "hidden": "lightgreen",
        "output": "lightcoral"
    }

    node_ids = []
    for idx, size in enumerate(layer_sizes):
        node_id = f"L{idx}"
        node_ids.append(node_id)

        if idx == 0:
            lbl = f"{node_id}\nInput\n{size}"
            fill = colors["input"]
        elif idx == len(layer_sizes) - 1:
            lbl = f"{node_id}\nOutput\n{size}"
            fill = colors["output"]
        else:
            lbl = f"{node_id}\nHidden\n{size}\n{activation_name}"
            fill = colors["hidden"]

        dot.node(node_id, label=lbl, shape="circle", style="filled", fillcolor=fill)

    # connect layers
    for i in range(len(node_ids) - 1):
        dot.edge(node_ids[i], node_ids[i + 1])

    # 4. Display the graph full‑width
    st.graphviz_chart(dot.source, use_container_width=True)

    # 5. Sidebar dropdown for layer details
    st.sidebar.subheader("Layer Details")
    selected = st.sidebar.selectbox("Choose a layer:", node_ids)

    # 6. Show stats for the selected layer
    st.subheader(f"Details for {selected}")
    idx = int(selected[1:])
    if idx == 0:
        st.write("- **Layer Type:** Input")
        st.write(f"- **Feature Dimension:** {layer_sizes[0]}")
        st.write(f"- **Sample Input Range:** [{X.min():.2f}, {X.max():.2f}]")
    else:
        # for idx>0, mlp.layers[idx-1] is the mapping into L{idx}
        layer = mlp.layers[idx - 1]
        w, b = layer.w.data, layer.b.data
        layer_type = "Output" if idx == len(layer_sizes)-1 else "Hidden"
        st.write(f"- **Layer Type:** {layer_type}")
        st.write(f"- **Weight Shape:** {w.shape}")
        st.write(f"- **Weight Stats:** mean {w.mean():.3f}, std {w.std():.3f}")
        st.write(f"- **Bias Shape:** {b.shape}")
        st.write(f"- **Bias Stats:** mean {b.mean():.3f}, std {b.std():.3f}")
        if layer_type == "Output":
            st.write(f"- **Target Classes:** {output_dim}")

# — Training Page —
if page == "Training":
    st.header("⚙️ Training on Iris (MSE)")

    # 1. Load data & one‑hot encode
    data = load_iris()
    X = data.data                          # (150, 4)
    y = data.target                        # (150,)
    num_classes = len(np.unique(y))
    Y_onehot = np.eye(num_classes)[y]      # (150, 3)

    # 2. Hyperparameter controls
    st.sidebar.subheader("Training Hyperparams")
    optimizer_name = st.sidebar.selectbox("Optimizer", ["SGD", "SGD with Momentum", "Adam"])
    lr = st.sidebar.number_input("Learning Rate", min_value=1e-5, max_value=1.0, value=0.01, format="%.5f")
    if optimizer_name == "SGD with Momentum":
        momentum = st.sidebar.slider("Momentum", 0.0, 0.99, 0.9, 0.01)
    if optimizer_name == "Adam":
        beta1 = st.sidebar.slider("Beta1", 0.0, 0.999, 0.9, 0.01)
        beta2 = st.sidebar.slider("Beta2", 0.0, 0.999, 0.999, 0.001)
        eps   = st.sidebar.number_input("Epsilon", min_value=1e-8, max_value=1e-4, value=1e-8, format="%.8f")
    epochs = st.sidebar.slider("Epochs", 1, 50, 10)

    # 3. Train on button click
    if st.sidebar.button("Start Training"):
        with st.spinner("Training..."):
            # Build model
            hidden1, hidden2 = 16, 8
            layer_sizes = [X.shape[1], hidden1, hidden2, num_classes]
            mlp = MLP(layer_sizes, activation="relu")

            # Choose optimizer
            if optimizer_name == "SGD":
                optim = SGD(mlp.parameters(), lr=lr)
            elif optimizer_name == "SGD with Momentum":
                optim = SGDMomentum(mlp.parameters(), momentum=momentum, lr=lr)
            else:  # Adam
                optim = Adam(mlp.parameters(), lr=lr, betas=(beta1, beta2), eps=eps)

            # Lists to store metrics
            losses, accs, grad_norms = [], [], []

            # Full‑batch training loop
            for ep in range(epochs):
                # forward
                x_tensor = ValueTensor(X)
                y_tensor = ValueTensor(Y_onehot)
                preds = mlp(x_tensor)

                # compute loss & backward
                loss = cross_entropy(y_tensor, preds)
                loss.backward()

                # record metrics
                loss_val = float(loss.data.squeeze())
                losses.append(loss_val)

                # compute “accuracy” by argmax on raw preds
                pred_labels = preds.data.argmax(axis=1)
                acc = (pred_labels == y).mean()
                accs.append(acc)

                # avg gradient‑norm across all params
                gn = np.mean([np.linalg.norm(p.grad) for p in mlp.parameters()])
                grad_norms.append(gn)

                # update & zero‑grad
                optim.step()
                mlp.zero_grad()

        # 4. Display metrics
        st.subheader("Training Metrics")
        df = pd.DataFrame({
            "Loss": losses,
            "Accuracy": accs,
            "Grad Norm": grad_norms
        }, index=np.arange(1, epochs+1))
        st.line_chart(df[["Loss", "Accuracy"]])
        st.line_chart(df[["Grad Norm"]])

        st.markdown(f"**Final Loss:** {losses[-1]:.4f}    **Final Accuracy:** {accs[-1]*100:.2f}%")

        # 5. Optional: show param histograms
        st.subheader("Final Parameter Distributions")
        for i, param in enumerate(mlp.parameters()):
            st.write(f"- Param {i} norm: {np.linalg.norm(param.data):.3f}")

# — Loss Landscape & Paths Page —
if page == "Loss Landscape":
    # import matplotlib.pyplot as plt
    import plotly.graph_objects as go
    st.header("🌄 Loss Landscape & Optimizer Paths")

    # Sidebar controls
    st.sidebar.subheader("Landscape Settings")
    fn = st.sidebar.selectbox("Loss Function", ["Quadratic", "Rosenbrock"])
    lr = st.sidebar.number_input("Learning Rate", min_value=1e-3, max_value=1.0, value=0.1, step=0.01, format="%.3f")
    steps = st.sidebar.slider("Steps", 5, 200, 50)
    optim_names = st.sidebar.multiselect("Optimizers", ["SGD", "SGD with Momentum", "Adam"], default=["SGD", "Adam"])
    momentum = st.sidebar.slider("Momentum (for SGD+M)", 0.0, 0.99, 0.9, 0.01)
    beta1 = st.sidebar.slider("Beta1 (Adam)", 0.0, 0.999, 0.9, 0.01)
    beta2 = st.sidebar.slider("Beta2 (Adam)", 0.0, 0.999, 0.999, 0.001)
    eps   = st.sidebar.number_input("Epsilon (Adam)", min_value=1e-8, max_value=1e-4, value=1e-8, format="%.1e")

    if st.sidebar.button("Run Trajectories"):

        # Define loss functions
        def compute_loss(xv, yv):
            if fn == "Quadratic":
                return xv**2 + (2*yv)**2
            else:  # Rosenbrock in 2D
                a, b = 1., 100.
                return (a - xv)**2 + b*(yv - xv**2)**2

        # Container for trajectories
        trajs = {}

        for name in optim_names:
            # Initialize parameters at (-2, 2)
            x = ValueTensor(np.array(-2.0))
            y = ValueTensor(np.array( 2.0))
            params = [x, y]

            # Choose optimizer
            if name == "SGD":
                optim = SGD(params, lr=lr)
            elif name == "SGD with Momentum":
                optim = SGDMomentum(params, momentum=momentum, lr=lr)
            else:
                optim = Adam(params, lr=lr, betas=(beta1, beta2), eps=eps)

            path = [(x.data.item(), y.data.item())]

            # Perform optimization steps
            for _ in range(steps):
                # compute loss and backprop
                loss = compute_loss(x, y)
                loss.backward()

                # update
                optim.step()
                for p in params:
                    p.zero_grad()

                path.append((x.data.item(), y.data.item()))

            trajs[name] = np.array(path)

        # Prepare contour grid
        grid_pts = 200
        xlin = np.linspace(-3, 3, grid_pts)
        ylin = np.linspace(-3, 3, grid_pts)
        Xg, Yg = np.meshgrid(xlin, ylin)
        Zg = compute_loss(ValueTensor(Xg), ValueTensor(Yg)).data

        # Plot
        # fig, ax = plt.subplots(figsize=(6, 6))
        # cs = ax.contour(Xg, Yg, Zg, levels=30, cmap='viridis')
        # ax.clabel(cs, inline=1, fontsize=8)
        # ax.set_title(f"{fn} Landscape & Optimizer Paths")
        # ax.set_xlabel("x")
        # ax.set_ylabel("y")

        # # Overlay trajectories
        # for name, path in trajs.items():
        #     ax.plot(path[:,0], path[:,1], marker='o', label=name)
        # ax.legend()

        # st.pyplot(fig, use_container_width=True)
        # Plot with Plotly for full responsiveness
        fig = go.Figure()

        # Contour
        fig.add_trace(go.Contour(
            z=Zg,
            x=xlin,
            y=ylin,
            colorscale='Viridis',
            contours=dict(showlabels=True),
            showscale=False
        ))

        # Optimizer paths
        for name, path in trajs.items():
            fig.add_trace(go.Scatter(
                x=path[:,0],
                y=path[:,1],
                mode='lines+markers',
                name=name
            ))

        fig.update_layout(
            title=f"{fn} Landscape & Optimizer Paths",
            xaxis_title="x",
            yaxis_title="y",
            margin=dict(l=20, r=20, t=40, b=20)
        )

        st.plotly_chart(fig) #, use_container_width=True)
