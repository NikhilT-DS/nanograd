import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from graphviz import Digraph
from sklearn.datasets import (
    load_iris, load_wine, load_breast_cancer,
    load_diabetes, fetch_california_housing
)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error
)

from engine import ValueTensor
from model import MLP
from optim import SGD, SGDMomentum, Adam
from objective import cross_entropy, mse

# — Page Setup —
st.set_page_config(layout="wide")
st.sidebar.title("Nanograd Dashboard")
page = st.sidebar.radio("Go to", ["Setup & Training", "Loss Landscape"])

# — Dataset Selection —
task = st.sidebar.selectbox("Task Type", ["Classification", "Regression"])
if task == "Classification":
    dataset_choice = st.sidebar.selectbox(
        "Dataset", ["Iris", "Wine", "Breast Cancer"]
    )
else:
    dataset_choice = st.sidebar.selectbox(
        "Dataset", ["Diabetes", "California Housing"]
    )

# — Load Data On Change —
# Reload whenever user picks a new dataset
if ("dataset_choice" not in st.session_state 
        or st.session_state.dataset_choice != dataset_choice):
    st.session_state.dataset_choice = dataset_choice
    # load selected dataset
    if dataset_choice == "Iris":
        data = load_iris()
    elif dataset_choice == "Wine":
        data = load_wine()
    elif dataset_choice == "Breast Cancer":
        data = load_breast_cancer()
    elif dataset_choice == "Diabetes":
        data = load_diabetes()
    else:
        data = fetch_california_housing()
    # persist features and targets
    st.session_state.X = data.data
    st.session_state.y = data.target
    # update dims
    st.session_state.input_dim = st.session_state.X.shape[1]
    st.session_state.output_dim = (
        len(np.unique(st.session_state.y)) if task == "Classification" else 1
    )

X = st.session_state.X
y = st.session_state.y
y = st.session_state.y

# — Setup & Training Page —
if page == "Setup & Training":
    st.header("🔧 Setup & Training")

    # Model Architecture Controls
    st.sidebar.subheader("Model Architecture")
    num_hidden = st.sidebar.slider("Number of Hidden Layers", 1, 3, 2)
    hidden_sizes = [
        st.sidebar.number_input(
            f"Units in hidden layer {i+1}", 1, 512,
            value=[16,8,4][i], step=1
        ) for i in range(num_hidden)
    ]
    activation = st.sidebar.selectbox(
        "Activation Function", ["relu", "tanh", "sigmoid", None]
    )

    # Persist architecture
    st.session_state.layer_sizes = (
        [st.session_state.input_dim] + hidden_sizes + [st.session_state.output_dim]
    )
    st.session_state.activation = activation

    # Instantiate Model
    mlp = MLP(st.session_state.layer_sizes, activation=activation)
    st.session_state.initial_weights = [w.data.copy() for w in mlp.parameters()]

    # Visualize Architecture
    dot = Digraph()
    dot.graph_attr.update(rankdir="LR")
    colors = {"input":"lightblue","hidden":"lightgreen","output":"lightcoral"}
    ids = []
    for idx, size in enumerate(st.session_state.layer_sizes):
        node_id = f"L{idx}"; ids.append(node_id)
        if idx==0:
            lbl,fill = f"{node_id}\nInput\n{size}", colors["input"]
        elif idx==len(st.session_state.layer_sizes)-1:
            label="Logits" if task=="Classification" else "Output"
            lbl,fill = f"{node_id}\n{label}\n{size}", colors["output"]
        else:
            lbl,fill = f"{node_id}\nHidden\n{size}\n{activation}", colors["hidden"]
        dot.node(node_id,label=lbl,shape="circle",style="filled",fillcolor=fill)
    for i in range(len(ids)-1): dot.edge(ids[i],ids[i+1])
    st.graphviz_chart(dot.source,use_container_width=True)

    # Training Hyperparameters
    st.sidebar.subheader("Training Hyperparams")
    optsel = st.sidebar.selectbox("Optimizer", ["SGD","SGD with Momentum","Adam"])
    lr = st.sidebar.number_input("Learning Rate",1e-5,1.0,value=0.01,format="%.5f")
    if optsel=="SGD with Momentum": momentum=st.sidebar.slider("Momentum",0.0,0.99,0.9,0.01)
    if optsel=="Adam":
        beta1=st.sidebar.slider("Beta1",0.0,0.999,0.9,0.01)
        beta2=st.sidebar.slider("Beta2",0.0,0.999,0.999,0.001)
        eps=st.sidebar.number_input("Epsilon",1e-8,1e-4,1e-8,format="%.1e")
    steps=st.sidebar.slider("Epochs",1,50,10)

    # Train
    if st.sidebar.button("Start Training"):
        # Prepare labels
        if task=="Classification":
            Y=np.eye(st.session_state.output_dim)[y]
        else: Y=y.reshape(-1,1)
        # Build fresh model and optimizer
        mlp = MLP(st.session_state.layer_sizes, activation=activation)
        if optsel=="SGD": optim=SGD(mlp.parameters(),lr=lr)
        elif optsel=="SGD with Momentum": optim=SGDMomentum(mlp.parameters(),momentum=momentum,lr=lr)
        else: optim=Adam(mlp.parameters(),lr=lr,betas=(beta1,beta2),eps=eps)

        # Track history and trajectories
        history={"Loss":[]}
        if task=="Classification": history.update({"Accuracy":[],"Precision":[],"Recall":[],"F1":[]})
        else: history.update({"MSE":[]})
        trajs={name:[] for name in [optsel]}

        for ep in range(steps):
            x_t=ValueTensor(X); y_t=ValueTensor(Y)
            preds=mlp(x_t)
            loss = (cross_entropy(y_t,preds) if task=="Classification" else mse(y_t,preds))
            loss.backward()
            history["Loss"].append(float(loss.data.squeeze()))
            # metrics
            pred_vals=preds.data
            if task=="Classification":
                y_pred=pred_vals.argmax(axis=1)
                history["Accuracy"].append(accuracy_score(y,y_pred))
                history["Precision"].append(precision_score(y,y_pred,average="macro",zero_division=0))
                history["Recall"].append(recall_score(y,y_pred,average="macro",zero_division=0))
                history["F1"].append(f1_score(y,y_pred,average="macro",zero_division=0))
            else:
                y_pred=pred_vals.squeeze()
                history["MSE"].append(mean_squared_error(y,y_pred))
            # record first-layer weight coords
            w0=mlp.layers[0].w.data[0,0]; w1=mlp.layers[0].w.data[0,1]
            trajs[optsel].append((w0,w1))
            optim.step(); mlp.zero_grad()

        # persist
        st.session_state.history=history
        st.session_state.trajs=trajs
        # plot metrics
        df=pd.DataFrame(history,index=np.arange(1,steps+1))
        fig=go.Figure()
        fig.add_trace(go.Scatter(x=df.index,y=df["Loss"],name="Loss",yaxis="y1"))
        for m in history:
            if m!="Loss": fig.add_trace(go.Scatter(x=df.index,y=df[m],name=m,yaxis="y2"))
        fig.update_layout(xaxis_title="Epoch",yaxis=dict(title="Loss"),yaxis2=dict(title="Metrics",overlaying="y",side="right"),legend=dict(x=0.5,y=1.1,orientation="h"))
        st.plotly_chart(fig,use_container_width=True)

# — Loss Landscape Page —
elif page=="Loss Landscape":
    st.header("🌄 Loss Landscape & Optimizer Paths")
    # Optimizer selection and hyperparams
    optim_names=st.multiselect("Optimizers",["SGD","SGD with Momentum","Adam"],default=["SGD","Adam"])
    lr=st.sidebar.slider("Learning Rate",1e-3,1.0,0.1)
    steps=st.sidebar.slider("Optimization Steps",5,200,50)
    momentum=st.sidebar.slider("Momentum (SGD+M)",0.0,0.99,0.9)
    beta1=st.sidebar.slider("Beta1 (Adam)",0.0,0.999,0.9)
    beta2=st.sidebar.slider("Beta2 (Adam)",0.0,0.999,0.999)
    eps=st.sidebar.number_input("Epsilon (Adam)",1e-8,1e-4,1e-8,format="%.1e")

    # grid setup
    layer_sizes=st.session_state.layer_sizes
    activation=st.session_state.activation
    X=st.session_state.X; y=st.session_state.y
    w0_vals,w1_vals=[],[]
    pts=50; delta=1.0
    # record landscape around initial weight
    initial_w=st.session_state.initial_weights[0]  # first weight matrix
    grid0=np.linspace(initial_w[0,0]-delta,initial_w[0,0]+delta,pts)
    grid1=np.linspace(initial_w[0,1]-delta,initial_w[0,1]+delta,pts)
    Z=np.zeros((pts,pts))
    for i,v0 in enumerate(grid0):
        for j,v1 in enumerate(grid1):
            w_tmp=initial_w.copy(); w_tmp[0,0],w_tmp[0,1]=v0,v1
            mlp2=MLP(layer_sizes,activation)
            mlp2.layers[0].w.data=w_tmp
            x_t=ValueTensor(X)
            if task=="Classification": y_t=ValueTensor(np.eye(st.session_state.output_dim)[y]); loss=cross_entropy(y_t,mlp2(x_t))
            else: y_t=ValueTensor(y.reshape(-1,1)); loss=mse(y_t,mlp2(x_t))
            Z[i,j]=float(loss.data.mean())
    # plot contour
    fig=go.Figure(go.Contour(z=Z,x=grid0,y=grid1,colorscale="Viridis",contours=dict(showlabels=True)))

    # overlay paths for each optimizer
    for name in optim_names:
        # run small training to get path
        mlp3=MLP(layer_sizes,activation)
        if name=="SGD": optim3=SGD(mlp3.parameters(),lr=lr)
        elif name=="SGD with Momentum": optim3=SGDMomentum(mlp3.parameters(),momentum=momentum,lr=lr)
        else: optim3=Adam(mlp3.parameters(),lr=lr,betas=(beta1,beta2),eps=eps)
        path=[]
        for _ in range(steps):
            x_t=ValueTensor(X); y_t=ValueTensor(np.eye(st.session_state.output_dim)[y] if task=="Classification" else y.reshape(-1,1))
            pred=mlp3(x_t)
            loss=cross_entropy(y_t,pred) if task=="Classification" else mse(y_t,pred)
            loss.backward()
            w0=mlp3.layers[0].w.data[0,0]; w1=mlp3.layers[0].w.data[0,1]
            path.append((w0,w1))
            optim3.step(); mlp3.zero_grad()
        pts_arr=np.array(path)
        fig.add_trace(go.Scatter(x=pts_arr[:,0],y=pts_arr[:,1],mode="lines+markers",name=name))

    fig.update_layout(title="Landscape & Optimizer Paths",xaxis_title="w[0,0]",yaxis_title="w[0,1]",margin=dict(l=20,r=20,t=40,b=20))
    st.plotly_chart(fig,use_container_width=True)
