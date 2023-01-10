import networkx as nx
import numpy as np
import plotly.graph_objects as go
import imageio
import os
import plotly
import json

def read_graph_from_aoc(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    G = nx.Graph()
    for line in lines:
        words = line.replace("\n", "").replace(";", "").replace(",", "").split(" ")
        name = words[1]
        flow_rate = int(words[4].split("=")[1])
        connections = words[9:]
        G.add_node(name)
        G.nodes[name]['fr'] = flow_rate

    for line in lines:
        words = line.replace("\n", "").replace(";", "").replace(",", "").split(" ")
        name = words[1]
        flow_rate = int(words[4].split("=")[1])
        connections = words[9:]
        for con in connections:
            G.add_edge(name, con)
    return G


def graph_to_plotly(node_pos, connection_matrix):
    # node_pos (T, N, D)
    (T, N, D) = node_pos.shape
    
    edge_list ={'x': [], 'y': []}
    for t in range(T):
        edge_list['x'].append([])
        edge_list['y'].append([])
        for n in range(N):
            for n2 in range(n, N):
                if connection_matrix[n, n2] == 0: continue
                edge_list['x'][t].append(node_pos[t, n, 0])
                edge_list['x'][t].append(node_pos[t, n2, 0])
                edge_list['x'][t].append(None)
                
                edge_list['y'][t].append(node_pos[t, n, 1])
                edge_list['y'][t].append(node_pos[t, n2, 1])
                edge_list['y'][t].append(None)
    
    node_list = {'x': [], 'y': []}
    for t in range(T):
        node_list['x'].append([])
        node_list['y'].append([])
        for n in range(N):
            node_list['x'][t].append(node_pos[t, n, 0])
            node_list['y'][t].append(node_pos[t, n, 1])
    
    return node_list, edge_list


def animate_network(node_pos, edge_pos, nodesize= 10, edge_width=2, width=600, height=600, x_range=None, y_range=None, minimum_layout=False):
    T = len(node_pos['x'])
    N = len(node_pos['x'][0])
    if x_range is None:
        minx = np.min(node_pos['x'][-1])
        maxx = np.max(node_pos['x'][-1])
        x_range = [minx-(maxx-minx)/10, maxx+(maxx-minx)/10]
    if y_range is None:
        miny = np.min(node_pos['y'][-1])
        maxy = np.max(node_pos['y'][-1])
        y_range = [miny-(maxy-miny)/10, maxy+(maxy-miny)/10]
        
    fig = go.Figure(
        data=[
            go.Scatter(x=edge_pos['x'][0], y=edge_pos['y'][0],
                        mode="lines",
                        name="Edges",
                        line=dict(color="#FE5F55", width=edge_width)),
            go.Scatter(x=node_pos['x'][0], y=node_pos['y'][0],
                        mode="markers",
                        name="Nodes",
                        marker=dict(color="#4F6367", size=nodesize))
                        ],
        layout=go.Layout(width=width, height=height,
                        xaxis=dict(range=x_range, autorange=False, zeroline=False),
                        yaxis=dict(range=y_range, autorange=False, zeroline=False),
                        title="Self Arranging Graf",
                        hovermode="closest",
                        updatemenus=[dict(type="buttons",
                                        buttons=[dict(label="Play",
                                                        method="animate",
                                                        args=[None])])
                                    ] if not minimum_layout else []
                        ),
        frames=[go.Frame(
            data=[
                go.Scatter(
                x= edge_pos['x'][k],
                y= edge_pos['y'][k],
                mode="lines",
                name="Edges",
                line=dict(color="#FE5F55", width=edge_width)),
                go.Scatter(
                x= node_pos['x'][k],
                y= node_pos['y'][k],
                mode="markers",
                name="Nodes",
                marker=dict(color="#4F6367", size=nodesize))
            ]) for k in range(T)]
    )

    return fig



def save_animation_as_gif(fig, file_name='example.gif'):
    temp_file = "temp.png"
    try:
        with imageio.get_writer(file_name, mode='I') as writer:
            for frame in fig.frames:
                frame['layout'] = fig['layout']
                new_fig = plotly.io.from_json(json.dumps(frame.to_plotly_json()))
                new_fig.write_image(temp_file)
                writer.append_data( imageio.imread(temp_file))
    finally:
        os.remove(temp_file)



def plot_loss(losses, line_width=2, width=1000, height=600):  
    fig = go.Figure(
        data=[
            go.Scatter(x=[i for i in range(len(losses))], y=losses,
                        mode="lines",
                        name="Loss",
                        line=dict(color="#FE5F55", width=line_width)
            )
        ],
        layout=go.Layout(width=width, height=height,
                        title="Loss of Self Arranging Graf"
                        )
    )
    return fig

