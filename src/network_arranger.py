import torch
import numpy as np
import src.loss_functions as lf


def get_index_maps(G):
    node_to_index = {}
    index_to_node = {}
    current_i = 0
    for node in G.nodes:
        node_to_index[str(node)] = current_i
        index_to_node[current_i] = str(node)
        current_i += 1
    return node_to_index, index_to_node


def get_connection_matrix(G, node_to_index):
    N = G.number_of_nodes()
    connected = torch.zeros((N, N), dtype=torch.float32)
    for edge in G.edges:
        connected[node_to_index[edge[0]], node_to_index[edge[1]]] = 1.
        connected[node_to_index[edge[1]], node_to_index[edge[0]]] = 1.
    return connected


def arrange_graph(G, T = 100, eps = 1e-1, optim = None, verbose = True, general_loss = None, connected_loss = None):
    N = G.number_of_nodes()
    node_to_index, index_to_node = get_index_maps(G)
    
    Ig = torch.ones((N,N), dtype=torch.float32)-torch.eye(N, N)
    Ic = get_connection_matrix(G, node_to_index)

    # Init node coordinates
    node_x = torch.normal(0, np.sqrt(N), (N,1), requires_grad=True, dtype=torch.float32)
    node_y = torch.normal(0, np.sqrt(N), (N,1), requires_grad=True, dtype=torch.float32)
    
    if optim is None:
        optim = torch.optim.Adam([node_x, node_y], lr=1.0)
    if general_loss is None:
        general_loss =  lf.div_log
    if connected_loss is None:
        connected_loss = lf.div_lin

    stats = {} 
    stats['node_to_index'] = node_to_index
    stats['connection_matrix'] = Ic
    if verbose:
        stats['loss'] = []
        stats['positions'] = np.zeros((T, N, 2))
        stats['positions'][0, : , 0] = node_x.clone().detach()[:, 0]
        stats['positions'][0, : , 1] = node_y.clone().detach()[:, 0]

    for t in range(T-1):
        R2 = torch.pow(node_x-node_x.T, 2) + torch.pow(node_y-node_y.T, 2)
        Gen = R2*Ig
        Con = R2*Ic
        
        Eg = general_loss(Gen/N+eps)
        Ec = connected_loss(Con+eps)

        E = Eg + Ec
        L = torch.mean(E)

        optim.zero_grad()
        L.backward()
        optim.step()
        
        if verbose: 
            stats['loss'].append(L.item())
            stats['positions'][t+1, : , 0] = node_x.clone().detach()[:, 0]
            stats['positions'][t+1, : , 1] = node_y.clone().detach()[:, 0]

    # Get final Loss
    R2 = torch.pow(node_x-node_x.T, 2) + torch.pow(node_y-node_y.T, 2)
    Gen = R2*Ig
    Con = R2*Ic
    
    Eg = general_loss(Gen/N+eps)
    Ec = connected_loss(Con+eps)

    E = Eg + Ec
    L = torch.mean(E)
    if verbose: 
        stats['loss'].append(L.item())

    return (node_x.clone().detach()[:,0],
            node_y.clone().detach()[:,0],
            stats)

