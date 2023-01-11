import torch
import numpy as np
import src.loss_functions as lf


def get_index_maps(G):
    node_to_index = {}
    index_to_node = {}
    current_i = 0
    for node in G.nodes:
        node_to_index[node] = current_i
        index_to_node[current_i] = node
        current_i += 1
    return node_to_index, index_to_node


def get_connection_matrix(G, node_to_index):
    N = G.number_of_nodes()
    connected = torch.zeros((N, N), dtype=torch.float32)
    for edge in G.edges:
        connected[node_to_index[edge[0]], node_to_index[edge[1]]] = 1.
        connected[node_to_index[edge[1]], node_to_index[edge[0]]] = 1.
    return connected


def arrange_graph(G, T = 100, eps = 1e-1, general_loss = None, connected_loss = None):
    """
    Returns two vectors with the x and y positions of the nodes in G. 
    G is expected to be a graph from the networkx library.
    The funstions also returns a stats dictionary with the information about the training process and 
    a mapping from the node names to the indices in the returned positions. 
    """
    N = G.number_of_nodes()
    node_to_index, index_to_node = get_index_maps(G)

    # Set general and connected mask
    Mg = torch.ones((N,N), dtype=torch.float32)-torch.eye(N, N)
    Mc = get_connection_matrix(G, node_to_index)

    # Set optimizer and loss functions    
    if general_loss is None:
        general_loss =  lf.div_log
    if connected_loss is None:
        connected_loss = lf.div_lin

    # Init node coordinates
    node_x = torch.normal(0, np.sqrt(N), (N,1), requires_grad=True, dtype=torch.float32)
    node_y = torch.normal(0, np.sqrt(N), (N,1), requires_grad=True, dtype=torch.float32)
    optim = torch.optim.Adam([node_x, node_y], lr=1.0)

    
    # Init stats
    stats = {} 
    stats['node_to_index'] = node_to_index
    stats['connection_matrix'] = Mc
    stats['loss'] = []
    stats['positions'] = np.zeros((T, N, 2))
    current_x = node_x.clone().detach()[:, 0]
    current_y = node_y.clone().detach()[:, 0]
    stats['positions'][0, : , 0] = current_x
    stats['positions'][0, : , 1] = current_y
    stats['movement'] = np.zeros((T,))
    movement = []

    # Training
    for t in range(T-1):
        # Calculate Loss
        R2 = torch.pow(node_x-node_x.T, 2) + torch.pow(node_y-node_y.T, 2)
        Gen = R2*Mg
        Con = R2*Mc
        
        Lg = general_loss(Gen/N+eps)
        Lc = connected_loss(Con+eps)

        L = torch.mean(Lg + Lc)

        # Update positions
        optim.zero_grad()
        L.backward()
        optim.step()
        
        # Save stats
        prev_x = current_x
        prev_y = current_y
        current_x = node_x.clone().detach()[:, 0]
        current_y = node_y.clone().detach()[:, 0]
        
        stats['loss'].append(L.item())
        stats['positions'][t+1, : , 0] = current_x
        stats['positions'][t+1, : , 1] = current_y
        stats['movement'][t+1] =torch.max(torch.sqrt((current_x-prev_x)**2+(current_y-prev_y)**2))

    # Get final Loss
    R2 = torch.pow(node_x-node_x.T, 2) + torch.pow(node_y-node_y.T, 2)
    Gen = R2*Mg
    Con = R2*Mc
    Lg = general_loss(Gen/N+eps)
    Lc = connected_loss(Con+eps)
    L = torch.mean(Lg + Lc)

    # Save stats
    stats['loss'].append(L.item())

    return (node_x.clone().detach()[:,0],
            node_y.clone().detach()[:,0],
            stats)


def arrange_graph_until_stable(G, step_limit = 0.01, eps = 1e-1, general_loss = None, connected_loss = None):
    """
    Returns two vectors with the x and y positions of the nodes in G. 
    G is expected to be a graph from the networkx library.
    The funstions also returns a stats dictionary with the information about the training process and 
    a mapping from the node names to the indices in the returned positions. 
    """
    N = G.number_of_nodes()
    node_to_index, index_to_node = get_index_maps(G)

    # Set general and connected mask
    Mg = torch.ones((N,N), dtype=torch.float32)-torch.eye(N, N)
    Mc = get_connection_matrix(G, node_to_index)

    # Set optimizer and loss functions    
    if general_loss is None:
        general_loss =  lf.div_log
    if connected_loss is None:
        connected_loss = lf.div_lin

    # Init node coordinates
    node_x = torch.normal(0, np.sqrt(N), (N,1), requires_grad=True, dtype=torch.float32)
    node_y = torch.normal(0, np.sqrt(N), (N,1), requires_grad=True, dtype=torch.float32)
    optim = torch.optim.Adam([node_x, node_y], lr=1.0)

    
    # Init stats
    stats = {} 
    stats['node_to_index'] = node_to_index
    stats['connection_matrix'] = Mc
    stats['loss'] = []
    current_x = node_x.clone().detach()[:, 0]
    current_y = node_y.clone().detach()[:, 0]
    stats['movement'] = []
    movement = []

    # Training
    while True:
        # Calculate Loss
        R2 = torch.pow(node_x-node_x.T, 2) + torch.pow(node_y-node_y.T, 2)
        Gen = R2*Mg
        Con = R2*Mc
        
        Lg = general_loss(Gen/N+eps)
        Lc = connected_loss(Con+eps)

        L = torch.mean(Lg + Lc)

        # Update positions
        optim.zero_grad()
        L.backward()
        optim.step()
        
        # Save stats
        prev_x = current_x
        prev_y = current_y
        current_x = node_x.clone().detach()[:, 0]
        current_y = node_y.clone().detach()[:, 0]
        max_step = torch.max(torch.sqrt((current_x-prev_x)**2+(current_y-prev_y)**2)).item()

        stats['loss'].append(L.item())
        stats['movement'].append(max_step)
        if max_step < step_limit:
            break

    # Get final Loss
    R2 = torch.pow(node_x-node_x.T, 2) + torch.pow(node_y-node_y.T, 2)
    Gen = R2*Mg
    Con = R2*Mc
    Lg = general_loss(Gen/N+eps)
    Lc = connected_loss(Con+eps)
    L = torch.mean(Lg + Lc)

    # Save stats
    stats['loss'].append(L.item())

    return (node_x.clone().detach()[:,0],
            node_y.clone().detach()[:,0],
            stats)
