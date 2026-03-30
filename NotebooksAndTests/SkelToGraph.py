import numpy as np
import pandas as pd
import networkx as nx
from scipy.spatial import cKDTree
from scipy.sparse.csgraph import dijkstra, connected_components
from scipy.sparse import csr_matrix
import plotly.graph_objects as go
from scipy.interpolate import splprep, splev
from collections import defaultdict


def Skel2Graph(skel: np.ndarray, k_neighbors: int = 6):
    points = np.array(np.where(skel)).T
    print(f"Total points: {len(points)}")
    tree = cKDTree(points)
    distances, indices = tree.query(points, k=k_neighbors)

    n = len(points)
    adj_matrix = np.zeros((n, n))
    for i in range(n):
        for j in indices[i][1:]:   # skip self
            d = distances[i][np.where(indices[i] == j)[0][0]]
            adj_matrix[i, j] = d
    graph = csr_matrix(adj_matrix)
    n_components, labels = connected_components(graph, directed=False)
    return graph, points, n_components, labels

def Graph2CSV(points: np.ndarray,
                graph: csr_matrix, 
                points_filename: str = 'points_float32.csv', 
                edges_filename: str = 'graph_edges.csv'):
    rows, cols = graph.nonzero()
    edges = np.column_stack((rows, cols))
    # Remove duplicate (i,j) with i>jfor undirected graph
    edges = edges[edges[:,0] < edges[:,1]]
    
    points_f32 = points.astype(np.float32)
    
    df_points = pd.DataFrame(points_f32, columns=['x', 'y', 'z'])
    df_points.to_csv(points_filename, index=False)
    df_points.to_csv(points_filename + '.gz', index=False)
    
    df_edges = pd.DataFrame(edges, columns=['node1', 'node2'])
    df_edges.to_csv(edges_filename, index=False)
    df_edges.to_csv(edges_filename + '.gz', index=False)

def CSV2Graph(points_filename: str = 'points_float32.csv',
                edges_filename: str = 'graph_edges.csv') -> tuple[csr_matrix, np.ndarray]:
    df_points = pd.read_csv(points_filename)
    points = df_points[['x', 'y', 'z']].values.astype(np.float32)
    
    df_edges = pd.read_csv(edges_filename)
    edges = df_edges[['node1', 'node2']].values
    
    n = len(points)
    adj_matrix = np.zeros((n, n))
    for i, j in edges:
        d = np.linalg.norm(points[i] - points[j])
        adj_matrix[i, j] = d
        adj_matrix[j, i] = d  # undirected
    graph = csr_matrix(adj_matrix)
    return graph, points

def CSR2NetworkX(sparse_graph: csr_matrix, points: np.ndarray) -> nx.Graph:
    G = nx.Graph()
    N = points.shape[0]

    G.add_nodes_from((i, {'pos': points[i]}) for i in range(N))

    coo = sparse_graph.tocoo()
    rows, cols, data = coo.row, coo.col, coo.data
    mask = (rows < cols) & (data > 0)
    u = rows[mask]
    v = cols[mask]
    w = data[mask]

    G.add_weighted_edges_from(zip(u, v, w))

    return G

def get_cycle_length(G: nx.Graph, cycle: list) -> float | None:
    total = 0.0
    n = len(cycle)
    for i in range(n):
        u, v = cycle[i], cycle[(i+1) % n]
        if G.has_edge(u, v):
            total += G[u][v]['weight']
        elif G.has_edge(v, u):
            total += G[v][u]['weight']
        else:
            return None
    return total

def get_valid_cycles(G: nx.Graph) -> tuple[list[list[int]], list[float]]:
    cycles = nx.cycle_basis(G)
    valid = []
    lengths = []
    for cycle in cycles:
        length = get_cycle_length(G, cycle)
        if length is not None:
            valid.append(cycle)
            lengths.append(length)
    return valid, lengths

def filter_cycles_by_size_and_length(cycles: list[list[int]], 
                                    lengths: list[float], 
                                    min_edges: int | None, 
                                    min_length: float | None) -> tuple[list[list[int]], list[float]]:
    large = []
    large_lengths = []
    for cyc, l in zip(cycles, lengths):
        # Check if cycle qualifies as LARGE
        size_ok = (min_edges is None or len(cyc) >= min_edges)
        length_ok = (min_length is None or l >= min_length)
        if size_ok and length_ok:
            large.append(cyc)
            large_lengths.append(l)
    return large, large_lengths

def break_cycles(G: nx.Graph, cycles_to_break: list[list[int]]) -> list[tuple[int, int]]:
    removed = []
    for cycle in cycles_to_break:
        max_len = -1
        max_edge = None
        for i in range(len(cycle)):
            u, v = cycle[i], cycle[(i+1) % len(cycle)]
            if G.has_edge(u, v):
                w = G[u][v]['weight']
            elif G.has_edge(v, u):
                w = G[v][u]['weight']
            else:
                continue
            if w > max_len:
                max_len = w
                max_edge = (min(u, v), max(u, v))
        if max_edge and G.has_edge(*max_edge):
            G.remove_edge(*max_edge)
            removed.append(max_edge)
    return removed

def detect_cycles_filtered(G: nx.Graph, min_cycle_edges: int | None = None, min_cycle_length: float | None = None,
                           break_small_cycles: bool = False, verbose: bool = True) -> dict:
    all_cycles, all_lengths = get_valid_cycles(G)
    if verbose:
        print(f"Total cycles detected: {len(all_cycles)}")
        if all_cycles:
            print(f"Cycle sizes: {[len(c) for c in all_cycles]}")
            print(f"Cycle lengths: {[f'{l:.2f}' for l in all_lengths]}")

    large_cycles, large_lengths = filter_cycles_by_size_and_length(
        all_cycles, all_lengths, min_cycle_edges, min_cycle_length)
    if verbose:
        print(f"Cycles meeting criteria: {len(large_cycles)}")

    removed_edges = []
    if break_small_cycles:
        small_cycles = []
        for cyc in all_cycles:
            if cyc not in large_cycles:
                small_cycles.append(cyc)
        if small_cycles:
            if verbose:
                print(f"\nBreaking {len(small_cycles)} small cycles...")
            removed_edges = break_cycles(G, small_cycles)
            if verbose and removed_edges:
                print(f"Removed {len(removed_edges)} edges to break cycles.")

    return {
        'total_cycles': len(all_cycles),
        'cycle_sizes': [len(c) for c in all_cycles],
        'cycle_lengths': all_lengths,
        'large_cycles': large_cycles,
        'large_cycle_lengths': large_lengths,
        'removed_edges': removed_edges
    }

def plot_graph_3d(graph, points, output_file='graph_3d.html',
                  node_size=3, edge_width=1, opacity=0.8,
                  node_color='blue', edge_color='red',
                  title='3D Graph Visualization'):
    """
    Create an interactive 3D plot of a graph using Plotly.

    Parameters
    ----------
    graph : networkx.Graph or scipy.sparse.csr_matrix
        The graph to plot. If a CSR matrix, it is assumed to be symmetric.
    points : numpy.ndarray, shape (N, 3)
        3D coordinates of the nodes.
    output_file : str, optional
        Path to save the HTML file.
    node_size : int, optional
        Size of the node markers.
    edge_width : int, optional
        Width of the edge lines.
    opacity : float, optional
        Opacity of the edges (0 to 1).
    node_color : str, optional
        Color of nodes (any Plotly color specification).
    edge_color : str, optional
        Color of edges.
    title : str, optional
        Title of the plot.

    Returns
    -------
    fig : plotly.graph_objects.Figure
        The figure object (also saves HTML file).
    """
    # Determine edges from input graph
    if isinstance(graph, csr_matrix):
        rows, cols = graph.nonzero()
        edges = list(zip(rows, cols))
        # Keep only one direction (undirected)
        edges = [(u, v) for u, v in edges if u < v]
    elif isinstance(graph, nx.Graph):
        edges = list(graph.edges())
    else:
        raise TypeError("graph must be a networkx.Graph or scipy.sparse.csr_matrix")

    # Prepare edge lines: each edge becomes a line segment with None between them
    x_lines, y_lines, z_lines = [], [], []
    for u, v in edges:
        x_lines.extend([points[u, 0], points[v, 0], None])
        y_lines.extend([points[u, 1], points[v, 1], None])
        z_lines.extend([points[u, 2], points[v, 2], None])

    # Edge trace
    edge_trace = go.Scatter3d(
        x=x_lines, y=y_lines, z=z_lines,
        mode='lines',
        line=dict(color=edge_color, width=edge_width),
        opacity=opacity,
        hoverinfo='none',
        name='edges'
    )

    # Node trace
    node_trace = go.Scatter3d(
        x=points[:, 0], y=points[:, 1], z=points[:, 2],
        mode='markers',
        marker=dict(size=node_size, color=node_color, opacity=1.0),
        hoverinfo='text',
        text=[f'Node {i}' for i in range(len(points))],
        name='nodes'
    )

    # Create figure
    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectmode='data'
        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )

    # Save to HTML
    fig.write_html(output_file)
    print(f"Interactive 3D plot saved to {output_file}")

    return fig

def draw_graph_graphviz(graph, points=None, layout='neato', output_file='graph.gv.pdf',
                        node_color='blue', edge_color='gray', prog='neato', dpi=300):
    """
    Draw a graph using Graphviz with adjustable resolution.

    Parameters
    ----------
    graph : networkx.Graph or scipy.sparse.csr_matrix
        Graph to visualize.
    points : numpy.ndarray, shape (N, 2 or 3), optional
        Node coordinates. If provided, they are used as fixed positions (first two dimensions).
        If None, the layout engine computes positions.
    layout : str, {'neato', 'fdp', 'dot', 'circo', 'twopi'}
        Graphviz layout engine used when `points` is None.
    output_file : str
        Output file name (extension determines format, e.g., .png, .pdf, .svg).
    node_color : str
        Node color.
    edge_color : str
        Edge color.
    prog : str
        Graphviz program used for rendering when `points` is provided (default 'neato').
    dpi : int
        Resolution for raster formats (PNG, JPG, etc.). Ignored for vector formats (PDF, SVG).
    """
    try:
        import pygraphviz as pgv
    except ImportError:
        print("pygraphviz not installed. Install with 'pip install pygraphviz'.")
        return

    # Convert CSR to NetworkX if needed
    if isinstance(graph, csr_matrix):
        G = nx.Graph()
        n = graph.shape[0]
        G.add_nodes_from(range(n))
        coo = graph.tocoo()
        rows, cols = coo.row, coo.col
        for i, j in zip(rows, cols):
            if i < j:
                G.add_edge(i, j)
    else:
        G = graph

    # Create a pygraphviz AGraph
    A = pgv.AGraph(directed=False, strict=False)

    # Add nodes with optional coordinates (use first two dimensions)
    for node in G.nodes():
        if points is not None and points.shape[1] >= 2:
            # Use x,y as positions (Graphviz expects "x,y!" format)
            A.add_node(node, pos=f"{points[node,0]},{points[node,1]}!")
        else:
            A.add_node(node)

    # Add edges
    for u, v in G.edges():
        A.add_edge(u, v)

    # Set colors
    A.node_attr['color'] = node_color
    A.edge_attr['color'] = edge_color

    # Determine output format and DPI arguments
    ext = output_file.split('.')[-1].lower()
    args = ''
    if ext in ('png', 'jpg', 'jpeg', 'tiff', 'bmp'):
        args = f'-Gdpi={dpi}'

    # Draw
    if points is None:
        # Use layout engine to compute positions, then render
        A.layout(prog=layout)
        A.draw(output_file, args=args)
    else:
        # Fixed positions: render with prog (e.g., neato) and pass args
        A.draw(output_file, prog=prog, args=args)

    print(f"Graph saved to {output_file}")

def draw_graph_graphviz_cycles(
    graph,
    cycle_groups=None,
    output_file='graph_cycles.svg',
    node_color='lightgray',
    edge_color='lightgray',
    base_edge_width=1,
    cycle_edge_width=3,
    multi_cycle_edge_width=5,
    prog='neato',
    dpi=300
):
    """
    Graphviz rendering that EXACTLY matches original layout behavior,
    with added cycle coloring.
    """

    try:
        import pygraphviz as pgv
    except ImportError:
        print("pygraphviz not installed. Install with 'pip install pygraphviz'.")
        return

    import networkx as nx
    from scipy.sparse import csr_matrix

    # --- Convert graph ---
    if isinstance(graph, csr_matrix):
        G = nx.Graph()
        coo = graph.tocoo()
        for u, v, w in zip(coo.row, coo.col, coo.data):
            if u < v and w > 0:
                G.add_edge(int(u), int(v))
    elif isinstance(graph, nx.Graph):
        G = graph
    else:
        raise TypeError("graph must be networkx or csr_matrix")

    # --- Build AGraph ---
    A = pgv.AGraph(directed=False, strict=False)

    # Nodes
    for n in G.nodes():
        A.add_node(n, color=node_color, style="filled", fillcolor="white")

    # Edges
    for u, v in G.edges():
        A.add_edge(u, v, color=edge_color, penwidth=str(base_edge_width))

    # --- Cycle coloring ---
    palette = [
        "red", "blue", "green4", "orange", "purple",
        "brown", "deeppink3", "cyan4"
    ]

    edge_usage = {}

    if cycle_groups is not None:
        for gid, grp in enumerate(cycle_groups):
            color = palette[gid % len(palette)]

            for (u, v) in grp["edges"]:
                key = (min(u, v), max(u, v))
                edge_usage[key] = edge_usage.get(key, 0) + 1

                if A.has_edge(u, v):
                    e = A.get_edge(u, v)
                    e.attr["color"] = color
                    e.attr["penwidth"] = str(cycle_edge_width)

    # Multi-cycle edges
    for (u, v), count in edge_usage.items():
        if count > 1 and A.has_edge(u, v):
            e = A.get_edge(u, v)
            e.attr["color"] = "black"
            e.attr["penwidth"] = str(multi_cycle_edge_width)

    # --- Important: DO NOT call layout() ---
    # Let draw() behave like your original function

    ext = output_file.split('.')[-1].lower()
    args = f'-Gdpi={dpi}' if ext in ('png','jpg','jpeg','tiff','bmp') else ''

    A.draw(output_file, prog=prog, args=args)

    print(f"Graph saved to {output_file}")




def extract_cycle_basis(G: nx.Graph) -> list[list[int]]:
    return [list(map(int, c)) for c in nx.cycle_basis(G)]


def cycle_edges(cycle: list[int]) -> set[tuple[int, ...]]:
    return {
        tuple(sorted((cycle[i], cycle[(i + 1) % len(cycle)])))
            for i in range(len(cycle))
    }


def cycle_length(G: nx.Graph, cycle: list[int]) -> float | None:
    total = 0.0
    for i in range(len(cycle)):
        u, v = cycle[i], cycle[(i + 1) % len(cycle)]
        if G.has_edge(u, v):
            total += G[u][v].get("weight", 1.0)
        elif G.has_edge(v, u):
            total += G[v][u].get("weight", 1.0)
        else:
            return None
    return float(total)


def build_cycle_records(G: nx.Graph, cycles: list[list[int]]) -> list[dict]:
    return [{
        "cycle_id": i,
        "cycle": c,
        "size": len(c),
        "length": cycle_length(G, c),
        "nodes": set(c),
        "edges": cycle_edges(c),
    } for i, c in enumerate(cycles)]

def filter_cycles(records: list[dict],
                  min_cycle_edges: int,
                  keep_triangles_touching_large: bool,
                  triangle_touch_nodes: int) -> list[dict]:

    large_sets = [r["nodes"] for r in records if r["size"] >= min_cycle_edges]

    def keep(r):
        if r["size"] >= min_cycle_edges:
            return True
        if r["size"] == 3 and keep_triangles_touching_large:
            return any(len(r["nodes"] & s) >= triangle_touch_nodes for s in large_sets)
        return False

    return [r for r in records if keep(r)]

def sort_cycles(records: list[dict]) -> list[dict]:
    return sorted(records, key=lambda r: (r["size"], r["length"]), reverse=True)


def find_best_group(groups: list[dict], nodes: set[int]) -> tuple[int | None, int]:
    best_gid, best_overlap = None, 0
    for gid, g in enumerate(groups):
        overlap = len(nodes & g["nodes"])
        if overlap > best_overlap:
            best_gid, best_overlap = gid, overlap
    return best_gid, best_overlap


def merge_into_group(group: dict, record: dict):
    group["cycles"].append(record)
    group["nodes"].update(record["nodes"])
    group["edges"].update(record["edges"])
    group["lengths"].append(record["length"])


def create_group(gid: int, record: dict) -> dict:
    return {
        "group_id": gid,
        "cycles": [record],
        "nodes": set(record["nodes"]),
        "edges": set(record["edges"]),
        "lengths": [record["length"]],
    }


def greedy_group_cycles(records: list[dict], min_shared_nodes: int):
    groups = []
    cycle_to_group = {}

    for r in records:
        gid, overlap = find_best_group(groups, r["nodes"])

        if gid is not None and overlap >= min_shared_nodes:
            merge_into_group(groups[gid], r)
            cycle_to_group[r["cycle_id"]] = gid
        else:
            gid = len(groups)
            groups.append(create_group(gid, r))
            cycle_to_group[r["cycle_id"]] = gid

    return groups, cycle_to_group


def finalize_groups(groups: list[dict]):
    for g in groups:
        g["n_cycles"] = len(g["cycles"])
        g["n_nodes"] = len(g["nodes"])
        g["n_edges"] = len(g["edges"])
        g["total_cycle_length"] = float(np.nansum(g["lengths"]))
        g["max_cycle_length"] = float(np.nanmax(g["lengths"])) if g["lengths"] else 0.0
        g["cycle_ids"] = [r["cycle_id"] for r in g["cycles"]]
        g["cycle_sizes"] = [r["size"] for r in g["cycles"]]
        g["representative_cycle"] = max(
            g["cycles"], key=lambda r: (r["size"], r["length"])
        )["cycle"]


def build_memberships(records, cycle_to_group):
    node_to_cycles = defaultdict(list)
    node_to_groups = defaultdict(list)
    edge_to_cycles = defaultdict(list)
    edge_to_groups = defaultdict(list)

    for r in records:
        gid = cycle_to_group.get(r["cycle_id"], -1)

        for n in r["nodes"]:
            node_to_cycles[n].append(r["cycle_id"])
            if gid != -1:
                node_to_groups[n].append(gid)

        for e in r["edges"]:
            edge_to_cycles[e].append(r["cycle_id"])
            if gid != -1:
                edge_to_groups[e].append(gid)

    return node_to_cycles, node_to_groups, edge_to_cycles, edge_to_groups


def build_cycle_df(records, cycle_to_group):
    return pd.DataFrame([{
        "cycle_id": r["cycle_id"],
        "group_id": cycle_to_group.get(r["cycle_id"], -1),
        "accepted": r["cycle_id"] in cycle_to_group,
        "size": r["size"],
        "length": r["length"],
        "nodes": sorted(r["nodes"]),
        "edges": sorted(r["edges"]),
    } for r in records])


def build_node_df(G, node_to_cycles, node_to_groups):
    return pd.DataFrame([{
        "node": int(n),
        "cycle_count": len(node_to_cycles.get(n, [])),
        "group_count": len(node_to_groups.get(n, [])),
        "cycles": sorted(set(node_to_cycles.get(n, []))),
        "groups": sorted(set(node_to_groups.get(n, []))),
    } for n in G.nodes()])


def build_edge_df(G, edge_to_cycles, edge_to_groups):
    edges = sorted({tuple(sorted((u, v))) for u, v in G.edges()})
    return pd.DataFrame([{
        "u": u,
        "v": v,
        "cycle_count": len(edge_to_cycles.get((u, v), [])),
        "group_count": len(edge_to_groups.get((u, v), [])),
        "cycles": sorted(set(edge_to_cycles.get((u, v), []))),
        "groups": sorted(set(edge_to_groups.get((u, v), []))),
    } for u, v in edges])


def build_group_df(groups):
    return pd.DataFrame([{
        "group_id": g["group_id"],
        "n_cycles": g["n_cycles"],
        "n_nodes": g["n_nodes"],
        "n_edges": g["n_edges"],
        "total_cycle_length": g["total_cycle_length"],
        "max_cycle_length": g["max_cycle_length"],
        "cycle_ids": g["cycle_ids"],
        "cycle_sizes": g["cycle_sizes"],
        "representative_cycle": g["representative_cycle"],
    } for g in groups])

def process_cycles(G: nx.Graph,
                   min_cycle_edges=4,
                   min_shared_nodes=2,
                   keep_triangles_touching_large=True,
                   triangle_touch_nodes=2):

    cycles = extract_cycle_basis(G)
    records = build_cycle_records(G, cycles)

    filtered = filter_cycles(
        records,
        min_cycle_edges,
        keep_triangles_touching_large,
        triangle_touch_nodes
    )

    sorted_cycles = sort_cycles(filtered)

    groups, cycle_to_group = greedy_group_cycles(
        sorted_cycles,
        min_shared_nodes
    )

    finalize_groups(groups)

    node_to_cycles, node_to_groups, edge_to_cycles, edge_to_groups = \
        build_memberships(records, cycle_to_group)

    return {
        "groups": groups,
        "cycle_df": build_cycle_df(records, cycle_to_group),
        "node_df": build_node_df(G, node_to_cycles, node_to_groups),
        "edge_df": build_edge_df(G, edge_to_cycles, edge_to_groups),
        "group_df": build_group_df(groups),
    }
