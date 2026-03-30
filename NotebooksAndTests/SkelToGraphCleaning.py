import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse.csgraph import dijkstra, connected_components
from scipy.sparse import csr_matrix
from scipy.interpolate import splprep, splev

# ------------------------------------------------------------
# 1. Load skeleton points (replace with your actual data)
points = np.array(np.where(skel)).T   # shape (N, 3)
print(f"Total points: {len(points)}")

# ------------------------------------------------------------
# 2. Build nearest‑neighbor graph (k=6 is a good starting point)
k_neighbors = 6
tree = cKDTree(points)
distances, indices = tree.query(points, k=k_neighbors)

n = len(points)
adj_matrix = np.zeros((n, n))
for i in range(n):
    for j in indices[i][1:]:   # skip self
        d = distances[i][np.where(indices[i] == j)[0][0]]
        adj_matrix[i, j] = d
graph = csr_matrix(adj_matrix)

# ------------------------------------------------------------
# 3. Find connected components
n_components, labels = connected_components(graph, directed=False)
print(f"Number of connected components: {n_components}")

# ------------------------------------------------------------
# 4. Prepare Plotly figure
fig = go.Figure()

# We'll assign a different color to each component (optional)
colors = ['blue', 'red', 'green', 'orange', 'purple', 'cyan', 'magenta', 'yellow']

# ------------------------------------------------------------
# 5. Process each component separately
for comp_idx in range(n_components):
    mask = labels == comp_idx
    comp_points = points[mask]
    comp_graph = graph[mask][:, mask]
    comp_color = colors[comp_idx % len(colors)]

    # Build adjacency list for this component
    adjacency = {}
    for i in range(len(comp_points)):
        adj = comp_graph[i].indices  # neighbors (sparse format)
        adjacency[i] = set(adj)

    # Identify endpoints (degree 1) and branch points (degree > 2)
    degrees = {i: len(adjacency[i]) for i in range(len(comp_points))}
    endpoints = [i for i, deg in degrees.items() if deg == 1]
    branch_points = [i for i, deg in degrees.items() if deg > 2]
    nodes_of_interest = set(endpoints + branch_points)

    print(f"Component {comp_idx}: {len(comp_points)} points, "
          f"{len(endpoints)} endpoints, {len(branch_points)} branch points")

    # If component is just a simple path (no branch points)
    if len(branch_points) == 0 and len(endpoints) >= 2:
        # Get the path between the two farthest endpoints
        # (using Dijkstra from the first endpoint)
        dist_from_start, pred = dijkstra(comp_graph, indices=endpoints[0],
                                         return_predecessors=True)
        far_end = np.argmax(dist_from_start)
        # Reconstruct path
        path = []
        curr = far_end
        while curr != -9999 and curr != endpoints[0]:
            path.append(curr)
            curr = pred[curr]
        path.append(endpoints[0])
        path.reverse()
        path_points = comp_points[path]
        # Fit spline if enough points
        if len(path_points) >= 4:
            tck, u = splprep([path_points[:,0], path_points[:,1], path_points[:,2]],
                             s=0, per=False)
            u_new = np.linspace(0, 1, 200)
            x, y, z = splev(u_new, tck)
            fig.add_trace(go.Scatter3d(
                x=x, y=y, z=z,
                mode='lines',
                line=dict(color=comp_color, width=4),
                name=f'Component {comp_idx} branch'
            ))
        else:
            # Fallback: plot straight line segments
            fig.add_trace(go.Scatter3d(
                x=path_points[:,0], y=path_points[:,1], z=path_points[:,2],
                mode='lines',
                line=dict(color=comp_color, width=3),
                name=f'Component {comp_idx} path'
            ))
        continue

    # ------------------------------------------------------------
    # 6. For components with branch points, extract all edges
    used_edges = set()
    for start in nodes_of_interest:
        for target in nodes_of_interest:
            if start >= target:
                continue
            # Shortest path between start and target
            dist, pred = dijkstra(comp_graph, indices=start, return_predecessors=True)
            if dist[target] == np.inf:
                continue  # should not happen in same component
            # Reconstruct path
            path = []
            curr = target
            while curr != -9999 and curr != start:
                path.append(curr)
                curr = pred[curr]
            path.append(start)
            path.reverse()
            # Check if the path contains any other node_of_interest in between
            intermediate = set(path[1:-1])
            if intermediate & nodes_of_interest:
                continue   # this is not a pure edge
            # Avoid duplicate edges
            edge_key = tuple(sorted((start, target)))
            if edge_key in used_edges:
                continue
            used_edges.add(edge_key)
            path_points = comp_points[path]
            # Fit spline
            if len(path_points) >= 4:
                tck, u = splprep([path_points[:,0], path_points[:,1], path_points[:,2]],
                                 s=0, per=False)
                u_new = np.linspace(0, 1, 200)
                x, y, z = splev(u_new, tck)
                fig.add_trace(go.Scatter3d(
                    x=x, y=y, z=z,
                    mode='lines',
                    line=dict(color=comp_color, width=3),
                    name=f'Comp{comp_idx} branch'
                ))
            else:
                # Fallback: plot straight line segments
                fig.add_trace(go.Scatter3d(
                    x=path_points[:,0], y=path_points[:,1], z=path_points[:,2],
                    mode='lines',
                    line=dict(color=comp_color, width=2),
                    name=f'Comp{comp_idx} edge'
                ))

# ------------------------------------------------------------
# 7. (Optional) Also plot the original points for reference
fig.add_trace(go.Scatter3d(
    x=points[:,0], y=points[:,1], z=points[:,2],
    mode='markers',
    marker=dict(size=2, color='lightgray', opacity=0.5),
    name='Skeleton points'
))

# ------------------------------------------------------------
# 8. Final layout
fig.update_layout(
    title='All branches of the coronary tree (smooth splines)',
    scene=dict(
        xaxis_title='X',
        yaxis_title='Y',
        zaxis_title='Z',
        aspectmode='data'   # or 'auto' to keep proportions
    ),
    width=900,
    height=700,
    showlegend=False  # set to True if you want to see individual branch labels
)

fig.show()

rows, cols = graph.nonzero()
edges = np.column_stack((rows, cols))
# Remove duplicate (i,j) with i>j if you want undirected
edges = edges[edges[:,0] < edges[:,1]]


import pandas as pd
# Convert numpy array to float32
points_f32 = points.astype(np.float32)

# Then create DataFrame
df_points = pd.DataFrame(points_f32, columns=['x', 'y', 'z'])
df_points.to_csv('points_float32.csv', index=False)
df_points.to_csv('points_float32.csv.gz', index=False)
df_edges = pd.DataFrame(edges, columns=['node1', 'node2'])
df_edges.to_csv('graph_edges.csv', index=False)
df_edges.to_csv('graph_edges.csv.gz', index=False)

fig = go.Figure()

# Add the points as a scatter trace
fig.add_trace(go.Scatter3d(
    x=points[:,0],
    y=points[:,1],
    z=points[:,2],
    mode='markers',
    marker=dict(size=2, color='lightblue', opacity=0.7),
    name='Skeleton points'
))

# Add each edge as a line
# To avoid a huge number of traces, we can add all edges in one go using line segments
# but Plotly's Scatter3d does not support multiple disconnected lines directly.
# A common approach is to create a single trace with lines connecting each pair,
# but that would create unwanted connections. So we'll add each edge as its own trace.
# With ~2000 edges, this is fine.
for i, (u, v) in enumerate(edges):
    fig.add_trace(go.Scatter3d(
        x=[points[u,0], points[v,0]],
        y=[points[u,1], points[v,1]],
        z=[points[u,2], points[v,2]],
        mode='lines',
        line=dict(color='red', width=1),
        showlegend=False,
        hoverinfo='none'
    ))

# Improve layout
fig.update_layout(
    title='Coronary tree skeleton (points + edges)',
    scene=dict(
        xaxis_title='X',
        yaxis_title='Y',
        zaxis_title='Z',
        aspectmode='data'
    ),
    width=900,
    height=700
)

fig.show()