import networkx as nx 
import matplotlib.pyplot as plt
import mlp


def visualize(mlp:mlp, y_spacing=1.5, x_spacing=3.0):
    has_bias = mlp.bias
    G = nx.Graph()
    pos = {}
    edge_labels = {}
    node_labels = {}
    bias_nodes = set()
    edge_label_pos = {}  # store individual label_pos per edge

    node_index = 0
    for i, layer in enumerate(mlp.layers):
        weights = layer.weights
        outputs = layer.outputs
        n_in, n_out = weights.shape[0], weights.shape[1]

        src_start = node_index
        dst_start = node_index + n_in

        bias_node = src_start + (n_in - 1)
        if has_bias:
            bias_nodes.add(bias_node)

        for j in range(n_in):
            src = src_start + j
            if src not in pos:
                pos[src] = (i * x_spacing, j * y_spacing)

            if src not in node_labels:
                if has_bias and src == bias_node:
                    node_labels[src] = "bias"
                elif i == 0:
                    node_labels[src] = src
                else:
                    prev_out = mlp.layers[i - 1].outputs.flatten()
                    node_labels[src] = round(float(prev_out[j]), 3)

            for k in range(n_out):
                dst = dst_start + k
                if dst not in pos:
                    pos[dst] = ((i + 1) * x_spacing, k * y_spacing)

                if dst not in node_labels:
                    node_labels[dst] = round(float(outputs.flatten()[k]), 3)

                G.add_edge(src, dst)
                edge_labels[(src, dst)] = round(weights[j][k], 4)

                # Spread label along edge based on source row index
                edge_label_pos[(src, dst)] = 0.2 + (j / max(n_in - 1, 1)) * 0.6  # range 0.2–0.8

        node_index = dst_start

    node_colors = ['red' if n in bias_nodes else 'black' for n in G.nodes()]
    max_nodes = max(layer.weights.shape[0] for layer in mlp.layers)
    plt.figure(figsize=(mlp.layer_num * 4, max_nodes * 2))

    nx.draw(G, pos, labels=node_labels, with_labels=True,
            node_color=node_colors, node_size=1500,
            edge_color='black', font_color='white', font_size=8)

    # Draw each edge label at its own unique position
    for (u, v), label in edge_labels.items():
        nx.draw_networkx_edge_labels(
            G, pos,
            edge_labels={(u, v): label},
            font_size=7,
            label_pos=edge_label_pos[(u, v)],  # unique position per edge
            bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='none', alpha=0.85)
        )

    plt.title("MLP Visualization")
    plt.tight_layout()
    plt.show()