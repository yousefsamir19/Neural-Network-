import networkx as nx 
import matplotlib.pyplot as plt
import mlp
import preprocessing as pre

var = pre.preprocessing()
df = var.get_preprocessed_df()
X_train, y_train, X_test, y_test,sc = var.split(df)

model = mlp.mlp(X_train,y_train,2,2,0.1,1,0,0)
# sum =0
# for i in range(len(X_train)):
#     sum+=1
#     model.forward_pass(X_train.iloc[i,:])
# print("the sum = ",sum) 

def visualize(mlp, has_bias=True):
    G = nx.Graph()
    pos = {}
    edge_labels = {}
    bias_nodes = set()

    node_index = 0
    for i, layer in enumerate(mlp.layers):
        weights = layer.weights
        n_in, n_out = weights.shape[0], weights.shape[1]

        src_start = node_index
        dst_start = node_index + n_in

        # Mark first node of each source layer as bias
        if has_bias:
            bias_nodes.add(src_start+(n_in-1))

        for j in range(n_in):
            src = src_start + j
            if src not in pos:
                pos[src] = (i, j)

            for k in range(n_out):
                dst = dst_start + k
                if dst not in pos:
                    pos[dst] = (i + 1, k)

                G.add_edge(src, dst)
                edge_labels[(src, dst)] = round(weights[j][k], 4)

        node_index = dst_start

    # Build color list in node order
    node_colors = ['red' if n in bias_nodes else 'black' for n in G.nodes()]
    print(G.nodes())
    nx.draw(G, pos, with_labels=False, node_color=node_colors,
            node_size=1000, edge_color='black')
    nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=8)
    plt.title("MLP Visualization")
    plt.show()

visualize(model,has_bias=model.bias)            # bias nodes shown in red
