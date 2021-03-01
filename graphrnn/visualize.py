import pickle
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np


def draw_graph(G, k=1, node_size=55,alpha=1, width=1.3):

    plt.axis("off")
    pos = nx.spring_layout(G, k=k/np.sqrt(G.number_of_nodes()),iterations=100)
    nx.draw_networkx_nodes(G, node_size=55, pos=pos, alpha=1, linewidths=0)
    nx.draw_networkx_edges(G, pos, width=width, alpha=alpha)


FNAME = "./graphs/out_1500_2.dat"
with (open(FNAME, "rb")) as openfile:
    out = pickle.load(openfile)

draw_graph(out[0])
plt.show()
