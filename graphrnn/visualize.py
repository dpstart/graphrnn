import pickle
import networkx as nx
import matplotlib.pyplot as plt


def draw_graph(G, prefix="test"):

    plt.axis("off")
    pos = nx.spring_layout(G)
    nx.draw_networkx(G, with_labels=True, node_size=35, pos=pos)


FNAME = "./graphs/fnamepred1000_3.dat"
with (open(FNAME, "rb")) as openfile:
    out = pickle.load(openfile)

draw_graph(out[0])
plt.show()
