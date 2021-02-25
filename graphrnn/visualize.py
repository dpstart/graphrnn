import pickle
import networkx as nx
import matplotlib.pyplot as plt


def draw_graph(G, prefix="test"):

    plt.axis("off")
    pos = nx.spring_layout(G)
    nx.draw_networkx(G, with_labels=True, node_size=35, pos=pos)


with (open("./graphs/fnamepred2100_2.dat", "rb")) as openfile:
    o = pickle.load(openfile)

draw_graph(o[0])
plt.show()
