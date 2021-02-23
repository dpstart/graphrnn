import pickle
import networkx as nx
import matplotlib.pyplot as plt

with (open("./graphs/fnamepred4_3.dat", "rb")) as openfile:
    o = pickle.load(openfile)

nx.draw(o[0])
plt.draw()
plt.show()