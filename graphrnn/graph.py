import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

def create_graph(args):
    graphs = []
    # synthetic graphs
    if args.graph_type == "grid":
        for i in range(10, 20):
            for j in range(10, 20):
                graphs.append(nx.grid_2d_graph(i, j))
                pos = nx.spring_layout(graphs[0])
                nx.draw_networkx(graphs[0], with_labels=True, node_size=35, pos=pos)
                plt.show()
                import sys;sys.exit(1)
        args.max_prev_node = 40
    elif args.graph_type == "grid_small":
        for i in range(2, 5):
            for j in range(2, 6):
                graphs.append(nx.grid_2d_graph(i, j))
        args.max_prev_node = 15
    return graphs
