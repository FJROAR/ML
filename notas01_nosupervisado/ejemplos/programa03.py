#https://towardsdatascience.com/bbn-bayesian-belief-networks-how-to-build-them-effectively-in-python-6b7f93435bba

import networkx as nx # for drawing graphs
import matplotlib.pyplot as plt # for drawing graphs

# for creating Bayesian Belief Networks (BBN)
from pybbn.graph.dag import Bbn
from pybbn.graph.edge import Edge, EdgeType
from pybbn.graph.jointree import EvidenceBuilder
from pybbn.graph.node import BbnNode
from pybbn.graph.variable import Variable
from pybbn.pptc.inferencecontroller import InferenceController

# Se introducen manualmente las tablas de probabilidad y condicionadas
asia = BbnNode(Variable(0, 'asia', ['Si', 'No']), [0.01, 0.99])
smoke = BbnNode(Variable(1, 'smoke', ['Si', 'No']), [0.5, 0.5])
tub = BbnNode(Variable(2, 'tub', ['Si', 'No']), [0.05, 0.95,
                                                 0.01, 0.99])
lung = BbnNode(Variable(3, 'lung', ['Si', 'No']), [0.1, 0.9,
                                                   0.01, 0.99])
bronc = BbnNode(Variable(4, 'bronc', ['Si', 'No']), [0.6, 0.4,
                                                     0.3, 0.7])
either = BbnNode(Variable(5, 'either', ['Si', 'No']), [1, 0,
                                                        1, 0, 
                                                        1, 0, 
                                                        0, 1])
xray = BbnNode(Variable(6, 'xray', ['Si', 'No']), [0.98, 0.02,
                                                   0.5, 0.95])
dysp = BbnNode(Variable(7, 'dysp', ['Si', 'No']), [0.9, 0.1,
                                                    0.7, 0.3,
                                                    0.8, 0.2,
                                                    0.1, 0.9])

# Se crea la estructura de interrelaciones o grafo bayesiano
bbn = Bbn() \
    .add_node(asia) \
    .add_node(smoke) \
    .add_node(tub) \
    .add_node(lung) \
    .add_node(bronc) \
    .add_node(either) \
    .add_node(xray) \
    .add_node(dysp) \
    .add_edge(Edge(asia, tub, EdgeType.DIRECTED)) \
    .add_edge(Edge(smoke, lung, EdgeType.DIRECTED)) \
    .add_edge(Edge(tub, either, EdgeType.DIRECTED)) \
    .add_edge(Edge(lung, either, EdgeType.DIRECTED)) \
    .add_edge(Edge(smoke, bronc, EdgeType.DIRECTED)) \
    .add_edge(Edge(either, xray, EdgeType.DIRECTED)) \
    .add_edge(Edge(either, dysp, EdgeType.DIRECTED)) \
    .add_edge(Edge(bronc, dysp, EdgeType.DIRECTED)) 

# Convert the BBN to a join tree
join_tree = InferenceController.apply(bbn)


# Set node positions
pos = {0: (-1, 2), 1: (1, 2), 2: (-1, 1), 3: (0, 1), 
       4: (1, 0), 5: (0, 0), 6: (0, -1), 7: (1, -1)}

# Set options for graph looks
options = {
    "font_size": 16,
    "node_size": 4000,
    "node_color": "white",
    "edgecolors": "black",
    "edge_color": "red",
    "linewidths": 5,
    "width": 5,}
    
# Generate graph
n, d = bbn.to_nx_graph()
nx.draw(n, with_labels=True, labels=d, pos=pos, **options)

# Update margins and print the graph
ax = plt.gca()
ax.margins(0.10)
plt.axis("off")
plt.show()


#Probabilidades marginales
def print_probs():
    for node in join_tree.get_bbn_nodes():
        potential = join_tree.get_bbn_potential(node)
        print("Node:", node)
        print("Values:")
        print(potential)
        print('----------------')
        
# Se aplica la función
print_probs()


#Introducción de evidencias

def evidence(ev, nod, cat, val):
    ev = EvidenceBuilder() \
    .with_node(join_tree.get_bbn_node_by_name(nod)) \
    .with_evidence(cat, val) \
    .build()
    join_tree.set_observation(ev)
    
    
# Use above function to add evidence
evidence('ev1', 'smoke', 'Si', 1.0)
evidence('ev2', 'dysp', 'Si', 1.0)

print_probs()