#https://towardsdatascience.com/bbn-bayesian-belief-networks-how-to-build-them-effectively-in-python-6b7f93435bba

import pandas as pd # for data manipulation 
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
edad = BbnNode(Variable(0, 'edad', ['S', 'N']), [, ])
economia = BbnNode(Variable(1, 'economia', [, ]), [, ])

ahorro = BbnNode(Variable(2, 'ahorro', ['S', 'N']), [0.8, 0.2,
                                                     0.4, 0.6,
                                                     0.5, 0.5,
                                                     0.3, 0.7])

trabajo = BbnNode(Variable(3, 'trabajo', ['S', 'N']), [--, 0.7,
                                                       --, 0.9,
                                                       --, 0.2,
                                                       --, ])

vivienda = BbnNode(Variable(4, 'vivienda', [, ]), [0.7, 0.3,
                                                         0.4, 0.6])


dificultad = BbnNode(Variable(5, 'dificultad', ['S', 'N']), [--, 0.99,
                                                             0.05,-- ,
                                                             --,-- , 
                                                             --,-- ])
paga = BbnNode(Variable(6, 'paga', [--,-- ]), [--,-- ,
                                                --,-- ,
                                                --,--,
                                                --,-- ])

# Se crea la estructura de interrelaciones o grafo bayesiano
bbn = Bbn() \
    .add_node(edad) \
    .add_node(economia) \
    .add_node(ahorro) \
    .add_node(trabajo) \
    .add_node(vivienda) \
    .add_node(dificultad) \
    .add_node(paga) \
    .add_edge(Edge(trabajo, ahorro, EdgeType.DIRECTED)) \
    .add_edge(Edge(edad, ahorro, EdgeType.DIRECTED)) \
    .add_edge(Edge(edad, trabajo, EdgeType.DIRECTED)) \
    .add_edge(Edge(economia, trabajo, EdgeType.DIRECTED)) \
    .add_edge(Edge(economia, vivienda, EdgeType.DIRECTED)) \
    .add_edge(Edge(ahorro, dificultad, EdgeType.DIRECTED)) \
    .add_edge(Edge(trabajo, dificultad, EdgeType.DIRECTED)) \
    .add_edge(Edge(vivienda, paga, EdgeType.DIRECTED)) \
    .add_edge(Edge(dificultad, paga, EdgeType.DIRECTED)) 

# Convert the BBN to a join tree
join_tree = InferenceController.apply(bbn)


# Set node positions
pos = {0: (-1, 2), 1: (1, 2), 2: (-1, 0), 
       3: (0, 0), 4: (1, 0), 5: (-1, -1), 6: (0, -1)}

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
evidence('ev1', 'economia', 'P', 1)
evidence('ev2', 'edad', 'N', 1) 
evidence('ev3', 'trabajo', 'S', 1)
evidence('ev4', 'vivienda', 'P', 1)



print_probs()



evidence('ev1', 'paga', 'N', 1)
evidence('ev2', 'edad', 'S', 1)
evidence('ev3', 'ahorro', 'S', 1)

print_probs()
