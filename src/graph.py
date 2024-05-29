import io

from graphviz import Digraph
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from .value import Value

class Graph:
    def __init__(self, root: Value) -> None:
        self.dot = Digraph(format='png', graph_attr={ 'rankdir': 'LR' })

        self.nodes = set()
        self.edges = set()

        self.build(root)
        self.draw()

    def build(self, x: Value):
        self.nodes.add(x)
        for child in x._prev:
            # Add a line from child to x
            self.edges.add((child, x))

            # Recursively build for all children
            self.build(child)
    
    def draw(self):
        for node in self.nodes:
            # UID of the node
            uid = str(id(node))

            # Create a node in the graph
            self.dot.node(
                name=uid,
                label=f"{node.label} | {node.data} | grad: {node.grad}",
                shape='record'
            )

            if node._op:
                # Create a node for the operation
                self.dot.node(name=uid + node._op, label=node._op)

                # Add a line from the operation node to the current node
                self.dot.edge(uid + node._op, uid)

        for node1, node2 in self.edges:
            # Add a line from node1 to node2's operation node
            self.dot.edge(str(id(node1)), str(id(node2)) + node2._op)

    def show(self):
        fp = io.BytesIO(self.dot.pipe(format='jpeg'))
        with fp:
            img = mpimg.imread(fp, format='jpeg')
        plt.imshow(img)
        plt.show()
