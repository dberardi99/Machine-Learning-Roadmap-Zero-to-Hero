import math
from graphviz import Digraph

class Value():
    '''
    A Value object encapsulates all the information of the node.
    
    Parameters
    ----------
    data : int
        An int that stores the node value.
    
    _children : tuple, default=()
        A tuple needed to understand how each Value object is linked to the previous one, namley how it is linked to its
        children. We are going to create a children every time we add or multiply two Value objects.
        
    _op : str, default=''
        A string needed to know which operation has created the particular Value object.
        
    label : str, default=''
        A string that stores the node name.
        
    Other Parameters
    ----------
    grad : float, default=0.0
        A variable to store the gradient of our function with respect to the considered node. A gradient of zero means no
        effect of the input on the loss function.
    
    _backward : function, default=None
        A function to apply backpropagation to our Value object. This function by default doesn't do anything.
        
    _prev : set
        A set to store the Value object's children.
    '''
    def __init__(self, data, _children = (), _op = '', label = ''):                
        self.data = data
        self.grad = 0.0 # a gradient of zero means no effect of the input on the loss function
        self._backward = lambda: None # it is a function that by default doesn't do anything
        self._prev = set(_children)
        self._op = _op
        self.label = label
    
    def __repr__(self): # to print the Value object value
        return f"Value(data={self.data})"
    
    def __add__(self, other):
        '''
        A function to add each other two Value objects (the children will be the two variables that we are adding, namely
        "self" and "other").
        '''
        out = Value(self.data + other.data, (self, other), '+') # The ouput will be a Value object again
        
        def _backward(): # we want this function to propagate the gradient into "self.grad" and "other.grad"
            self.grad += 1.0 * out.grad # the local derivative of an addition operation is equal to 1.0, while "out.grad" represents the application of the "chain rule"
            other.grad += 1.0 * out.grad # "+=" to fix the bug of summing the same node (our gradient is initialized at 0.0, so this works well)
        out._backward = _backward # assign to the output node the relative operation
        
        return out
    
    def __mul__(self, other):
        '''
        A function to multiply each other two Value objects.
        '''
        out = Value(self.data * other.data, (self, other), '*') # the ouput will be a Value object again
        
        def _backward():
            self.grad += other.data * out.grad # the local derivative of the product operation is the other term of the product
            other.grad += self.data * out.grad
        out._backward = _backward
        
        return out
    
    def tanh(self):
        '''
        A function to implement the hyperbolic function (namely the "tanh" function).
        '''
        n = self.data
        t = (math.exp(2*n) - 1) / (math.exp(2*n) + 1) # tanh
        out = Value(t, (self, ), 'tanh')
        
        def _backward():
            self.grad += (1 - t**2) * out.grad # the local derivative of the "tanh" operation is (1 - tanh(x)^2)
        out._backward = _backward
        
        return out
    
    def backward(self):
        '''
        A function to implement backpropagation through the node.
        '''
        topo = [] # "topo" is a local variable
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v) # mark the node as visited
                for child in v._prev:
                    build_topo(child)
                topo.append(v) # append child node only when it has no other children
        build_topo(self)
        
        self.grad = 1.0 # base case (the gradient of a function with respect to itself is always 1)
        for node in reversed(topo): # compute top down gradients ("chain rule")
            node._backward() # here I compute the gradient of the node children
            '''
            When my node has children (namely it has been created from some operation), if I call
            node._backward() is the same as calling a.__add__(b) in the case of a sum for example
            or a.tanh() in the case of a tanh operation (where "a" and "b" are two children).
            Thus, when I call "node._backward()" I'm computing the gradients of the node's children.
            ''' 

# we are going to use the "graphviz" package to visualize our expressions, but we need to define two custom functions
def trace(root):
    # build a set of all nodes and edges in a graph
    nodes, edges = set(), set()
    def build(v):
        if v not in nodes: # check if Value object is in nodes
            nodes.add(v) # add the node at top of set
            for child in v._prev: # iterate thorugh children of Value
                edges.add((child, v)) # add both child and relative Value as tuple to edges
                build(child) # recursion over each child to find its children
    build(root)
    return nodes, edges

def draw_dot(root):
    dot = Digraph(format = 'svg', graph_attr = {'rankdir': 'LR'}) # LR = left to right
    
    nodes, edges = trace(root) # here we have built all our nodes and edges (links between nodes)
    for n in nodes:
        uid = str(id(n)) # assign id to each node of the graph
        # for any value in the graph, create a rectangular ('record') node for it
        dot.node(name = uid, label = "{ %s | data %.4f | grad %.4f }" % (n.label, n.data, n.grad), shape = 'record')
        if n._op:
            # if this value is a result of some operation, create an op node for it
            dot.node(name = uid + n._op, label = n._op) # create a "fake" node to make visible the operation between nodes 
            # and connect this node to it
            dot.edge(uid + n._op, uid)
    
    for n1, n2 in edges:
        # connect n1 to the op node of n2
        dot.edge(str(id(n1)), str(id(n2)) + n2._op)
    
    return dot