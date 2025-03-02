'''
Notes from "The spelled-out intro to neural networks and backpropagation: building micrograd" of Andrej Karpathy
Link: https://www.youtube.com/watch?v=VMj-3S1tku0
----------

Micrograd is an Autograd engine, where Autograd stands for "Automatic Gradients", which basically implements backpropagation.
Backpropagation allows to evaluate the gradient of some kind of loss function with respect to the weights of a neural network.
Backpropagation is going to start with a certain function "g" and going backwards to the function expression evaluating the
derivative of "g" with respect to all the terms which appears in g: this will tell us how much is "g" influenced by a small
change of one of its terms.
Backpropagation is not related only to neural networks, in fact it is a more general mathematical concept.

The derivative of a function is defined as follows:
                                                      f(x + h) - f(x)
                                        df / dx =  lim  ------------
                                                h -> 0       x
It represents the response of the function with respect to a small change ("h") of its "x" input. To get the slope (namely our
derivative) we need to normalize by "h".

Through this library we are going to build up a "Value" object to store all the information about our neural networks' nodes.
How this "Value" class work?

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

import os
os.environ["PATH"] += os.pathsep + 'C:/Users/danie/Downloads/Graphviz-12.2.1-win64/bin' # it is needed to view Graphviz's graphs
import numpy as np
import matplotlib.pyplot as plt
from micrograd import Value, draw_dot

'''
----------
1) Trying stuffs with derivatives
----------
'''

def f(x):
    return 3*x**2 - 4*x + 5 # this is nothing else than a parabola

xs = np.arange(-5, 5, 0.25)
ys = f(xs)
plt.plot(xs, ys)
# plt.show()

# derivative of "f" with respect to "x": df / dx = (f(x + h) - f(x)) / h
h = 0.0000001
x = 3.0
print((f(x + h) - f(x)) / h) # the smaller "h" the closer to 0 the result of the derivative will be

h = 0.0000001
x = -3.0
print((f(x + h) - f(x)) / h) # at -3, where the parabola is steeper, we expect a higher derivative

h = 0.0000001
x = 2/3
print((f(x + h) - f(x)) / h) # at rougly 2/3 the parabola is flat, thus we expect a derivative close to 0

# consider a more complex function made of three scalar inputs
a = 2.0
b = -3.0
c = 10.0
d = a*b + c
print(d)

# compute the derivative of "d" with respect to "a", "b" and "c"
h = 0.0001

a = 2.0
b = -3.0
c = 10.0
d1 = a*b + c
a += h
d2 = a*b + c
print('d1', d1)
print('d2', d2) # if I increase "a" of a tiny amount, I expect "d2" to decrease since "a" is multplied by -3 ("b")
print('slope', (d2 - d1) / h) # derivative == Slope

a = 2.0
b = -3.0
c = 10.0
d1 = a*b + c
b += h
d2 = a*b + c
print('d1', d1)
print('d2', d2) # if I increase "b" of a tiny amount, I expect "d2" to increase since "a*b" becomes less negative
print('slope', (d2 - d1) / h)

a = 2.0
b = -3.0
c = 10.0
d1 = a*b + c
c += h
d2 = a*b + c
print('d1', d1)
print('d2', d2) # if I increase "a" of a tiny amount, I expect "d2" to increase
print('slope', (d2 - d1) / h)

'''
----------
2) Starting creating Value objects
----------
'''

a = Value(2.0, label = 'a')
b = Value(-3.0, label = 'b')
c = Value(10.0, label = 'c')
e = a * b; e.label = 'e'
d = e + c; d.label = 'd'
f = Value(-2.0, label = 'f')
L = d * f; L.label = 'L'
print(a)
print(b)
print(a + b)
print(a * b)
print(d)
print((a.__mul__(b)).__add__(c)) # it is the same as "a*b + c"
print(d._prev) # to print the children of "d"
print(d._op)
# draw_dot(L).view() # using "graphviz" package to visualize our expressions

'''
----------
3) Running backpropagation to compute all the derivatives of our function
----------
'''

'''
We want to run backpropagation starting from the very last input of our function, namely deriving in this order:
1. dL / dL -> dL / dL = 1 -> This is the base case, namely the case where the gradient is equal to 1.

2. dL / df -> L = d * f -> dL / df = d
dL / df says the impact of "f" on "L".

3. dL / dd -> L = d * f -> dL / dd = ((d + h) * f - d * f) / h = (d * f + h * f - d * f) / h = f
dL / dd says the impact of "d" on "L".

4. dL / de -> dL / de = dL / dd * dd / de (this is the "chain rule") -> d = c + e -> dd / de = ((c + e + h) - (c + e)) / h = 1 -> dL / de = dL / dd * 1
dL / de says the impact of "e" on "L".

5. dL / dc -> dL / dc = dL / dd * dd / dc ("chain rule") = dL / dd * 1
dL / dc says the impact of "c" on "L".

6. dL / da -> dL / da = dL / de * de / da ("chain rule") -> e = a * b -> de / da = b
dL / da says the impact of "a" on "L".

7. dL / db -> dL / da = dL / de * de / db ("chain rule") -> e = a * b -> de / db = a
dL / db says the impact of "b" on "L".
'''

# create a function to manipulate our nodes locally (without any effect on the global scope) and compute the gradient
def lol():
    h = 0.0001
    
    a = Value(2.0, label = 'a')
    b = Value(-3.0, label = 'b')
    c = Value(10.0, label = 'c')
    e = a * b; e.label = 'e'
    d = e + c; d.label = 'd'
    f = Value(-2.0, label = 'f')
    L = d * f; L.label = 'L'
    L1 = L.data # L1 represents f(x)
    
    a = Value(2.0, label = 'a') # a = Value(2.0 + h, label = 'a') to compute a.grad
    b = Value(-3.0, label = 'b') # b = Value(-3.0 + h, label = 'b') to compute b.grad
    c = Value(10.0, label = 'c') # c = Value(10.0 + h, label = 'c') to compute c.grad
    e = a * b; e.label = 'e'; e.data += h # e.data += h to compute e.grad
    d = e + c; d.label = 'd' # d.data += h to compute d.grad
    f = Value(-2.0, label = 'f') # f = Value(-2.0 + h, label = 'f') to compute f.grad
    L = d * f; L.label = 'L'
    L2 = L.data # L2 represents f(x + h)
    
    print('*** Gradient = ', (L2 - L1) / h, ' ***') # print the gradient (f(x + h) - f(x)) / h where h -> 0

lol()

'''
dL / da = (dL / dd) * (dd / de) * (de / da) ("chain rule") = (f) * (1) * (b) = 6.0
dL / db = (dL / dd) * (dd / de) * (de / db) ("chain rule") = (f) * (1) * (a) = -4.0
dL / dc = (dL / dd) * (dd / dc) ("chain rule") = (f) * (1) = -2.0
dL / de = (dL / dd) * (dd / de) ("chain rule") = (f) * (1) = -2.0
dL / dd = f = -2.0
dL / df = d = 4.0
dL / dL = 1
'''

# assign manually gradients computed using the "lol" function
L.grad = 1.0
f.grad = 4.0
d.grad = -2.0
e.grad = -2.0
c.grad = -2.0
b.grad = -4.0
a.grad = 6.0
# draw_dot(L).view()

# it is clear how "a", "b", "c" and "f" are the nodes on which we have control
a.data += 0.01 * a.grad
b.data += 0.01 * b.grad
c.data += 0.01 * c.grad
f.data += 0.01 * f.grad
e = a * b
d = e + c
L = d * f
print(L.data) # "L" will be ((2 + 0.06) * (-3 - 0.04) + (10 - 0.02)) * (-2 + 0.04) = -7.2

'''
----------
4) Implement backpropagation to a neuron
----------
'''

'''
We are now going to backpropagate through a neuron, but what is a neuron?
- In a neuron there are some inputs (x1, x2, x3, ...)
- Then there are the synapses with weights on them (w1, w2, w3, ...). Synapse and input are correlated by a product relationship
- In addition, there is a bias (b), which is a kind of trigger of happiness
- Finally we pass the "sum(xi * wi) + b" as input of an activation function (which could be a sigmoid, a tanh, etc.)

neuron = sum(xi * wi) + b
'''

plt.cla()
plt.plot(np.arange(-5, 5, 0.2), np.tanh(np.arange(-5, 5, 0.2))) # The tanh is an hyperbolic function which goes up until 1 and down until -1. At 0, tanh(0) = 0 
plt.grid()
# plt.show()

# let's build a neuron
x1 = Value(2.0, label = 'x1') # inputs x1, x2
x2 = Value(0.0, label = 'x2')
w1 = Value(-3.0, label = 'w1') # weights w1, w2
w2 = Value(1.0, label = 'w2')
b = Value(6.8813735870195432, label = 'b') # bias of the neuron

# x1*w1 + x2*w2 + b
x1w1 = x1 * w1; x1w1.label = 'x1*w1'
x2w2 = x2 * w2; x2w2.label = 'x2*w2'
x1w1x2w2 = x1w1 + x2w2; x1w1x2w2.label = 'x1*w1 + x2*w2'
n = x1w1x2w2 + b; n.label = 'n' # neuron

# output (neuron fed into tanh function)
o = n.tanh()
o.label = 'o'
# draw_dot(o).view()

'''
Now we want to backpropagate in order to obtain the gradient of our two weights w1 and w2, since in a neural network
they represent the knobs that we are going to adjust to fit output and desired output.

1. do / do = 1
2. do / dn -> o = tanh(n) -> tanh = (e^2x - 1) / (e^2x + 1) -> do / dn = 1 - tanh(x)^2 = 1 - o^2
3. do / db = (do / dn) * (dn / db) ("chain rule") -> do / db = (1 - o^2) * 1
4. do / d(x1w1x2w2) = (do / dn) * (dn / d(x1w1x2w2)) ("chain rule") -> do / db = (1 - o^2) * 1 = 1 - o^2
5. do / d(x1w1) = (do / d(x1w1x2w2)) * (d(x1w1x2w2) / d(x1w1)) = (1 - o^2) * 1 = 1 - o^2
6. do / d(x2w2) = (do / d(x1w1x2w2)) * (d(x1w1x2w2) / d(x2w2)) = (1 - o^2) * 1 = 1 - o^2
7. do / dx1 = (do / d(x1w1)) * (d(x1w1) / dx1) = (1 - o^2) * w1
8. do / dw1 = (do / d(x1w1)) * (d(x1w1) / dw1) = (1 - o^2) * x1
9. do / dx2 = (do / d(x2w2)) * (d(x2w2) / dx2) = (1 - o^2) * w2
10. do / dw2 = (do / d(x2w2)) * (d(x2w2) / dw2) = (1 - o^2) * x2

The weight w2 has no influence on the final output since its gradient is zero. This is logic since it is multiplied by x2,
which is equal to 0.0!
'''

o.grad = 1.0
n.grad = 1.0 - o.data**2
x1w1x2w2.grad = n.grad
b.grad = n.grad
x1w1.grad = n.grad
x2w2.grad = n.grad
x1.grad = x1w1.grad * w1.data
w1.grad = x1w1.grad * x1.data
x2.grad = x2w2.grad * w2.data
w2.grad = x2w2.grad * x2.data
# draw_dot(o).view()

# define another "lol" function to compute manually the gradients
def lol2():
    h = 0.0001
    
    x1 = Value(2.0, label = 'x1')
    x2 = Value(0.0, label = 'x2')
    w1 = Value(-3.0, label = 'w1')
    w2 = Value(1.0, label = 'w2')
    b = Value(6.8813735870195432, label = 'b')
    x1w1 = x1 * w1
    x2w2 = x2 * w2
    x1w1x2w2 = x1w1 + x2w2
    n = x1w1x2w2 + b # neuron
    o1 = n.tanh() # f(x)
    
    x1 = Value(2.0, label = 'x1') # x1 = Value(2.0 + h, label = 'x1')
    x2 = Value(0.0, label = 'x2') # x2 = Value(0.0 + h, label = 'x2')
    w1 = Value(-3.0, label = 'w1') # w1 = Value(-3.0 + h, label = 'w1')
    w2 = Value(1.0, label = 'w2') # w2 = Value(1.0 + h, label = 'w2')
    b = Value(6.8813735870195432 + h, label = 'b') # b = Value(6.8813735870195432 + h, label = 'b')
    x1w1 = x1 * w1 # x1w1.data += h
    x2w2 = x2 * w2 # x2w2.data += h
    x1w1x2w2 = x1w1 + x2w2 # x1w1x2w2.data += h
    n = x1w1x2w2 + b # n.data += h
    o2 = n.tanh() # f(x + h)
    
    print('*** Gradient = ', (o2.data - o1.data) / h, ' ***')

lol2()

# since we have implemented the "_backward()" function, we don't need anymore to compute manually the gradients
x1 = Value(2.0, label = 'x1') # inputs x1, x2
x2 = Value(0.0, label = 'x2')
w1 = Value(-3.0, label = 'w1') # weights w1, w2
w2 = Value(1.0, label = 'w2')
b = Value(6.8813735870195432, label = 'b') # bias of the neuron
x1w1 = x1 * w1; x1w1.label = 'x1*w1'
x2w2 = x2 * w2; x2w2.label = 'x2*w2'
x1w1x2w2 = x1w1 + x2w2; x1w1x2w2.label = 'x1*w1 + x2*w2'
n = x1w1x2w2 + b ; n.label = 'n' # x1*w1 + x2*w2 + b
o = n.tanh(); o.label = 'o' # output (neuron fed into tanh function)
# draw_dot(o).view() # we have all the gradients equal to zero here

# since "x1", "w1", "x2" and "w2" doesn't perform any operation, the ._backward() is not needed for them
o.grad = 1.0 # "o.grad", which represents "out.grad" while we call "o._backward()", is initialized as 0.0, thus we need to assign it to 1.0 manually (that's the base case)
o._backward() # propagate through the tanh(n)
n._backward() # propagate through the sum ((x1*w1 + x2*w2) + b)
b._backward() # propagate through b
x1w1x2w2._backward() # propagate through the sum (x1*w1 + x2*w2)
x1w1._backward() # propagate through the product (x1*w1)
x2w2._backward() # propagate through the product (x2*w2)
# draw_dot(o).view() # here there are all the gradients

'''
----------
5) Implement topological sort (a topological sort is a graph traversal in which each node "v" is visited only after
all its dependencies (namely children) are visited, see: https://en.wikipedia.org/wiki/Topological_sorting)
----------
'''

topo = []
visited = set()
def build_topo(v):
    if v not in visited:
        visited.add(v) # mark the node as visited
        for child in v._prev:
            build_topo(child)
        topo.append(v) # append child node only when it has no other children

x1 = Value(2.0, label = 'x1') # inputs x1, x2
x2 = Value(0.0, label = 'x2')
w1 = Value(-3.0, label = 'w1') # weights w1, w2
w2 = Value(1.0, label = 'w2')
b = Value(6.8813735870195432, label = 'b') # bias of the neuron
x1w1 = x1 * w1; x1w1.label = 'x1*w1'
x2w2 = x2 * w2; x2w2.label = 'x2*w2'
x1w1x2w2 = x1w1 + x2w2; x1w1x2w2.label = 'x1*w1 + x2*w2'
n = x1w1x2w2 + b; n.label = 'n' # x1*w1 + x2*w2 + b
o = n.tanh(); o.label = 'o' # output (neuron fed into tanh function)
# draw_dot(o).view() # all the gradients are zero

o.grad = 1.0 # base case
topo = []
visited = set() # empty set to store visited nodes
build_topo(o)
print(topo) # "topo" list will contain our output as very last value since I have first added all the children

for node in reversed(topo): # need to reverse "topo" to apply correctly the "chain rule"
    node._backward()

# draw_dot(o).view()

'''
----------
6) Compute gradients using the "backward" function
----------
'''

topo = []
visited = set() # empty set to store visited nodes

x1 = Value(2.0, label = 'x1') # inputs x1, x2
x2 = Value(0.0, label = 'x2')
w1 = Value(-3.0, label = 'w1') # weights w1, w2
w2 = Value(1.0, label = 'w2')
b = Value(6.8813735870195432, label = 'b') # bias of the neuron
x1w1 = x1 * w1; x1w1.label = 'x1*w1'
x2w2 = x2 * w2; x2w2.label = 'x2*w2'
x1w1x2w2 = x1w1 + x2w2; x1w1x2w2.label = 'x1*w1 + x2*w2'
n = x1w1x2w2 + b; n.label = 'n' # x1*w1 + x2*w2 + b
o = n.tanh(); o.label = 'o' # output (neuron fed into tanh function)
draw_dot(o).view() # all the gradients are zero

o.backward()
draw_dot(o).view() # all the gradients are computed

'''
----------
7) Bug fix when a node is used multiple times  (if a node is used multiple times, I overwrite its gradient
since, in the case of a sum for example, "self" and "other" are the same object, and thus I'm writing on the
same object's gradient two times)
----------
'''

a = Value(3.0, label = 'a')
b = a + a # self = a; other = a; b = a.__sum__(a)
b.label = 'b'
b.backward() # db / da should be (da / da + da / da) = 2 and not 1
draw_dot(b).view()

# check this bug on a more complex expression
a = Value(-2.0, label = 'a')
b = Value(3.0, label = 'b')
d = a * b; d.label = 'd'
e = a + b; e.label = 'e'
f = d * e; f.label = 'f'
f.backward() # the gradient of "a" and "b" is wrong since they have been used more than once (thus we need to accumulate this gradient)
draw_dot(f).view()

'''
----------
8) Break up "tanh" expression into its various components
----------
'''