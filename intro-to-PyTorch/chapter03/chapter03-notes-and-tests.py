"""
Notes from chapter 03 of "Deep Learning with PyTorch" by Eli Stevens, Luca Antiga and Thomas Viehmann
Link: None
----------

Chapter 3: "It starts with a tensor"

In the context of deep learning, "tensors" refer to the generalization of vectors and matrices to an arbitrary number of dimensions.
Another name for the same concept is "multidimensional array". The dimensionality of a tensor coincides with the number of indexes
used to refer to scalar values within the tensor. Thus, a tensor can be a scalar (0D), a vector (1D), a matrix (2D), or a 3D/.../nD
array.

PyTorch is not the only library that deals with multidimensional arrays. NumPy is by far the most popular multidimensional array
library, to the point that it has now arguably become the lingua franca of data science. Compared to NumPy arrays, PyTorch tensors
have a few superpowers, such as the ability to perform very fast operations on graphical processing units (GPUs), distribute operations
on multiple devices or machines, and keep track of the graph of computations that created them.

Python's list indexing VS PyTorch tensor indexing:
    1. We can access the first element of a list using the corresponding zero-based index. The same happens with tensors, also if under
    the hood lists and tensors are completely different.
    One big difference between lists and tensors is that when we access one or more elements in a tensor, the output is a tensor again
    (in other words, it is a different view of the same initial data), but this doesn't mean that a new tensor has been created, since
    this would be very inefficient. Thus, there isn't a new chunk of memory allocated when accessing elements in a tensor!

    2. The range indexing notation too works in the same manner between lists and tensors. Note that with list we use the [i][j]
    indexing, while with tensor the [i, j, k, h, ...] one, where i, j, k, h are the various tensor dimensions (axes).

    3. Tensors can be named. "Named tensors" allow users to give explicit names to tensor dimensions. In most cases, operations that
    take dimension parameters will accept dimension names, avoiding the need to track dimensions by position. In addition, named tensors
    use names to automatically check that APIs are being used correctly at runtime, providing extra safety. Names can also be used to
    rearrange dimensions, for example, to support “broadcasting by name” rather than “broadcasting by position”.
    The "refine_names" method can be used when we already have a tensor and want to add names to its dimensions.
    
    What is broadcasting in Python?
    NumPy broadcasting: "broadcasting" refers to how NumPy treats arrays with different shapes during arithmetic operations. When
    operating on two arrays, NumPy compares their shapes element-wise. It starts with the trailing (i.e. rightmost) dimension and
    works its way left. Two dimensions are compatible when:
        - they are equal
        - or one of them is 1.
    If these conditions are not met, a "ValueError: operands could not be broadcast together" exception is thrown, indicating that
    the arrays have incompatible shapes.
    Link: https://www.youtube.com/watch?v=oG1t3qlzq14
"""

import torch
import numpy as np

a = [1.0, 2.0, 3.0]
print("---")
print(a[0])
print(a[1])
print(a[2])

a = torch.ones(3) # one-dimensional tensor of size 3 filled with ones
print("---")
print(a)
print(a[0])
print(a[1])
print(a[2])
a[2] = 3.0
print(a)

points = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]) # 2D tensor of size 3 (rows) x 2 (cols) - Note that we are passing a list
print("---")
print(points)
print(points[2]) # it should return [5.0, 6.0]
print(points[1][1], '-', type(points[1][1])) # it should return 4.0 (like in a Python's list) - Note that all these outputs are tensors!
print(points.shape, '-', type(points.shape)) # return the size of the tensor along each dimension

new_list = list(range(6)) # [0, 1, 2, 3, 4, 5]
print("---")
print(new_list)
print(new_list[1:]) # [1, 2, 3, 4, 5] - Note that the left index is inclusive, while the right one is exclusive!
print(new_list[:2]) # [0, 1]
print(new_list[1:5:2]) # [1, 3]
print(new_list[:]) # print the entire list
print(new_list[:-1]) # print from the start of the list to one before last element
print(new_list[:-2]) # print from the start of the list to two before last element

print("---")
print(points)
print(points[1:]) # print all rows of the tensor after the first
print(points[1:, 0]) # print the first column of all rows after the first
print(points[0:, 0]) # print the first column of all rows
print(points[2, 0:]) # print all the columns of the third row

img = torch.randn(3, 5, 7) # create a 3D tensor filled with random numbers
weights = torch.tensor([0.2126, 0.7152, 0.0722], names = ['channels']) # create a named tensor
print("---")
print(img) # default tensor dimensions' names are None
print(weights)
img_named = img.refine_names(..., 'x', 'y') # give a name only to the two last dimensions of the tensor
# print(img) # img has not been changed
print(img_named) # img_named = img + names

img_named = img.refine_names('channels', 'rows', 'columns') # if we call refine_names method on img_named we throw an error
print("---")
print(img_named)
weights_named = weights.align_as(img_named)
print(weights_named)

# trying stuffs with NumPy broadcasting
a = np.array([1, 2, 3])
b = np.array([[0, 0, 0], [1, 1, 1]])
c = a + b
d = np.array([[0, 1], [2, 3], [4, 5]])
e = np.array([[1], [2], [3]])
f = d + e
# e = a + d # a and d they cannot be broadcasted together since they have not compatible shapes
print("---")
print(f'"a" shape is --> {a.shape}') # (3,) == (1, 3)
print(f'"b" shape is --> {b.shape}')
print(f'"c" shape is --> {c.shape}')
print(f'"d" shape is --> {d.shape}')
print(f'"e" shape is --> {e.shape}')
print(f'"f" shape is --> {f.shape}')
print(a)
print(b)
print(c)
print(d)
print(e)
print(f)

print("---")
print(f'"img_named" shape is --> {img_named.shape}')
print(f'"weights" shape is --> {weights.shape}')
print(f'"weights_named" shape is --> {weights_named.shape}')
product = img_named * weights_named
print(product)
print(f'"product" shape is --> {product.shape}')
print(product.names)
sum1 = product.sum('channels') # sum over the channels dimension (the output tensor will contain only the rows and columns dimensions)
print(sum1)
print(sum1.shape)
sum2 = product.sum('columns') # sum over the columns dimension --> output tensor shape = [3, 5]
print(sum2)
print(sum2.shape)
sum3 = product.sum('rows') # sum over the rows dimension --> output tensor shape = [3, 7]
print(sum3)
print(sum3.shape)

tensor1 = torch.randn(2, 4, 6, 8, 10)
print("---")
print(tensor1.shape) # [2, 4, 6, 8, 10]
print(tensor1.names)
tensor1 = tensor1.refine_names('A', ..., 'D', 'E') # it is possible to use just one ellipsis ("...")
print(tensor1.names)
tensor1 = tensor1.rename(A = 'aaa', E = 'eee') # rename method is useful to rename already existing dimensions' names
print(tensor1.names)
tensor1 = tensor1.rename('A', 'B', 'C', 'D', 'E')
print(tensor1.names)
tensor1 = tensor1.rename(None) # reset dimensions' names
print(tensor1.names)
tensor1 = tensor1.rename('A', ..., 'E') # tensor1.rename('A', ..., 'E') == tensor1.refine_names('A', ..., 'E')
print(tensor1.names)
tensor1 = tensor1.rename(None)
print(tensor1.names)
tensor1 = tensor1.refine_names('A', ..., 'E') # some dimensions are named while some others are not
print(tensor1.names)
