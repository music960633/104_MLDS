import theano
import theano.tensor as T
import numpy
import random
from itertools import izip 

x = T.vector()
w = theano.shared(numpy.array([1.0, -1.0]))
b = theano.shared(0.0)

z = T.dot(w, x) + b
y = 1 / (1 + T.exp(-z)) # activation function

# neuron function
neuron = theano.function(inputs = [x], outputs = y)

y_hat = T.scalar()
cost = T.sum((y - y_hat) ** 2)
dw, db = T.grad(cost, [w, b])

# update function
def updateFunc(param, grad):
  mu = 0.3
  param_updates = [(p, p - mu * g) for p, g in izip(param, grad)]
  return param_updates

# grdient function
gradient = theano.function(
    inputs = [x, y_hat],
    updates = updateFunc([w, b], [dw, db])
)

x = [1, -1]
y_hat = 1

for i in range(100):
  print neuron(x)
  gradient(x, y_hat)
  print w.get_value(), b.get_value()
