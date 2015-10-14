import theano
import theano.tensor as T
import random

x = T.vector()
w = T.vector()
b = T.scalar()

z = T.dot(w, x) + b
y = 1 / (1 + T.exp(-z)) # activation function

neuron = theano.function(inputs = [x, w, b], outputs = [y])
w = [-1, 1]
b = 0

for i in range(100):
   x = [random.random(), random.random()]
   print x
   print neuron(x, w, b)
