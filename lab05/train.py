import random
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt
from engine import Value
import time

print("--- Task 5: Training on make_moons ---")
X, y = make_moons(n_samples=100, noise=0.1, random_state=42)
# Convert labels to -1 and 1 for simple loss
y = [1.0 if val == 1 else -1.0 for val in y]

class Neuron:
    def __init__(self, nin):
        self.w = [Value(random.uniform(-1,1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1,1))
    def __call__(self, x):
        # w * x + b
        act = sum((wi*xi for wi, xi in zip(self.w, x)), self.b)
        return act.relu()

class Layer:
    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]
    def __call__(self, x):
        return [n(x) for n in self.neurons]

class MLP:
    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(nouts))]
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x[0] # Return single output for binary classification

# 2 -> 8 -> 1
model = MLP(2, [8, 1])
epochs = 50

start_time = time.time()
for epoch in range(epochs):
    # Forward
    ypred = [model(x) for x in X]
    
    # BCE Loss approximation (Hinge loss is easier for custom classes)
    # We will use simple squared error for simplicity in this lab
    loss = sum((yout - ygt)**2 for ygt, yout in zip(y, ypred))
    
    # Backward
    for p in model.layers[0].neurons[0].w + model.layers[1].neurons[0].w:
        p.grad = 0.0
    loss.backward()
    
    # Update (SGD)
    lr = 0.1
    for layer in model.layers:
        for neuron in layer.neurons:
            for p in [neuron.b] + neuron.w:
                p.data -= lr * p.grad

end_time = time.time()
print(f"Custom MLP Training Time (50 epochs): {end_time - start_time:.4f} seconds")