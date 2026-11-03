import math

class Value:
    """
    Task 1: Design a Value node that stores data, gradient, parent nodes, and local derivative rule.
    Task 2: Implement addition, multiplication, powers, and ReLU.
    Task 3: Implement backward pass with topological sort and accumulation.
    """
    def __init__(self, data, _children=(), _op=''):
        self.data = data
        self.grad = 0.0  # Stores the gradient
        self._backward = lambda: None  # Local derivative rule
        self._prev = set(_children)    # Parent nodes
        self._op = _op                 # Operation (for debugging)

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')
        
        def _backward():
            # Local derivative of addition is 1. Accumulate (+=) do not overwrite (=)
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')
        
        def _backward():
            # Local derivative of multiplication: d(a*b)/da = b, d(a*b)/db = a
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        out = Value(self.data ** other, (self,), f'**{other}')
        
        def _backward():
            # Power rule: d(x^n)/dx = n * x^(n-1)
            self.grad += (other * (self.data ** (other - 1))) * out.grad
        out._backward = _backward
        return out

    def relu(self):
        out = Value(0 if self.data < 0 else self.data, (self,), 'ReLU')
        
        def _backward():
            # If data was positive, gradient passes through. If negative, it is 0.
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward
        return out

    def backward(self):
        # Task 3: Topological sort
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        # Set output gradient to 1 and visit in reverse order
        self.grad = 1.0
        for node in reversed(topo):
            node._backward()

    # Helper methods to make building graphs easier
    def __radd__(self, other): return self + other
    def __rmul__(self, other): return self * other
    def __neg__(self): return self * -1
    def __sub__(self, other): return self + (-other)
    def __truediv__(self, other): return self * (other**-1)
    def __repr__(self):
        return f"Value(data={self.data:.4f}, grad={self.grad:.4f})"