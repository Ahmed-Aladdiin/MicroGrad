import math

class Value:
    def __init__(self, data, label='', _prev=(), _op=''):
        assert isinstance(data, (int, float)), "only support integer or float values"
        self.data = float(data) 
        self.label = label
        self._prev = set(_prev)
        self._op = _op

        self.grad = 0.0    
        self._backward = lambda: None
    
    def __repr__(self):
        return f'Value(data: {self.data})'

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other, f'{other}')
        output = Value(self.data + other.data, f'({self.label}+{other.label})', (self, other), '+')

        def _backward():
            self.grad += 1.0 * output.grad
            other.grad += 1.0 * output.grad
        output._backward = _backward

        return output

    def __radd__(self, other):
        return self + other

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)
    
    def __pow__(self, pow):
        assert isinstance(pow, (int, float)), "only support integer or float values"
        x = self.data ** pow
        output = Value(x, f'({self.label}**{pow})', (self,), f'**{pow}')

        def _backward():
            self.grad += pow * (self.data ** (pow-1)) * output.grad
        output._backward = _backward

        return output

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other, f'{other}')
        output = Value(self.data * other.data, f'({self.label}*{other.label})', (self, other), '*')


        def _backward():
            self.grad += other.data * output.grad
            other.grad += self.data * output.grad
        output._backward = _backward

        return output
    
    def __rmul__(self, other):
        return self * other
    
    def __truediv__(self, other):
        return self * (other ** -1)

    def exp(self):
        x = math.exp(self.data)        
        output = Value(x, f'(e**{self.label})', (self, ), 'exp')


        def _backward():
            self.grad += x * output.grad

        output._backward = _backward

        return output
 
    def tanh(self):
        x = 2 * self.data
        x = (math.exp(x)- 1)/(math.exp(x) + 1)
        output = Value(x, f'tanh({self.data})', (self, ), 'tanh')

        def _backward():
            self.grad += (1-x**2) * output.grad
        output._backward = _backward

        return output
 
    def backward(self):
        topo = []
        visited = set()

        def fill(node):
            if node in visited: return
            visited.add(node)
            for child in node._prev:
                fill(child)
            topo.append(node)

        fill(self)
        
        self.grad = 1
        for node in reversed(topo):
            node._backward()
    
    def zero_grad(self):
        self.grad = 0