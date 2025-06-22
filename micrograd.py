class Value:
    def __init__(self, data, label='', _prev=(), _op=''):
        assert isinstance(data, (int, float)), "only support integer or float values"
        self.data = float(data) 
        self.label = label
        self._prev = set(_prev)
        self._op = _op

        self.grad = 0.0    
        self._backward = lambda: None

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        output = Value(self.data + other.data, f'({self.label}+{other.label})', (self, other), '+')

        def _backward():
            self.grad += 1.0 * output.grad
            other.grad += 1.0 * output.grad

            self._backward()
            other._backward()
        output._backward = _backward

        return output
    
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        output = Value(self.data * other.data, f'({self.label}*{other.label})', (self, other), '*')


        def _backward():
            self.grad += other.data * output.grad
            other.grad += self.data * output.grad

            self._backward()
            other._backward()
        output._backward = _backward

        return output

    def backward(self):
        self.grad = 1
        self._backward()
       