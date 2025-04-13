class Optimizers:
    def __init__(self, params, lr = 1e-5):
        self.params = params
        self.lr = lr
    
    def step(self):
        raise NotImplementedError

    def zero_grad(self):
        for p in self.params:
            p.zero_grad()

class SGD(Optimizers):
    def step(self):
        for p in self.params:
            p.data -= self.lr * p.grad
    def __repr__(self):
        return f"SGD, learning rate {self.lr}"