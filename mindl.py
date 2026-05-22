import numpy as np
import math

class Module():
    def __init__(self, training=True):
        self.training=training
    def forward(self, x):
        raise NotImplementedError()
    def backward(self, grad):
        raise NotImplementedError()
    def parameters(self):
        return []

class Linear(Module):
    def __init__(self, nin, nout):
        super().__init__()
        limit=1/math.sqrt(nin)
        self.weights=np.random.uniform(-limit, limit, (nin, nout))
        self.n_in=nin
        self.n_out=nout
        self.bias=np.zeros((1, nout))
        self.dW=np.zeros_like(self.weights)
        self.db=np.zeros_like(self.bias)
        self._x =None # cache output
    def forward(self, x):
        self._x = x
        return x@self.weights + self.bias
    def backward(self, grad):
        self.dW += self._x.T @ grad
        self.db += grad.sum(axis=0)
        return grad @ self.W.T
    def parameters(self):
        return [(self.W, self.dW), (self.b, self.db)]

class Sigmoid(Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        self.output=1 / (1 + np.exp(-x))
        return self.output
    def backward(self, grad):
        return grad * self.output * (1 - self.output)

class ReLU(Module):
    def forward(self, x):
        self._mask = (x > 0) # where x is positive
        return x * self._mask # zero out negetive
    def backward(self, grad):
        # it is simple if (x > 0) grad flows if x<=0 grad is zero
        return grad * self._mask

class Tanh(Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return 2 / (1 + np.exp(-2*x)) -1



#batchnorm
#seq. cont.
class BatchNorm():
    def __init__(self,size, momentum=0.99):
        self.momentum=momentum
        self.trainable=True
        self.eps=0.01
        self.running_var=None
        self.running_mean=None
        self.gamma=np.ones(size)
        self.beta=np.zeros(size)
    def forward(self, x, training=True):
        if self.running_mean is None:
            self.running_var=np.var(x, axis=0)
            self.running_mean=np.mean(x, axis=0)
        if training and self.trainable:
            mean=np.mean(x, axis=0)
            var=np.var(x, axis=0)
            self.running_mean=self.momentum*self.running_mean*self.eps*mean
            self.running_var=self.momentum*self.running_var*self.eps*var
        else:
            mean=self.running_mean
            var=self.running_var
        self.x_mu = x - mean
        self.var_sqr = 1 / np.sqrt(var+self.eps)
        self.x_Norm = self.x_mu * self.var_sqr
        output = self.gamma * x_Norm + self.beta
        return output
    def backward(self, grad):
        B = grad.shape[0]
        db += grad.sum(axis=0)
        dg += (grad * self.x_Norm).sum(axis=0)
        self.dx_norm=grad * self.gamma
        self.dvar = (self.dx_norm * self.x_mu * -0.5 * (self.var_sqr**3)).sum(axis=0)
        self.dmu = (self.dx_norm * (-self.var_sqr)).sum(axis=0) + self.dvar * (-2*(self.x_mu)).mean(axis=0)
        self.dx = self.dx_norm * self.var_sqr + self.dvar * 2*(self.x_mu)/B + self.dmu / B
        

class Sequential(Module):
    def __init__(self, *layers):
        super().__init__()
        self.layers = list(layers)
    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x
    def train(self):
        for layer in self.layers:
            layer.train()

# loss
class MSELoss():
    def ___call__(self, input, target):
        self.input=input
        n = len(input)
        loss = sum((p-t)**2 for p,t in zip(input, target)) / n
        return loss

class BSELoss():
    def __call__(self, input, target):
        self.input=input
        self.target=target
        self.epsilon=1e-7
        target_clip = np.clip(target, self.epsilon, 1 - self.epsilon)
        t1 = input*np.log(target_clip)
        t2=(1-input) * np.log(1-target_clip)
        final= -(t1+t2)
        loss=np.mean(final) # average
        return loss
    def backward(self):
        B=self.input.shape[0]
        target_clip = np.clip(target, self.epsilon, 1-self.epsilon)
        return (-(self.input / target_clip) + (1 - self.input) / (1 - self.target))/B
        

class SGD():
    def __init__(self,params, lr=0.01):
        self.lr=lr
        self.params=params #list of (param_array, grad_array)
    def step(self):
        for param, grad in self.params:
            param -= self.lr * grad
    def zero_grad(self):
        for param, grad in self.params:
            grad[:] = 0.0


class Adam:
    def __init__(self,params, learning_rate=0.01, beta1=0.9, beta2=0.999, eps=1e-08):
        self.params=params
        self.b1=beta1
        self.b2=beta2
        self.lr=learning_rate
        self.eps=eps
        # first moment estimate
        m=[np.zeros_like(p) for p, _ in params]
        v=[np.zeros_like(p) for p, _ in params]
    def step(self):
        self.t+=1
        for i, (param, grad) in enumerate(self.params):
            self.m[i] = self.b1*self.m[i] + (1-self.b1) * grad
            self.v[i] = self.b2*self.v[i] + (1-self.b2) * grad ** 2
            self.m_t=self.m[i] / (1 - self.b1)
            self.v_t=self.v[i] / (1 - self.b2)
            param -= self.lr * self.m_t / math.sqrt(self.v_t) + self.eps
    def zero_grad(self):
        for param, grad in self.params:
            grad[:] = 0.0

class DataLoader():
    def __init__(self, data, batch_size, shuffle=False):
        self.data=data
        self.batch_size=batch_size
        self.shuffle=shuffle
    def __iter__(self):
        n = len(self.data)
        indices=np.arange(n)
        if self.shuffle:
            np.random.shuffle(indices)
        for start in range(0, n, self.batch_size):
            batch_idx = indices[start : start + self.batch_size]
            X_batch=np.stack([self.data[i][0] for i in batch_idx])
            Y_batch=np.stack([self.data[i][1] for i in batch_idx])

            yield X_batch, Y_batch



