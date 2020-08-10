import torch as t
from torch.autograd import Variable as v
import torch


m = v(t.FloatTensor([[2, 3]]), requires_grad=True)

j = t.zeros(2 ,2)
k = v(t.zeros(1, 2))

k[0, 0] = m[0, 0] ** 2 + 3 * m[0 ,1]
k[0, 1] = m[0, 1] ** 2 + 2 * m[0, 0]
k.backward(t.FloatTensor([[1, 0]]), retain_graph=True)
print(m.grad)
j[:, 0] = m.grad.data
m.grad.data.zero_()
k.backward(t.FloatTensor([[0, 1]]))
print(m.grad)
j[:, 1] = m.grad.data
print('jacobian matrix is')
print(j)