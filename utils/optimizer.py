import copy
import math
import torch

# class ExperimentOptimizer(torch.optim.Optimizer): 
#     # Init Method: 
#     def __init__(self, params, lr=1e-2): 
#         self.loss = None
#         self.lr = lr
#         super(ExperimentOptimizer, self).__init__(params, defaults={'lr': lr})
      
#     # Step Method 
#     def step(self,loss): 
#         if self.loss!= None and loss>self.loss:
#             print(self.loss,loss,self.lr*self.loss/loss)
#             self.lr = self.lr*self.loss/loss * 0.9
#         self.loss = loss
#         for group in self.param_groups: 
#             for param in group['params']: 
#                 param.data -= torch.ones_like(param.data)*self.lr/param.grad.data 


class AdaptiveOptimizer(torch.optim.Optimizer): 
    # Init Method: 
    def __init__(self, params, lr=1e-2, gamma=0.9): 
        self.lr = lr
        self.s=1
        self.gamma = gamma
        self.gradient_num = 0
        super(AdaptiveOptimizer, self).__init__(params, defaults={'lr': lr})
      
    # Step Method 
    def step(self): 
        if self.gradient_num == 0: 
            for group in self.param_groups: 
                for param in group['params']: 
                    if param.grad != None:
                        self.gradient_num += torch.numel(param.grad.data)
        gradients_total = 0
        for group in self.param_groups: 
            for param in group['params']: 
                if param.grad != None:
                    gradients_total += torch.sum(param.grad.data**2)
        self.s = self.gamma * self.s + (1-self.gamma) * gradients_total / self.gradient_num
        for group in self.param_groups: 
            for param in group['params']: 
                if param.grad != None:
                    param.data -= self.lr / math.sqrt(self.s) * param.grad.data 


if __name__ == '__main__': 
    model1 = torch.nn.Linear(1, 1)
    model2 = copy.deepcopy(model1)
    model3 = copy.deepcopy(model1)
    criterion = torch.nn.MSELoss()
    optimizer1 = AdaptiveOptimizer(model1.parameters(), lr=1,gamma=0.9)
    optimizer2 = torch.optim.SGD(model2.parameters(), lr=0.01)
    optimizer3 = torch.optim.Adam(model3.parameters(), lr=1)

    x = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
    y = torch.tensor([[2.0], [4.0], [6.0], [8.0]])

    print('Adaptive Optimizer')
    for epoch in range(101):
        optimizer1.zero_grad()
        output = model1(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer1.step()
        if epoch % 100 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item()}')
    
    print('SGD Optimizer')
    for epoch in range(101):
        optimizer2.zero_grad()
        output = model2(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer2.step()
        if epoch % 100 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item()}')

    print('Adam Optimizer')
    for epoch in range(101):
        optimizer3.zero_grad()
        output = model3(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer3.step()
        if epoch % 100 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item()}')