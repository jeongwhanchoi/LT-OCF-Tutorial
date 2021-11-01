import torch
import torch.nn as nn 
import world
if world.adjoint:
    from torchdiffeq import odeint_adjoint as odeint
else:
    from torchdiffeq import odeint
"""
# ODE Block and ODE function
We use the [torchDiffeq package](https://github.com/rtqichen/torchdiffeq) to design co-evolutionary ODEs.
"""

class ODEFunction(nn.Module):
    def __init__(self, Graph):
        super(ODEFunction, self).__init__()
        self.g = Graph

    def forward(self, t, x):
        out = torch.sparse.mm(self.g, x)
        """
        ## linear GCN (non-time-dependent) in ODE function
        Its linear graph convolutional layer definition is as follows:

        \begin{align}
        \boldsymbol{E}_{k} = \tilde{\boldsymbol{A}}\boldsymbol{E}_{k-1},
        \end{align}

        where $\tilde{\boldsymbol{A}} is a normalized adjacency matrix of the graph.
        """
        return out

class ODEBlock(nn.Module):
    """## ODE Block"""
    def __init__(self, odeFunction, solver, init_time, final_time):
        super(ODEBlock, self).__init__()
        self.odefunc = odeFunction
        self.integration_time = torch.tensor([init_time,final_time]).float()
        self.solver = solver

    def forward(self, x):
        self.integration_time = self.integration_time.type_as(x)
        out = odeint(func=self.odefunc, y0=x, t=self.integration_time, method=self.solver)
        """### return the last time of ODE Integration
        \begin{align}
        \boldsymbol{h}(t_{i+1}) = \boldsymbol{h}(t_i) + \int_{t_i}^{t_{i+1}}f(\boldsymbol{h}(t),t;\boldsymbol{\theta}_f)dt,
        \end{align}
        """
        return out[1]

class ODEBlockTimeFirst(nn.Module):
    def __init__(self, odeFunction, num_split, solver):
        super(ODEBlockTimeFirst, self).__init__()
        self.odefunc = odeFunction
        self.num_split = num_split
        self.zero = torch.tensor([0.], requires_grad=False).to('cuda')
        self.solver = solver
                    
    def forward(self, x, t):
        odetime = t
        odetime_tensor = torch.cat(odetime, dim=0)
        all_time = torch.cat([self.zero, odetime_tensor], dim = 0).to('cuda')
        
        all_time1 = all_time.type_as(x)
        total_integration_time = all_time1
        if self.solver == 'euler':
            out = odeint(func = self.odefunc, y0 = x, t = total_integration_time, method=self.solver)
        elif self.solver == 'dopri5':
            out = odeint(func = self.odefunc, y0 = x, t = total_integration_time, method=self.solver, rtol=world.rtol, atol=world.atol)
        elif self.solver == 'rk4':
            out = odeint(func = self.odefunc, y0 = x, t = total_integration_time, method=self.solver)
        else:
            out = odeint(func = self.odefunc, y0 = x, t = total_integration_time, method=self.solver)
        return out[1]

class ODEBlockTimeMiddle(nn.Module):
    def __init__(self, odeFunction, num_split, solver):
        super(ODEBlockTimeMiddle, self).__init__()
        self.odefunc = odeFunction
        self.num_split = num_split
        self.solver = solver

    def forward(self, x, t1, t2):
        odetime_1 = t1
        odetime_2 = t2
        odetime_1_tensor = torch.cat(odetime_1, dim=0)
        odetime_2_tensor = torch.cat(odetime_2, dim=0)
        all_time = torch.cat([odetime_1_tensor, odetime_2_tensor], dim = 0).to('cuda')

        all_time1 = all_time.type_as(x)
        total_integration_time = all_time1
        if self.solver == 'euler':
            out = odeint(func = self.odefunc, y0 = x, t = total_integration_time, method=self.solver)
        elif self.solver == 'dopri5':
            out = odeint(func = self.odefunc, y0 = x, t = total_integration_time, method=self.solver, rtol=world.rtol, atol=world.atol)
        elif self.solver == 'rk4':
            out = odeint(func = self.odefunc, y0 = x, t = total_integration_time, method=self.solver)
        else:
            out = odeint(func = self.odefunc, y0 = x, t = total_integration_time, method=self.solver)
        return out[1]

class ODEBlockTimeLast(nn.Module):
    def __init__(self, odeFunction, num_split, solver):
        super(ODEBlockTimeLast, self).__init__()
        self.odefunc = odeFunction
        self.num_split = num_split
        self.one = torch.tensor([4.], requires_grad=False).to('cuda')
        self.solver = solver
        
    def forward(self, x, t):
        odetime = t
        odetime_tensor = torch.cat(odetime, dim=0)
        all_time = torch.cat([odetime_tensor,self.one], dim = 0).to('cuda')

        all_time1 = all_time.type_as(x)
        total_integration_time = all_time1
        if self.solver == 'euler':
            out = odeint(func = self.odefunc, y0 = x, t = total_integration_time, method=self.solver)
        elif self.solver == 'dopri5':
            out = odeint(func = self.odefunc, y0 = x, t = total_integration_time, method=self.solver, rtol=world.rtol, atol=world.atol)
        elif self.solver == 'rk4':
            out = odeint(func = self.odefunc, y0 = x, t = total_integration_time, method=self.solver)
        else:
            out = odeint(func = self.odefunc, y0 = x, t = total_integration_time, method=self.solver)
        return out[1]

def ODETime(num_split):
        return [torch.tensor([1 / num_split * i], dtype=torch.float32, requires_grad=True, device='cuda') for
                i in range(1, num_split)]

def ODETimeSetter(num_split, K):
        eta = K/ num_split
        return [torch.tensor([i*eta], dtype=torch.float32, requires_grad=True, device='cuda') for i in range(1, num_split)]

def ODETimeSplitter(num_split, K):
        eta = K / num_split
        return [i*eta for i in range(1, num_split)]