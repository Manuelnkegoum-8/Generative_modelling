import torch
from torch import nn
from torch.distributions.multivariate_normal import MultivariateNormal
import torch.nn.init as init
import numpy as np

class Base_model(nn.Module):
    """
    feed forward network for defining the operations of scaling and translation needed in realnvp
    """
    def __init__(self,in_dim=2,out_dim=2,hidden_dim=8):
        super(Base_model,self).__init__()
        self.fcn = nn.Sequential(
                nn.Linear(in_dim,hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim,hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim,out_dim),
        )        
        self.initialize_weights()
    def forward(self, x):
        y = self.fcn(x)
        return y
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)


class Coupling_layer(nn.Module):
    """Coupling layer in RealNVP.

    Args:
        dim (int): Number of channels in the input.
        hidden_dim (int): Number of channels in the `s` and `t` network.
        mask_type (tensor): binary mask to have in track what value should be kept inchanged
    """
    def __init__(self,dim=4,hidden_dim=32,mask=None):
        super(Coupling_layer,self).__init__()
        self.dim = dim
        self.t1 = Base_model(dim,dim,hidden_dim)
        self.s1 = Base_model(dim,dim,hidden_dim)
        self.mask = mask # binaty mask (1 then  value not modified 0 value to be modified)

    def forward(self, x): 
        #observed to latent space
        keeped = x*self.mask
        t1 = self.t1(keeped)
        s1 = torch.tanh(self.s1(keeped))
        z = keeped + (1-self.mask)*(x-t1)*torch.exp(-s1)
        log_det = torch.sum((1-self.mask)*-s1, dim=1)
        return z, log_det

    def inverse(self, z):
        #latent to observed
        keeped = z*self.mask
        t1 = self.t1(keeped)
        s1 = torch.tanh(self.s1(keeped))
        x = keeped + (1-self.mask)*(z*torch.exp(s1) +t1)
        log_det = torch.sum((1-self.mask)*s1, dim=1)
        
        return x, log_det



class Real_nvp(nn.Module):
    """RealNVP Model

    Based on the paper:
    "Density estimation using Real NVP"
    by Laurent Dinh, Jascha Sohl-Dickstein, and Samy Bengio
    (https://arxiv.org/abs/1605.08803).

    Args:
        dim (int): Number of channels in the input.
        hidden_dim (int): Number of channels in the intermediate layers.
        num_coupling (int): Number of `Coupling` layers.
    """
    def __init__(self,dim=4,hidden_dim=32,num_coupling=4,masks=None):
        super(Real_nvp,self).__init__()
        self.dim = dim
        self.layers = nn.ModuleList()
        self.num_coupling = num_coupling
        self.distribution = MultivariateNormal(torch.zeros(self.dim),torch.eye(self.dim))
        self.masks = nn.ParameterList(
            [nn.Parameter(torch.Tensor(m),requires_grad = False)
             for m in masks])
        for i in range(self.num_coupling):
            self.layers.append(Coupling_layer(self.dim,hidden_dim,self.masks[i]))


    def forward(self, x):
        #observed to latent space
        device_ = x.device
        log_det = torch.zeros(x.size(0),device=device_)
        x = torch.log(x/(1-x)) # inverse the normalization layer
        log_det += torch.sum(torch.log(torch.abs(1/(x*(1-x)))),-1)
        for i in range(self.num_coupling):
            x,det = self.layers[i](x)
            log_det +=det
        z = x
        pi = torch.tensor(np.pi)
        tmp = -0.5*(torch.norm(z,p=2,dim=1)**2 +self.dim*torch.log(2*pi))
        loss_batch = tmp.to(device_)+log_det
        return z,log_det,loss_batch
    
    def inverse(self,z):
        #latent to observed
        device_ = z.device
        log_det = torch.zeros(z.size(0),device=device_)
        for i in range(self.num_coupling-1,-1,-1):
            z,det = self.layers[i].inverse(z)
            log_det +=det
        x = torch.sigmoid(z)  #Normalize the output because we want it to be between 0 and 1
        log_det += torch.sum(torch.log(torch.abs(z*(1-z))),-1)
        return x,log_det

    def sample(self,num_samples,device):
        #generate samples
        z = self.distribution.sample((num_samples,))
        z = z.to(device)
        x,_ = self.inverse(z)
        return z,x


def train_one_epoch(model, train_loader, optimizer,device=None):
    model.train()
    avg_loss = 0.0
    anderson_metric, kendall_error = 0.0, 0.0
    for data in train_loader:
        optimizer.zero_grad()
        data = data.to(device)
        z,log_det,loss_batch = model(data)
        loss = -loss_batch.mean()
        loss.backward()
        optimizer.step()
        metrics = compute_metrics(data)
        anderson_metric += metrics[0].item()
        kendall_error += metrics[1].item()
        avg_loss += loss.item()
        del data
        torch.cuda.empty_cache()
    return avg_loss / len(train_loader), anderson_metric / len(train_loader) ,kendall_error / len(train_loader)