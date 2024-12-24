import torch
import torch.nn as nn

import matplotlib.pyplot as plt

class GELU(nn.Module):
    def __init__(self_):
        '''
        Class implementation of the Gaussian Error Linear Unit (GELU) activation function.
        https://arxiv.org/pdf/1606.08415v5
        '''
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Performs the forward pass of the GELU activation function.
        
        Args:
            x (torch.Tensor): A tensor of any shape.
        Returns:
            torch.Tensor: A tensor of the same shape as x, representing the output of the GELU activation function.
        Raises:
            TypeError: If x is not a torch.Tensor.
        '''
        if not isinstance(x, torch.Tensor):
            raise TypeError("Input must be a PyTorch tensor")
        
        return 0.5 * x * (1 + torch.tanh(torch.sqrt(torch.tensor(2/torch.pi)) * (x + 0.044715 * torch.pow(x, 3))))
    

class SwiGLU(nn.Module):
    def __init__(self):
        '''
        Class implementation of the Swish activation function.
        https://arxiv.org/pdf/1710.05941v1
        '''
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Performs the forward pass of the Swish activation function.
        
        Args:
            x (torch.Tensor): A tensor of any shape.
        Returns:
            torch.Tensor: A tensor of the same shape as x, representing the output of the Swish activation function.
        Raises:
            TypeError: If x is not a torch.Tensor.
        '''
        if not isinstance(x, torch.Tensor):
            raise TypeError("Input must be a PyTorch tensor")
        
        return x * torch.sigmoid(x)
