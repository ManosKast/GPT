import torch
import torch.nn as nn

from activation_functions import *

class FeedForwardNetwork(nn.Module):
    def __init__(self, embedding_dimension: int, output_dimension: int | None = None):
        '''
        Implementation of the feedforward network used in the transformers architecture, 
        which consists of two linear layers with a GELU activation function.
        '''
        assert isinstance(embedding_dimension, int) and embedding_dimension > 0, 'Embedding dimension must be a positive integer'
        assert isinstance(output_dimension, int) or output_dimension is None, 'Output dimension must be an integer or None'
        
        super().__init__()
        if output_dimension is None:
            output_dimension = 4 * embedding_dimension

        self.layers = nn.Sequential(
            nn.Linear(embedding_dimension, output_dimension),
            GELU(),
            nn.Linear(output_dimension, embedding_dimension)
        )
    
    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        '''
        Performs the forward pass of the feedforward network.
        
        Args:
            input_tensor (torch.Tensor): A 3-dimensional tensor of shape 
                (batch_size, sequence_length, embedding_dimension) representing the input tensor.
        Returns:
            torch.Tensor: A tensor of shape (batch_size, sequence_length, embedding_dimension) 
                representing the output of the feedforward network.
        Raises:
            TypeError: If input_tensor is not a torch.Tensor.
            ValueError: If input_tensor does not have 3 dimensions.
        '''
        # There's no need to check the type of input_tensor, since it has been checked within the Transformer class.
        return self.layers(input_tensor)
        