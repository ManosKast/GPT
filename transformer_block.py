import torch
import torch.nn as nn
from typing import Tuple

from feedforward_network import FeedForwardNetwork
from layer_normalisation import LayerNormalisation
from multihead_attention import MultiHeadAttention

class Transformer(nn.Module):
    def __init__(self,
                 embedding_dimension: int,
                 context_length: int,
                 heads_count: int = 8,
                 dropout_rate: float | Tuple[float, float] = 0.2,
                 bias_flag: bool = True):
            '''
            Implementation of the transformer architecture.
            '''
            assert isinstance(embedding_dimension, int) and embedding_dimension > 0, 'Embedding dimension must be a positive integer'
            assert isinstance(context_length, int) and context_length > 0, 'Context length must be a positive integer'
            assert isinstance(heads_count, int) and heads_count > 0, 'Heads count must be a positive integer'
            assert (isinstance(dropout_rate, tuple) and len(dropout_rate) == 2 and all(isinstance(i, float) for i in dropout_rate)) or \
                    isinstance(dropout_rate, float), 'Dropout rate must be a float or a tuple of 3 floats'
            assert isinstance(bias_flag, bool), 'Bias flag must be a boolean'

            super().__init__()

            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            layer_normalisation_1 = LayerNormalisation(embedding_dimension).to(device)
            layer_normalisation_2 = LayerNormalisation(embedding_dimension).to(device)
            self.layer_normalisation = nn.ModuleList([layer_normalisation_1, layer_normalisation_2])

            self_attention_mechanism = MultiHeadAttention(input_dimension=embedding_dimension,
                                                               output_dimension=embedding_dimension,
                                                               context_length=context_length,
                                                               heads_count=heads_count,
                                                               bias_flag=bias_flag,
                                                               dropout_rate=dropout_rate[0] if isinstance(dropout_rate, tuple) else dropout_rate)
            self_attention_mechanism.to(device)
            feedforward_network = FeedForwardNetwork(embedding_dimension=embedding_dimension).to(device)
            self.networks = nn.ModuleList([self_attention_mechanism, feedforward_network])

            dropout_1 = nn.Dropout(dropout_rate[1] if isinstance(dropout_rate, tuple) else dropout_rate)    
            dropout_2 = nn.Dropout(dropout_rate[1] if isinstance(dropout_rate, tuple) else dropout_rate)
            self.dropout = nn.ModuleList([dropout_1, dropout_2])


    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        '''
        Performs the forward pass of the transformer.
        Args:
            input_tensor (torch.Tensor): A 3-dimensional tensor of shape 
                (batch_size, sequence_length, embedding_dimension) representing the input tensor.
        Returns:
            torch.Tensor: A tensor of shape (batch_size, sequence_length, embedding_dimension) 
                representing the output of the transformer.
        Raises:
            TypeError: If input_tensor is not a torch.Tensor.
            ValueError: If input_tensor does not have 3 dimensions.
        '''
        if not isinstance(input_tensor, torch.Tensor):
            raise TypeError("Input tensor must be a PyTorch tensor")
        if input_tensor.dim() != 3:
            raise ValueError("Input tensor must have 3 dimensions")

        # The input tensor is regularised by the first layer normalisation, which is followed by the self-attention mechanism.
        # The output of the self-attention mechanism is regularised by the dropout layer and added to the shortcut connection.
        # The result is regularised by the second layer normalisation, followed by the feedforward network.
        # The output of the feedforward network is regularised by the dropout layer and added to the shortcut connection,
        # which is the final output of the transformer.
        x: torch.Tensor = input_tensor
        for network, layer_normalisation, dropout in zip(self.networks, self.layer_normalisation, self.dropout):
            shortcut = x
            x = layer_normalisation(x)
            x = network(x)
            x = dropout(x)
            x = shortcut + x
        
        return x
        
        