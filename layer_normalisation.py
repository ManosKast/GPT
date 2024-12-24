import torch
import torch.nn as nn

class LayerNormalisation(nn.Module):    
    def __init__(self, embedding_dimension: int, epsilon: float = 1e-6):
        '''
        Class implementation of the layer normalisation mechanism. It contains trainable parameters for the scale and bias.
        This allows the model to learn the optimal scale and bias for the normalisation, based on the data fed to it.
        
        Args:
            embedding_dimension (int): The dimension of the embeddings.
            epsilon (float): A small value to prevent division by zero, during the normalisation process. Default is 10^-6.
        '''
        assert isinstance(embedding_dimension, int) and embedding_dimension > 0, 'Embedding dimension must be a positive integer'
        assert isinstance(epsilon, (float, int)) and epsilon > 0, 'Epsilon must be a positive float'
        super().__init__()
        # The scale and bias are trainable parameters, because they could learn the optimal values for the normalisation of the
        # given data. They are initialised to 1 and 0, respectively, because these are the default values for normalisation.
        self.scale = nn.Parameter(torch.ones(embedding_dimension))
        self.bias = nn.Parameter(torch.zeros(embedding_dimension))
        self.epsilon = epsilon

    def forward(self, input_sequences: torch.Tensor) -> torch.Tensor:
        '''
        Performs the forward pass of the layer normalisation mechanism.
        
        Args:
            input_sequences (torch.Tensor): A 3-dimensional tensor of shape 
                (batch_size, sequence_length, embedding_dimension) representing the input sequences.
        Returns:
            torch.Tensor: A tensor of shape (batch_size, sequence_length, embedding_dimension) 
                representing the output of the layer normalisation mechanism.
        Raises:
            TypeError: If input_sequences is not a torch.Tensor.
            ValueError: If input_sequences does not have 3 dimensions.
        '''        
        if not isinstance(input_sequences, torch.Tensor):
            raise TypeError("Input sequence must be a PyTorch tensor")
        #if input_sequences.dim() != 3:
        #    raise ValueError("Input sequence must have 3 dimensions")
        
        mean = input_sequences.mean(dim=-1, keepdim=True)
        std = input_sequences.std(dim=-1, keepdim=True)
        normalised_input = (input_sequences - mean) / (std + self.epsilon)
        return self.scale * normalised_input + self.bias