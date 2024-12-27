import torch
import torch.nn as nn
from typing import Tuple

from layer_normalisation import LayerNormalisation
from transformer_block import Transformer

class GPTModel(nn.Module):
    def __init__(self,
                 vocabulary_size: int,
                 embedding_dimension: int,
                 context_length: int,
                 heads_count: int,
                 transformer_layers: int,
                 dropout_rate: float | Tuple[float, float, float] = 0.2,
                 bias_flag: bool = True):
        '''
        Implementation of the GPT model.
        '''
        assert isinstance(vocabulary_size, int) and vocabulary_size > 0, 'Vocabulary size must be a positive integer'
        assert isinstance(embedding_dimension, int) and embedding_dimension > 0, 'Embedding dimension must be a positive integer'
        assert isinstance(context_length, int) and context_length > 0, 'Context length must be a positive integer'
        assert isinstance(heads_count, int) and heads_count > 0, 'Heads count must be a positive integer'
        assert isinstance(transformer_layers, int) and transformer_layers > 0, 'Transformer layers must be a positive integer'
        assert (isinstance(dropout_rate, tuple) and len(dropout_rate) == 3 and all(isinstance(i, float) for i in dropout_rate)) or \
                isinstance(dropout_rate, float), 'Dropout rate must be a float or a tuple of 3 floats'
        assert isinstance(bias_flag, bool), 'Bias flag must be a boolean'

        super().__init__()
        self.token_embeddings = nn.Embedding(vocabulary_size, embedding_dimension)
        self.positional_embeddings = nn.Embedding(context_length, embedding_dimension)
        self.dropout = nn.Dropout(dropout_rate[0] if isinstance(dropout_rate, tuple) else dropout_rate)

        self.transformer_blocks = nn.Sequential(
                                                *[Transformer(embedding_dimension=embedding_dimension, 
                                                              context_length=context_length, 
                                                              heads_count=heads_count, 
                                                              dropout_rate=dropout_rate if isinstance(dropout_rate, float) else dropout_rate[1:],
                                                              bias_flag=bias_flag) for _ in range(transformer_layers)]
                                                )
        
        self.layer_normalisation = LayerNormalisation(embedding_dimension)
        self.output = nn.Linear(embedding_dimension, vocabulary_size, bias=False)

    def forward(self, decoded_sequence: torch.Tensor) -> torch.Tensor:
        '''
        Performs the forward pass of the GPT model.
        Args:
            decoded_sequence (torch.Tensor): A 2-dimensional tensor of shape (batch_size, sequence_length) representing the input tensor.
        Returns:
            torch.Tensor: A tensor of shape (batch_size, sequence_length, vocabulary_size) representing the output of the GPT model.
        Raises:
            TypeError: If decoded_sequence is not a torch.Tensor.
            ValueError: If decoded_sequence does not have 2 dimensions.
        '''
        if not isinstance(decoded_sequence, torch.Tensor):
            raise TypeError('Input tensor must be a torch.Tensor')
        if len(decoded_sequence.shape) != 2:
            raise ValueError('Input tensor must have 2 dimensions')
        
        _, sequence_length = decoded_sequence.shape
        
        # Generates the positional embeddings for the total sequence length: positions = self.positional_embeddings([0:sequence_length])
        positional_indices = torch.arange(sequence_length, device=decoded_sequence.device)
        positional_embedding = self.positional_embeddings(positional_indices)
        
        # decoded_sequence contains rows of integers representing the tokens within the vocabulary.
        token_embedding = self.token_embeddings(decoded_sequence)

        # Since self-attention is order-agnostic, the positional embeddings ensure the model can understand the order of the tokens.
        # Therefore, we inject the positional embeddings into the token embeddings to turn it into a more enriched input sequence.
        input_sequence = token_embedding + positional_embedding
        input_sequence = self.dropout(input_sequence)
        
        # Upon input enrichment and regularisation, we pass the input sequence through the multiple transformer blocks,
        # layer-normalise the context vector and pass it through a linear layer to compute the logits.
        context_vector = self.transformer_blocks(input_sequence)
        normalised_final_output = self.layer_normalisation(context_vector)
        logits = self.output(normalised_final_output)
        return logits