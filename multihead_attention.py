import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    '''
    Implementation of the multi-head attention mechanism.
    '''
    def __init__(self, 
                 input_dimension: int, 
                 output_dimension: int, 
                 context_length: int,
                 heads_count: int,
                 bias_flag: bool = True,
                 dropout_rate: float = 0.25):
        super().__init__()
        assert isinstance(input_dimension, int) and input_dimension > 0, 'Input dimension must be a positive integer'
        assert isinstance(output_dimension, int) and output_dimension > 0, 'Output dimension must be a positive integer'
        assert isinstance(context_length, int) and context_length > 0, 'Context length must be a positive integer'
        assert isinstance(heads_count, int) and heads_count > 0, 'Heads count must be a positive integer'
        assert isinstance(bias_flag, bool), 'Bias flag must be a boolean'
        assert isinstance(dropout_rate, (float, int)) and 0 <= dropout_rate < 1, 'Dropout rate must be a float between 0 and 1'
        assert input_dimension % heads_count == 0, 'Input dimension must be divisible by heads count'

        self.input_dimension = input_dimension
        self.output_dimension = output_dimension
        self.heads_count = heads_count
        self.head_dimension = input_dimension // heads_count

        self.dropout = nn.Dropout(dropout_rate)

        self.query = nn.Linear(input_dimension, output_dimension, bias=bias_flag)
        self.key = nn.Linear(input_dimension, output_dimension, bias=bias_flag)
        self.value = nn.Linear(input_dimension, output_dimension, bias=bias_flag)
        self.output = nn.Linear(output_dimension, input_dimension, bias=bias_flag) # similar to out_proj in the original implementation

        # The causal mask is a matrix that ensures that the attention mechanism does not consider tokens that occur after the current token.
        # I have implemented the causal mask as a buffer, so that it is not a parameter of the model and loads properly on the specified device.
        self.register_buffer('mask', torch.triu(torch.ones(context_length, context_length), diagonal=1))
    
    def forward(self, input_sequences: torch.Tensor) -> torch.Tensor:
        '''
        Performs the forward pass of the multi-head attention mechanism.
        Args:
            input_sequences (torch.Tensor): A 3-dimensional tensor of shape 
                (batch_size, sequence_length, embedding_dim) representing the input sequences.
        Returns:
            torch.Tensor: A tensor of shape (batch_size, sequence_length, embedding_dim) 
                representing the output of the multi-head attention mechanism.
        Raises:
            TypeError: If input_sequences is not a torch.Tensor.
            ValueError: If input_sequences does not have 3 dimensions.
        '''        
        if not isinstance(input_sequences, torch.Tensor):
            raise TypeError("Input sequence must be a PyTorch tensor")
        if input_sequences.dim() != 3:
            raise ValueError("Input sequence must have 3 dimensions")

        query, key, value = self._format_attention_input(input_sequences)
        attention_scores = self._compute_attention_scores(query, key)
        attention_scores = self._implement_causal_masking(attention_scores, input_sequences)
        attention_weights = self._compute_attention_weights(attention_scores)
        # The attention weights are multiplied by the value to get the weighted sum of the values, which represents the content of the input sequence.
        context_vector = attention_weights.matmul(value)
        context_vector = self._revert_to_original_shape(context_vector)
        context_vector = self.output(context_vector)
        return context_vector


    def _format_attention_input(self, input_sequence: torch.Tensor) -> torch.Tensor:
        # Perform linear transformation to get query, key and value
        query: torch.Tensor = self.query(input_sequence)
        key: torch.Tensor = self.key(input_sequence)
        value: torch.Tensor = self.value(input_sequence)

        # Get the batch size, number of tokens, and the dimension of the input sequence
        batch_size, tokens_count, _ = query.shape

        # Reshape the query, key, and value to have the correct shape for the attention mechanism
        # by dividing the embedding dimension into the number of heads and their respective the head dimension.
        # After, we transpose the query, key, and value to have the shape (batch_size, heads_count, tokens_count, head_dimension),
        # so that we can perform the attention mechanism across the heads in parallel.
        query = query.contiguous().view(batch_size, tokens_count, self.heads_count, self.head_dimension).transpose(1, 2)
        key = key.contiguous().view(batch_size, tokens_count, self.heads_count, self.head_dimension).transpose(1, 2)
        value = value.contiguous().view(batch_size, tokens_count, self.heads_count, self.head_dimension).transpose(1, 2)
        return query, key, value
    
    def _compute_attention_scores(self, query: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
        # Self-attention computes the relationship between each query and every key, in order to determine the relevance of each input token.
        # The key is transposed so that the dot product can be computed between the query and key and the result is regularised by the square root
        # of the embedding dimension, so that vanishing gradient is mitigated.
        attention_scores = query.matmul(key.transpose(-1, -2)) / (self.head_dimension ** 0.5)
        attention_scores.to(torch.float64) # Ensure that the attention scores are in float64 --not ideal
        return attention_scores
    
    def _implement_causal_masking(self, attention_scores: torch.Tensor, input_sequences: torch.Tensor) -> torch.Tensor:
        # The causal mask is a matrix that ensures that the attention mechanism does not consider tokens that occur after the current token.
        # This is important for language generation tasks, where the model should not have access to future tokens.
        token_count = input_sequences.shape[1]
        masking = self.mask[:token_count, :token_count].bool()
        masked_attention_scores = attention_scores.masked_fill_(masking, -torch.inf)
        return masked_attention_scores
    
    def _compute_attention_weights(self, masked_attention_scores: torch.Tensor) -> torch.Tensor:
        # The attention weights are computed by applying the softmax function to the masked attention scores.
        # This ensures that the weights sum to 1 and represent the importance of each token in the input sequence.
        # As regularisation, we apply dropout to the attention weights, which helps to prevent overfitting by ensuring
        # that the model does not rely too heavily on any one token.
        attention_weights = nn.functional.softmax(masked_attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        return attention_weights

    def _revert_to_original_shape(self, context_vector: torch.Tensor) -> torch.Tensor:
        # The context vector is reshaped to have the same shape as the input sequence, so that it can be passed to the next layer.
        print(context_vector.shape)
        batch_size, _, tokens_count, _ = context_vector.shape

        # In _format_attention_input, we transposed the query, key, and value to have the shape (batch_size, heads_count, tokens_count, head_dimension),
        # Now, in order to revert to the original shape, we transpose the context vector to have the shape (batch_size, tokens_count, heads_count, head_dimension),
        # and then reshape it to have the shape (batch_size, tokens_count, output_dimension).
        context_vector = context_vector.transpose(1, 2)
        context_vector = context_vector.contiguous().view(batch_size, tokens_count, self.output_dimension)
        return context_vector
        