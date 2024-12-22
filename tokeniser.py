from typing import List
import re
import tiktoken

#TODO: Write my own byte-pair encoder
class Tokeniser:
    '''
    A class to tokenise and detokenise text.
    '''
    def __init__(self):
        self.tokeniser = tiktoken.get_encoding('gpt2')

    def encode(self, text: str) -> List[int]:
        '''
        Receives a text and returns a list of tokens.
        Args:
            text (str): The text to be tokenised.
        Returns:
            List[int]: A list of integers representing the tokens.
        '''
        if not isinstance(text, str):
            raise TypeError('Text must be a string')
        encoded_text = self.tokeniser.encode(text, allowed_special={'<|endoftext|>'})
        return encoded_text
    
    def decode(self, tokens: List[int]) -> str:
        '''
        Receives a list of tokens and returns the decoded text.
        Args:
            tokens (List[int]): A list of integers representing the tokens.
        Returns:
            str: The decoded text.
        '''
        if not isinstance(tokens, list) or not all(isinstance(token, int) for token in tokens):
            raise TypeError('Tokens must be integers')
        decoded_text = self.tokeniser.decode(tokens)
        return decoded_text
