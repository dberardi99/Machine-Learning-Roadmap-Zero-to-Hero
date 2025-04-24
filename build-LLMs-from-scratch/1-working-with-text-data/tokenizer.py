import re
import json

class SimpleTokenizerV1: # this is not the real tokenizer used by GPT models, but it is a simplified version
    '''
    A first simplified version of tokenizer.
    '''
    def __init__(self, vocab): # the class takes a vocabulary as input
        self.str_to_int = vocab # regular vocabulary (string to integer mapping)
        self.int_to_str = {i: s for s, i in vocab.items()} # inverted vocabulary (integer to string mapping)

    def encode(self, text):
        '''
        A function to break down text into tokens using regular expressions.
        '''
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text) # creation of tokenized version of input text
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        ids = [self.str_to_int[s] for s in preprocessed] # creation of token ID for each token/string
        return ids
    
    def decode(self, ids):
        '''
        A function to re-create the original text starting from the token IDs.
        '''
        text = " ".join([self.int_to_str[i] for i in ids])
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text) # replace spaces before the specified punctuations
        return text

class SimpleTokenizerV2: # advanced version of our tokenizer including special tokens
    '''
    A second version of tokenizer which is also able to deal with unknown words.
    '''
    def __init__(self, vocab): # the class takes a vocabulary as input
        self.str_to_int = vocab # regular vocabulary (string to integer mapping)
        self.int_to_str = {i: s for s, i in vocab.items()} # inverted vocabulary (integer to string mapping)

    def encode(self, text):
        '''
        A function to break down text into tokens using regular expressions and including special tokens.
        '''
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text) # creation of tokenized version of input text
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        preprocessed = [item if item in self.str_to_int else "<|unk|>"
                        for item in preprocessed] # replace the string if it isn't in the vocabulary
        ids = [self.str_to_int[s] for s in preprocessed] # creation of token ID for each token/string
        return ids
    
    def decode(self, ids):
        '''
        A function to re-create the original text starting from the token IDs.
        '''
        text = " ".join([self.int_to_str[i] for i in ids])
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text) # replace spaces before the specified punctuations
        return text
