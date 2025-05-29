'''
Notes from "Build an LLM from Scratch 2: Working with text data" of Sebastian Raschka
Link: https://www.youtube.com/watch?v=341Rb8fJxY0&list=PLTKMiZHVd_2IIEsoJrWACkIxLRdfMlw11&index=6
----------

This section covers data preparation and sampling to get input data "ready" for the LLM (see step 1 of stage 1 in figure
LLM-workflow.png). There are many forms of embeddings, here we will focus on text embeddings.

We are going to take text and tokenize it, and then convert this tokenized sub parts of the text into token IDs which we will
encode into vectors. Thus, our goal is going from letters and words into a numerical representation.

    1. What is tokenization?
    Tokenization is an important step before pre-training a large language model (LLM) and it represents the process of
    converting text into a sequence of tokens, which can be words, subwords, or characters. These tokens are the smallest units 
    of meaning in a text that can be processed by a language model. For example, the sentence “Hello, world!” can be tokenized
    into [“Hello”, “,”, “world”, “!”]. Tokenization simplifies the text and allows the model to work with manageable chunks of
    data.
    In the LLM pre-training and inference workflows, after the tokenization, each token is assigned a unique integer. For each
    integer, there is a corresponding row in a lookup table, which is the vector representation of that token. This vector is
    then used as input for a particular token in the language model.

    2. Why do we need tokenization?
    Suppose you want to pre-train your own language model, and you have 10 GB of language data available. However, the problem
    is that your data is in textual form. Language models, or any machine learning models, process numbers. Therefore, you need
    a method to convert your text into numbers. This is where tokenization comes into play!
    To use the language model on our dataset, we’ll first tokenize the text. Then we’ll assign each token a unique integer. These
    integers are then vectorized and fed into the language model. The language model processes this vectorized input and outputs
    integers again, which are subsequently converted back into text. Without tokenization, language models would not be able to
    operate on just raw textual data.

    3. TODO
    
    Link: https://medium.com/thedeephub/all-you-need-to-know-about-tokenization-in-llms-7a801302cf54
'''

import os # needed for working with files
import urllib.request # needed for downloading stuffs from the internet
import re # regular expressions
from tokenizer import SimpleTokenizerV1, SimpleTokenizerV2, GPTDatasetV1, create_dataloader_v1
import tiktoken # we will use the byte pair algorithm (BPE) implemented by OpenAI's tiktoken API
import torch

"""
----------
1) Tokenizing text starting from an input text (step 1 of text-embedding-workflow.png).
----------
"""

if not os.path.exists("the-verdict.txt"):
    url = ("https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/refs/heads/main/ch02/01_main-chapter-code/the-verdict.txt")
    file_path = "the-verdict.txt"
    urllib.request.urlretrieve(url, file_path)

with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()
print("---")
print(len(raw_text))

text = "Hello, world. This, is a test."
result = re.split(r'(\s)', text) # it splits text into individual tokens whenever a whitespace is found
print("---")
print(result)

text = "Hello, world. Is this-- a test?"
result = re.split(r'([,.]|\s)', text) # it splits text whenever it finds whitspaces/punctuation
print("---")
print(result)
result = [item for item in result if item.strip()] # removing blank tokens (the "strip" method removes spaces in the string)
print("---")
print(result) # this was the simplest way of tokenizing text

preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text) # using regular expression on our example
preprocessed = [item for item in preprocessed if item.strip()]
print("---")
print(preprocessed[:30])
print(len(preprocessed)) # we have 4690 tokens!

"""
----------
2) Converting tokens into token IDs (step 2 of text-embedding-workflow.png). The first step is to build a vocabulary, namely a
unique mapping between each word that can occur and a unique integer. Once that we have a vocabulary, we can convert tokens
into token IDs.
----------
"""

all_words = sorted(set(preprocessed)) # we can get rid of duplicates in our tokens
vocab_size = len(all_words)
print("---")
print(all_words[:30])
print(vocab_size)

vocab = {token: integer for integer, token in enumerate(all_words)}
print("---")
print(vocab)

tokenizer = SimpleTokenizerV1(vocab) # we have already created the vocabulary
text = """"It's the last he painted, you know," Mrs. Gisburn said with pardonable pride.""" # this is an extract from the-verdict.txt
ids = tokenizer.encode(text) # these are the token IDs of the above text
print("---")
print(ids)
print(tokenizer.decode(ids))

"""
----------
3) Adding special context tokens (we want to extend our vocabulary with special tokens...). Some of these special tokens could be
useful for unknown words (namely words that are not in the vocabulary) or for the end of text.
----------
"""

text = "Hello, do you like tea. is this-- a text?"
# tokenizer.encode(text) # it gives a KeyError since the word "Hello" is not included in our vocabulary

all_tokens = sorted(list(set(preprocessed)))
all_tokens.extend(["<|endoftext|>", "<|unk|>"])
vocab = {token: integer for integer, token in enumerate(all_tokens)} # we re-generate the vocabulary as before
print("---")
print(vocab)
tokenizer = SimpleTokenizerV2(vocab) # we use the new tokenizer with the new vocabulary
ids = tokenizer.encode(text)
print("---")
print(ids)
print(tokenizer.decode(ids)) # here we won't see the unknown words

"""
----------
4) Using Byte Pair Enconding algorithm to handle unknown tokens. Byte Pair Enconding (BPE) is an algorithm used by GPT-1, GPT-2, etc.
tokenizers to deal with unknown tokens. For example, tiktokenizer (https://tiktokenizer.vercel.app/?model=gpt2) splits the unknown
word in two or more (known) subwords based on the training data used for the particular model (GPT-1, GPT-2, etc.).
Check this link https://github.com/openai/gpt-2/blob/master/src/encoder.py to see how GPT-2 has implemented the BPE algorithm.
----------
"""

tokenizer = tiktoken.get_encoding("gpt2") # we instantiate a new tokenizer with its full vocabulary, the way it was trained, etc.
print("---")
print(tokenizer.encode("Hello world"))
print(tokenizer.decode(tokenizer.encode("Hello world")))

text = ("Hello, do you like tea? <|endoftext|> In the sunlit terraces of someunknownPlace "
        "jsahsjhjshSAK<SJAKshA") # the <|endoftext|> special token is used to signal to the LLM where a document ends and another starts
print("---")
print(tokenizer.encode(text, allowed_special={"<|endoftext|>"}))
print(tokenizer.decode(tokenizer.encode(text, allowed_special={"<|endoftext|>"})))

"""
----------
5) Data sampling with a sliding window. How do we provide smaller chunks of these token IDs to the LLM to train it efficiently?
For example, GPT-2 uses a chunk size of 1024, while GPT-4 of 8,192! Here it will be used a chunk size of 4 for demonstration purpose.
----------
"""

with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

enc_text = tokenizer.encode(raw_text) # tokenize our training text using tiktoken's tokenizer
print("---")
print(len(enc_text)) # how to provide these 5145 token IDs efficently (as sub-chunks) to our LLM?

enc_sample = enc_text[50:] # we truncate the first fifty characters
print("---")
print(len(enc_sample))
context_size = 4 # chunk size: this is the size of our input (each input will contain four tokens)
x = enc_sample[:context_size]
y = enc_sample[1:context_size]
print("---")
print(f"x: {x}")
print(f"y:      {y}")

for i in range(1, context_size+1): # i = 1, 2, 3, 4
    context = enc_sample[:i] # what we give as input to the LLM
    desired = enc_sample[i] # what we except as output from the LLM
    print("---")
    print(f"TOKENS: {context} ----> {desired}")
    print(f"WORDS: {tokenizer.decode(context)} ----> {tokenizer.decode([desired])}") # "decode" method takes a list as input

# using "Dataset" and "DataLoader" classes from PyTorch ("torch") library to create our chunks of data (each made of four tokens)
with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()
print("---")
print(raw_text)

dataloader = create_dataloader_v1(raw_text, batch_size = 4, max_length = 256, stride = 1, shuffle = False)