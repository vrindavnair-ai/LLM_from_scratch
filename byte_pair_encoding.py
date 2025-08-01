#Doing it from scratch is complicated
#python open source library tiktoken
#open Ai model are using this
#tiktoken to tokenize text
#pip install tiktoken
import importlib
import importlib.metadata
import tiktoken
print("tiktoken version:", importlib.metadata.version("tiktoken"))
#Instantiate BPE tokenizer from tiktoken
tokenizer = tiktoken.get_encoding("gpt2")
#it is similar to simple tokenizer version2 in tokenization.py
#GPT2 extensively use "end of text", it is part of vocabulary
#someunknownplace -> BEP will take care of this OOV
text = ("Hello, do you like tea?<|endoftext|> In the sunlit terraces""of someunknownplace")
integers= tokenizer.encode(text, allowed_special={"<|endoftext|>"})
print(integers)
#output ->[15496, 11, 466, 345, 588, 8887, 30, 50256, 554, 262, 4252, 18250, 8812, 2114, 1659, 617, 34680, 5372]
#50256 is token id of end of text
#it is the vocabulary size of tokenization scheme used in GPT2 or GPT3
#actual word count is 200000, but BPE reduced it by 1/3 rd
#Every tokenizer has an encoder and a decoder
#It works very well on completely random words as well
strings = tokenizer.decode(integers)
print(strings)
integers = tokenizer.encode("Akwirw ier")
print(integers)
strings = tokenizer.decode(integers)
print(strings)
