#Data preparation and sampling - tokenization
"""Step 1: Tokenization"""
#Download the book from "https://github.com/rasbt/LLMs-from-scratch/blob/main/ch02/01_main-chapter-code/the-verdict.txt"
with open("the-verdict.txt","r", encoding = "utf-8") as f:
    raw_text = f.read()

print("Total number of characters:",len(raw_text))
print(raw_text[:99])
#outpt is -> """Total number of characters: 20479
#"""I HAD always thought Jack Gisburn rather a cheap genius--though a good fellow enough--so it was no """
#Our goal is to tokenize 20479 characters in the file
#millions of articles in common cases
#Regular expression library to split any given text based on white spaces or any other characters

import re
text = "Hello, World. This is a test"
#\s means -> split whereever white spaces are encountered
result = re.split(r'(\s)',text)
print(result)
#output is -> "['Hello,', ' ', 'World.', ' ', 'This', ' ', 'is', ' ', 'a', ' ', 'test']"
#split commas and periods. In re include "," and "."
result = re.split(r'([,.]|\s)',text)
print(result)
#output -> "['Hello', ',', '', ' ', 'World', '.', '', ' ', 'This', ' ', 'is', ' ', 'a', ' ', 'test']"
#We need to remove white space 
#store the result nly if item.strio is true
#removing whitespace helps in improving the memory
#but keeping white spaces helps in ensuring structure of text (Eg: in case pf python code)
#Don't remove white spaces in case if it is sensitive to white space
result = [item.strip() for item in result if item.strip()]
print(result)
#output - >['Hello', ',', 'World', '.', 'This', 'is', 'a', 'test']
#we want ?, --, etc are to be seprate token. So add those in re.split
text = "Hello, World. This is -- a test?"
result = re.split(r'([,.:;?_!"()\']|--|\s)',text)
result = [item.strip() for item in result if item.strip()]
print(result)
#o/p -> ['Hello', ',', 'World', '.', 'This', 'is', '--', 'a', 'test', '?']
#Apply the statement in our raw_text
preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)',raw_text)
#final list containing list of tokens
preprocessed = [item.strip() for item in preprocessed if item.strip()]
print(preprocessed[:30])
print(len(preprocessed))

""" Step 2 : convert tokens to token_IDs"""
#make vocabulary, with unique tokens and token ids
#sort the tokens and map it to unique integer
all_words = sorted(set(preprocessed))
vocab_size = len(all_words)
#vocab_size include only unique words
print(vocab_size)
#assigning integer for each word in the token
#enumerare will assign integer to each token in "all_words"
vocab = {token:integer for integer,token in enumerate(all_words)}
for i, item in enumerate(vocab.items()):
    print(item)
    if i==50:
        break
#Think of it as encoding : mapping token to token_id
#Decoder will give the token from token id 
#for tha We need an inverse of vocabulary

class SimpleTokenizerV1:
    #init method is called by default when the instance of the class is created
    #it takes vocabulary(the mapping from toke to toekn ID-> pass it when we create instance)
    def __init__(self,vocab):
        self.str_to_int = vocab
        #create reverse vocabulary
        #to convert back to token
        self.int_to_str = {i:s for s,i in vocab.items()}

    #token to token id
    def encode(self,text):
        #split text to token
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)',text)
        #to remove white soace
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        #convert to token ids
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids

    #token id to token
    def decode(self,ids):
        #use the reverse dictionary to convert to token from token id
        #join the tokens together
        text = " ".join([self.int_to_str[i] for i in ids])
        #Replace spaces before specific punctuations
        #the for chased . => the for chased. (To get rid of the space before ".")
        text = re.sub(r'\s+([,.?!"()\'])',r'\1',text)
        return text
    
#Create an instance of class
tokenizer = SimpleTokenizerV1(vocab)
#Sentence from tarining set
text = """"It's the last he painted, you know,"Mrs. Gisburn said with pardonable pride."""
ids = tokenizer.encode(text)
print(ids)
#convert token id back to original text
text = tokenizer.decode(ids)
print(text)

""" #text which is not in training dataset
text = "Hello, do you like tea?"
print(tokenizer.encode(text))
#it will throw key error as "Hello" is not in training dataset and hence vocabulary"""
#WE need to train it on huge dataset
#add special context token to deal with it incase of LLM
#Dealing with special context token 
#unknown token and end of text token
#Extended the vocabulary with "unknown token" and "end of text" token
all_tokens = sorted(list(set(preprocessed)))
all_tokens.extend(["<|endoftext|>","<|unk|>"])
vocab = {token:integer for integer,token in enumerate(all_tokens)}
print(len(vocab.items()))
#the new vocabulary size is 1132. reviously it was 1130
#print last 5 tokens with ids
for i,item in enumerate(list(vocab.items())[-5:]):
    print(item)

#Tokenizer class version 2
class SimpleTokenizerV2:
    #init method is called by default when the instance of the class is created
    #it takes vocabulary(the mapping from toke to toekn ID-> pass it when we create instance)
    def __init__(self,vocab):
        self.str_to_int = vocab
        #create reverse vocabulary
        #to convert back to token
        self.int_to_str = {i:s for s,i in vocab.items()}

    #token to token id
    def encode(self,text):
        #split text to token
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)',text)
        #to remove white soace
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        #convert to token ids
        #replace the tokens which are not present in vocabulary with '<|unk>'
        preprocessed = [item if item in self.str_to_int 
                        else "<|unk|>" for item in preprocessed]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids

    #token id to token
    #no change in decoder block
    def decode(self,ids):
        #use the reverse dictionary to convert to token from token id
        #join the tokens together
        text = " ".join([self.int_to_str[i] for i in ids])
        #Replace spaces before specific punctuations
        #the for chased . => the for chased. (To get rid of the space before ".")
        text = re.sub(r'\s+([,.?!"()\'])',r'\1',text)
        return text
    
tokenizer = SimpleTokenizerV2(vocab)
text1 = "Hello, do you like tea?"
text2 = "In the sunlit terraces of the palace."
text = " <|endoftext|> ".join((text1,text2))
print(text)
ids = tokenizer.encode(text)
print(ids)
#token id for hello is 1131 and for end of text "1132"
#o/p - > Hello, do you like tea? <|endoftext|> In the sunlit terraces of the palace.
#[1131, 5, 355, 1126, 628, 975, 10, 1130, 55, 988, 956, 984, 722, 988, 1131, 7]
print(tokenizer.decode(ids))
#We will get 2 unknown tokens instead of "hello" and "palace". as these words are not in training dataset
