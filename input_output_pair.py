import importlib
import importlib.metadata
import tiktoken

#Instantiate BPE tokenizer from tiktoken
tokenizer = tiktoken.get_encoding("gpt2")
with open("the-verdict.txt","r", encoding = "utf-8") as f:
    raw_text = f.read()
#Tokenizing the dataset
enc_text = tokenizer.encode(raw_text)
print(len(enc_text))
#output is 5145 - > total number of tokens intraining set is 5145 
#tokens can be subwords or charcter
#Remove first 50 tokens
enc_sample = enc_text[50:]
#Create input and output target pairs X and Y
#x =[1,2,3,4] y =[2,3,4,5]
#when x is '1' , y is 2
#when [1,2,3] are input , '4' will be output
#Context size determines how many tokens are included in the input
context_size =4 #input size can be maximum 4
#how many words
x = enc_sample[:context_size]
y = enc_sample[1:context_size+1]
print(f"x: {x}")
print(f"y:      {y}")
#output -> x: [290, 4920, 2241, 287]
#y:      [4920, 2241, 287, 257].  #for examples if x is [290,4920] then y is 2241
#targets are input shifted by one position
for i in range (1, context_size+1):
    context = enc_sample[:i] #when i =1 , context will be just 290
    desired = enc_sample[i] # when i= 1 , desired is 4920
    print(context,"------>",desired)
    #to check the real text input and output
    print(tokenizer.decode(context), "----->",tokenizer.decode([desired]))

"""output -->
[290] ------> 4920
[290, 4920] ------> 2241
[290, 4920, 2241] ------> 287
[290, 4920, 2241, 287] ------> 257"""
#we need to make pytorch input and output tensors
#Dataset and data loader classes
#Implementing a data loader
#tokenize text
#use sliding window to chunk the book into overlapping sequences
#return total rows
#return single row from dataset
#/usr/bin/python3 -m pip install torch
#importing dataset and dataloader from torch.util.data
from torch.utils.data import Dataset, DataLoader
class GPTDatasetV1(Dataset):
    #txt is the text file we have
    #specify tokenize )eg: bytepair)
    #max length is the context size
    #stride determines how much we slide for input (eg : 4, first 4 will be the first input row(0,1,2,3), second 4 (ie.,4,5,6,7) will be second row of input)
    def __init__(self,txt,tokenizer, max_length, stride):
        #to store inout
        self.input_ids = []
        #to store output id
        self.target_ids = []

        #tokenize the entire text
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext>"})

        #Use the sliding window to chunk the book into overlapping sequences of max_length
        for i in range(0, len(token_ids)-max_length, stride):
            #"-max_length" because last one will include context size, we don't want to spil over the dataset
            #when i =1 -> first row of tensor is updated
            #when i = 0, token_id will be from 0 to 4
            input_chunk = token_ids[i:i+max_length]
            #when i=0 , token_id will be 1 to 5
            target_chunk = token_ids[i+1:i+max_length+1]
            #append this to input id tensor
            self.input_ids.append(torch.tensor(input_chunk))
            #append this to output id tensor
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self,idx):
        #To return the particular row based on the input provided
        #if idx = 0, it will return first row of input tensor and first row of output tensor
        #Data loader will look at this function
        #it will tell data sloader what type of data we need
        return self.input_ids[idx], self.target_ids[idx]

#Load the inputs in batches via a pythons DataLoader

def create_dataloader_v1 (txt, batch_size = 4, max_length=256, stride =128, shuffle=True, drop_last=True, num_workers=0):
    #help create input output data pair
    #txt while is the dataset we have
    #batch_size = how many batches or how many cpu processes we want to run parallely(by default is 4(based on number of cores we can use 4 or 8))
    #max_length is context_length(model can look at 256 words at a time and predict the next word)
    #stride = -> how much we need to skip before creating next batch
    #number of workers is number of CPU threads which can run simultaneously
    #step 1 : initilize tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")
    #step2: Create dataset, by creating instance of GPTdatasetv1
    dataset = GPTDatasetV1(txt,tokenizer,max_length, stride)
    #step 3: Create DataLoader

    dataloader = DataLoader(
        #this dataloader will check getitem method and return input -output pair as given in that method
        #create input -output tensor
        #Data loader will help in parallel processing
        dataset,
        batch_size=batch_size,
        #the number of batches the model processes at once before updating the parameters
        #after analysing 4 batches model will update the parameters
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
        #for parallel processing on different thread os CPU
    )
    return dataloader

import torch

print("Pytorch version: ", torch.__version__)
#convert dataloader into a pythoniterator to fetch the next entry via python's built in next() function
dataloader = create_dataloader_v1(raw_text, batch_size=1, max_length=4, stride=1, shuffle=False)
data_iter = iter(dataloader)
first_batch = next(data_iter)
print(first_batch)
#output->[tensor([[  40,  367, 2885, 1464]]), tensor([[ 367, 2885, 1464, 1807]])]
# input tensor is ([[  40,  367, 2885, 1464]]),and output tensor is([[ 367, 2885, 1464, 1807]])]
second_batch = next(data_iter)
print(second_batch)
#[tensor([[ 367, 2885, 1464, 1807]]), tensor([[2885, 1464, 1807, 3619]])]
#the input is just shifted by one token as we kept stride =1
#fiest tensor stores input token id and second tensor store output token_id
#since the max_length is set to 4, each of the two tensors contains 4 token IDS
#More stride means we move over data in faster manner and hence lesser computations
#it is common to train llm with max_length of 256
#small batch size requires less memory during training but leads to more noisy updates(parameter update will be very quick)
#large batch size - > the model will go through entire dataset before updating parameter
#It should be selected by experiment
dataloader = create_dataloader_v1(raw_text, batch_size=8, max_length=4, stride=4, shuffle=False)
#since stride =4 there is no overlap between rows in input tensor
#input tensor has 8 rows since batch size = 8
#model will process this batch before making parameter update
#more overlap increase more overfitting, reduce overlap to prevent overfitting
data_iter = iter(dataloader)
inputs, targets = next(data_iter)
print("Input : \n", inputs)
print("\nTarget : \n",targets)

