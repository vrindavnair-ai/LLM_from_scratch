#/realistic embedding size = 256
#in GPT embedding size is 12288 dimension
#gpt2 vocab size is 50257
import torch
import tiktoken
from torch.utils.data import Dataset, DataLoader

vovab_size = 50257
output_dim = 256
#create an embedding layer
tokenizer = tiktoken.get_encoding("gpt2")
token_embedding_layer = torch.nn.Embedding(vovab_size, output_dim)
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
#intialize input
with open("the-verdict.txt","r", encoding = "utf-8") as f:
    raw_text = f.read()
#Tokenizing the dataset
max_length = 4
#data loader just helps us to manage task of inouting and batching data
dataloader = create_dataloader_v1(raw_text, batch_size=8, max_length=max_length, stride=max_length, shuffle=False)
date_iter = iter(dataloader)
inputs, targets = next(date_iter)
print("Token IDs :\n", inputs)
print("\nInputs shape:\n",inputs.shape)
# 8 input sequences with 4 tokens
"""input of batch
Token IDs :
 tensor([[   40,   367,  2885,  1464],
        [ 1807,  3619,   402,   271],
        [10899,  2138,   257,  7026],
        [15632,   438,  2016,   257],
        [  922,  5891,  1576,   438],
        [  568,   340,   373,   645],
        [ 1049,  5975,   284,   502],
        [  284,  3285,   326,    11]])

Inputs shape:
 torch.Size([8, 4])"""
#Each input tokens need to be converted to 256 dimensional vectors usning embedding layer
#embedding layer to embed token Ids to vectors
token_embedding = token_embedding_layer(inputs)
print(token_embedding.shape)
#tensor with size, torch.Size([8, 4, 256])
#each time only 4 tokens are need to be processed because context_size is 4
#number of rows will be context length and columns will be vector dimension ie, 256
#create positional embedding layer
context_length = max_length #Length of input token
#number of row is output dimension
pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)
#it will create 0,1,2,3 sequence-> it will generate 4 vectors each of size 256
pos_embedding = pos_embedding_layer(torch.arange(max_length))
print(pos_embedding.shape)
#torch.Size([4, 256]) - > 4 vectors, one is for position number 1, second is for position number 2, etc.
#we can add this directly to token embedding