import torch
#let us assume token ids associated with words as -> 2,3,5,1
input_ids = torch.tensor([2,3,5,1])
#quick(token_id=4) for(0) is(3) in(2) the(5) house(1)
#for simplicity use small vocabulary of 6 tokens. (50257 is vocabulary used in GPT-2)
#assume vector dimension as 3
vocab_size = 6
output_dim = 3
torch.manual_seed(123)
#create an embeding layer
#this initialize weight of embedding vector is a random manner
embedding_layer = torch.nn.Embedding(vocab_size, output_dim) 
#simple dictionary is created
#to get all the weights which are initialized (6*3) -> 6 rows and 3 columns
print(embedding_layer.weight)

#it consists of small random numbers. these values are optimized during LLM training
#vector for id number 3, it is the row corresponding to that id.[1st id is 0th row, so 3rd id is row 4]
print(embedding_layer(torch.tensor([3])))
#embedding layer is look-up table
print(embedding_layer(torch.tensor(input_ids)))
#it looks at particular row based on IDs