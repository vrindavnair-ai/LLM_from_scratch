#pip install gensim
import gensim.downloader as api
#download the model and return as object ready to use
model = api.load("word2vec-google-news-300")
#it can take any word as input and covert it to 300 dimesion vector
#assign dictionary, every word is mapped to 300 dimension vector
word__vectors = model
#Accessing the vector for the word 'computer'
print(word__vectors['computer'])
print(word__vectors['cat'].shape)
#king+women-Man should return vector similar to queen
#vector for man and king resembles mascualinity
#vector for woman and queen resembles femininity
print(word__vectors.most_similar(positive=['king','woman'],negative=['man'], topn=10))
#calculating similarity
print(word__vectors.similarity('woman','man')) #similarity score is high here
print(word__vectors.similarity('king','queen'))
print(word__vectors.similarity('uncle','aunt'))
print(word__vectors.similarity('boy','girl'))
print(word__vectors.similarity('nephew','niece'))
print(word__vectors.similarity('paper','water')) #they are not similar
#to check top 5 simiar words to "tower"
print(word__vectors.most_similar("tower", topn=5))
import numpy as np
#words to compare
word1 = 'man'
word2 = 'woman'
word3 = 'semiconductor'
word4 = 'earthworm'
word5 = 'nephew'
word6 = 'niece'
#Calculate the vector difference
vector_difference1 = model[word1] - model[word2]
vector_difference2 = model[word3] - model[word4]
vector_difference3 = model[word5] - model[word6]
#Calculate the magnitude of the vector difference
#to find norm of the difference between 2 vectors
magnitude_of_difference1 = np.linalg.norm(vector_difference1)
magnitude_of_difference2 = np.linalg.norm(vector_difference2)
magnitude_of_difference3 = np.linalg.norm(vector_difference3)
#print the magnitude of the difference
print("The magnitude of the difference between '{}' and '{}' is {:.2f}".format(word1, word2, magnitude_of_difference1))
print("The magnitude of the difference between '{}' and '{}' is {:.2f}".format(word3, word4, magnitude_of_difference2))
print("The magnitude of the difference between '{}' and '{}' is {:.2f}".format(word5, word6, magnitude_of_difference3))
