from sentence_transformers import SentenceTransformer
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Sentences we want to encode. Example:
sentence = ['This framework generates embeddings for each input sentence', 'Sentences are passed as a list of string.', 'The quick brown fox jumps over the lazy dog.']

# Sentences are encoded by calling model.encode()
embedding = model.encode(sentence)


#Testing script stuff 
if __name__ == '__main__':
    print(embedding)
    print(len(embedding))
    print(type(embedding))
    print(embedding[0].shape)
    print(embedding[0])
    

    #Average the vectors for each sentence
    print(embedding.mean(axis=0))

        # This works!

    #Vectors are of shape (n, 384) where n is the number of sentences in the input list