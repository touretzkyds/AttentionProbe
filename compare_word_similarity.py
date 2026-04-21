#load imports
print("starting")
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from torch.nn.functional import cosine_similarity
print("imports done")

model_name = "google/flan-t5-large"
tokenizer = T5Tokenizer.from_pretrained(model_name)
print("Tokenizer loaded")
model = T5ForConditionalGeneration.from_pretrained(model_name)
print("Model loaded")

embedding_matrix = model.shared.weight.detach()
print("Matrix detached", embedding_matrix.shape)

#looks up the embeddings for two words and computes the cosine similarity 
def similarity_between(word1, word2):

    #tokenize words given
    tok1 = tokenizer.tokenize(word1)
    tok2 = tokenizer.tokenize(word2)

    #converting tokens to the ids found in the dictionary
    id1 = tokenizer.convert_tokens_to_ids(tok1)[0]
    id2 = tokenizer.convert_tokens_to_ids(tok2)[0]

    #get the embedding vectors
    vec1 = embedding_matrix[id1]
    vec2 = embedding_matrix[id2]
    print("vec1 shape", vec1.shape)

    # cosine similarity
    cos = cosine_similarity(vec1.unsqueeze(0), vec2.unsqueeze(0)).item()

    return cos, tok1[0], tok2[0]


#running the script
if __name__ == "__main__":

    while True:
        w1 = input("Enter the first word: ").strip()
        w2 = input("Enter the second word: ").strip()

        #gets the cosine similarity from the function, the first token, and the second token
        cos, tok1, tok2 = similarity_between(w1, w2)

        print(f"tokenized as: '{tok1}'  and  '{tok2}'")
        print(f"cosine similarity:  {cos:.4f}")
        print()








#dot = torch.dot(v1, v2).item()