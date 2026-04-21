import torch  
from transformers import T5Tokenizer, T5ForConditionalGeneration
from torch.nn.functional import cosine_similarity

model_name = "google/flan-t5-large"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

embedding_matrix = model.shared.weight.detach()

def get_leastsimilar_tokens(word, top_k = 10):
    tokens = tokenizer.tokenize(word)
    id = tokenizer.convert_tokens_to_ids(tokens)[0]# tokenizing and then convert to ids

    vec = embedding_matrix[torch.tensor(id)]# finding where each id is

    #with torch.no_grad(): ---- this is raw dot product
        #sims = embedding_matrix @ q  # [32128]
        #top_vals, top_ids = torch.topk(sims, top_k + 1)
    
    #just telling pytorch to compute the numbers and don't remember how u got them
    with torch.no_grad(): #this is normalizing before computing dot product
        #changes each embedding to have the same magnititude scale
        sims = cosine_similarity(vec, embedding_matrix)

        #finds the tokens with the highest scores
        top_vals, top_ids = torch.topk(sims, top_k + 1, largest = False)
        #if we were to switch this to largest true


        results = []
        q_first_id = id
        for val, tid in zip(top_vals.tolist(), top_ids.tolist()):
            #if tid == q_first_id:
                #continue  # skip the id that is the same word you inputted
            tok = tokenizer.convert_ids_to_tokens([tid])[0]
            results.append((tok, val))
        return results

if __name__ == "__main__":
    query_word = input("Enter a word to find its least similar tokens: ").strip()
    print(f"top least tokens to '{query_word}'")
    for token, score in get_leastsimilar_tokens(query_word):
        print(f"{token:15s} dot = {score:.4f}")