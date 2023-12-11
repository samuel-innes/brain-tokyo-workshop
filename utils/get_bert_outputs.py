from transformers import BertTokenizer, BertModel
import datasets
import pickle


task = "cola"
model_checkpoint = "bert-base-uncased"

dataset = datasets.load_dataset('glue', task)

tokenizer = BertTokenizer.from_pretrained(model_checkpoint)
model = BertModel.from_pretrained(model_checkpoint)

for part in ["train", "validation", "test"]:
    inputs = [tokenizer(sentence, padding=True, truncation=True, return_tensors='pt') for sentence in dataset[part]["sentence"]]
    print("tokenizing finished for " + part)
    
    outputs = [model(**input) for input in inputs] # shape: [num_sents, batch_size, tokens, hidden_dim]
    print("outputs finished for " + part)
    
    last_hidden_states = [output.last_hidden_state for output in outputs]

    assert len(last_hidden_states) == len(dataset[part]["sentence"])
    
    # We only work with the embeddings generated for the CLS token
    cls_embeddings = [last_hidden_state[0,0,:] for last_hidden_state in last_hidden_states]
    
    assert len(cls_embeddings) == len(dataset[part]["sentence"])
    assert len(cls_embeddings[0]) == 768

    with open("cola_embed/"+part+".pkl", 'wb') as f:
        pickle.dump(cls_embeddings, f)

    print("cls embeddings saved to cola_embed/"+part+".pkl")
