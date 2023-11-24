from transformers import BertTokenizer, BertForSequenceClassification
import csv
import pickle

# load data
def load_cola_data(path, part):
    """
    Loads the cola data downloaded from
    https://nyu-mll.github.io/CoLA/cola_public_1.1.zip
    Param:
        path (string): where the directory is located
        part (string): which data partition ("train", "test" or "dev")

    Returns:
        data (list of lists)
    """
    data = []
    filename = None
    if part == "train":
        filename = "in_domain_train.tsv"
    
    elif part == "dev":
        filename = "in_domain_dev.tsv"

    elif path == "test":
        raise NotImplementedError

    with open(path+'/raw/'+filename, 'r') as f:
        reader = csv.reader(f, delimiter='\t')
        for dp in reader:
            data.append(dp)
    print("no. datapoints in " + part + ": " + str(len(data)))
    return data


if __name__ == '__main__':
    for part in ["dev", "train"]: # ,"train", test"
        # load data
        raw_data = load_cola_data("cola_public", part)
        sents = [dp[3] for dp in raw_data]
        # initialise model
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForSequenceClassification.from_pretrained('bert-base-uncased')

        inputs = [tokenizer(sent, padding=True, truncation=True, return_tensors='pt') for sent in sents]
        print("tokenizing finished for " + part)
        outputs = [model(**input, output_hidden_states=True) for input in inputs] # shape: [num_sents, batch_size, tokens, hidden_dim]
        print("outputs finished for " + part)

        assert len(outputs) == len(sents)

        last_hidden_states = [output.hidden_states[-1] for output in outputs] 

        cls_embeddings = [last_hidden_state[0,0,:] for last_hidden_state in last_hidden_states]
        assert len(cls_embeddings) == len(sents)
        assert len(cls_embeddings[0]) == 768

        with open("cola_public/cls_embed/"+part+".pkl", 'wb') as f:
            pickle.dump(cls_embeddings, f)

        print("cls embeddings saved to cola_public/cls_embed/"+part+".pkl")
