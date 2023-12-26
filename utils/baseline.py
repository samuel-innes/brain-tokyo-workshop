from datasets import load_dataset, load_metric
from transformers import BertTokenizer, BertForSequenceClassification, TrainingArguments, Trainer
import numpy as np
import torch.nn as nn
import torch

task = "cola"
model_checkpoint = "bert-base-uncased"
batch_size = 16
num_epochs = 5

# --- model params ----
freeze_bert = False
dense_layer = True
# ---------------------

dataset = load_dataset("glue", task)
metric = load_metric('glue', task)

tokenizer = BertTokenizer.from_pretrained(model_checkpoint)

def preprocess_function(examples):
    return tokenizer(examples["sentence"], truncation=True)

pre_tokenizer_columns = set(dataset["train"].features)
encoded_dataset = dataset.map(preprocess_function, batched=True)
tokenizer_columns = list(set(encoded_dataset["train"].features) - pre_tokenizer_columns)
print("Columns added by tokenizer:", tokenizer_columns)

id2label = {0: "Invalid", 1: "Valid"}
label2id = {val: key for key, val in id2label.items()}

class CustomBERTModel(nn.Module):
    def __init__(self, num_labels=2):
        super(CustomBERTModel, self).__init__()
        self.bert = BertForSequenceClassification.from_pretrained(
            model_checkpoint, num_labels=num_labels, id2label=id2label, label2id=label2id
        )
        self.linear1 = nn.Linear(768, 256)  # Assuming BERT hidden size is 768
        self.linear2 = nn.Linear(256, num_labels)

    def forward(self, input_ids, attention_mask, token_type_ids, labels=None):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            labels=labels,
            output_hidden_states=True
        )
        last_hidden_state = outputs.hidden_states[-1]
        linear1_output = self.linear1(last_hidden_state)
        linear1_output = self.linear1(last_hidden_state[:, 0, :])

        logits = self.linear2(linear1_output)

        if labels is not None:
            loss = outputs.loss
            return loss, logits
        else:
            return logits

model = CustomBERTModel()

if freeze_bert:
    for name, param in model.bert.named_parameters():
        if 'classifier' not in name:  # classifier layer
            param.requires_grad = False

metric_name = "matthews_correlation"
model_name = model_checkpoint.split("/")[-1]

args = TrainingArguments(
    f"{model_name}-finetuned-{task}",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=num_epochs,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model=metric_name
)

def compute_metrics(eval_predictions):
    logits, labels = eval_predictions
    predictions = np.argmax(logits, axis=1)
    return metric.compute(predictions=predictions, references=labels)

validation_key = "validation"
trainer = Trainer(
    model,
    args,
    train_dataset=encoded_dataset["train"],
    eval_dataset=encoded_dataset[validation_key],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

print("Start training")

trainer.train()

print("Start evaluation")

eval_result = trainer.evaluate()

print(eval_result)
