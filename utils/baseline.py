from datasets import load_dataset, load_metric
from transformers import BertTokenizer, BertForSequenceClassification, TrainingArguments, Trainer, TrainerCallback
from copy import deepcopy
import numpy as np
import torch.nn as nn
from copy import deepcopy

task = "cola"
model_checkpoint = "bert-base-uncased"
batch_size = 32
num_epochs = 10

# --- model params ----
freeze_bert = False
dense_layer = False
# ---------------------
print("Model parameters:")
print("BERT frozen?", freeze_bert)
print("MLP?", dense_layer)

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
        self.relu = nn.ReLU()
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, input_ids, attention_mask, token_type_ids, labels=None):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            labels=labels,
            output_hidden_states=True
        )
        last_hidden_state = outputs.hidden_states[-1]
        linear1_output = self.relu(self.linear1(last_hidden_state[:, 0, :]))

        logits = self.linear2(linear1_output)

        if labels is not None:
            return self.loss_fn(logits, labels), logits
        else:
            return logits

if dense_layer:
    model = CustomBERTModel()
else: 
    model = BertForSequenceClassification.from_pretrained(
            model_checkpoint, num_labels=2, id2label=id2label, label2id=label2id
        )

if freeze_bert:
    for name, param in model.bert.named_parameters():
        if 'classifier' not in name:  # classifier layer
            param.requires_grad = False

metric_name = "matthews_correlation"
model_name = model_checkpoint.split("/")[-1]

class CustomCallback(TrainerCallback):
    def __init__(self, trainer) -> None:
        super().__init__()
        self._trainer = trainer
    
    def on_epoch_end(self, args, state, control, **kwargs):
        if control.should_evaluate:
            control_copy = deepcopy(control)
            self._trainer.evaluate(eval_dataset=self._trainer.train_dataset, metric_key_prefix="train")
            return control_copy

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

class CustomCallback(TrainerCallback):
    def __init__(self, trainer) -> None:
        super().__init__()
        self._trainer = trainer
    
    def on_epoch_end(self, args, state, control, **kwargs):
        if control.should_evaluate:
            control_copy = deepcopy(control)
            self._trainer.evaluate(eval_dataset=self._trainer.train_dataset, metric_key_prefix="train")
            return control_copy

trainer = Trainer(
    model,
    args,
    train_dataset=encoded_dataset["train"],
    eval_dataset=encoded_dataset["validation"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)
trainer.add_callback(CustomCallback(trainer))
print("Start training")

trainer.train()

print("Start evaluation")

eval_result = trainer.evaluate()

print(eval_result)
