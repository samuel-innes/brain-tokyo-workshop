from datasets import load_dataset, load_metric
from transformers import BertTokenizer, BertForSequenceClassification, TrainingArguments, Trainer
import numpy as np


task = "cola"
model_checkpoint = "bert-base-uncased"
batch_size = 16
num_epochs = 3


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

model = BertForSequenceClassification.from_pretrained(
    model_checkpoint, num_labels=2, id2label=id2label, label2id=label2id
)

metric_name = "matthews_correlation"
model_name = model_checkpoint.split("/")[-1]

args = TrainingArguments(
    f"{model_name}-finetuned-{task}",
    evaluation_strategy = "epoch",
    save_strategy = "epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=num_epochs,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model=metric_name
)


def compute_metrics(eval_predictions):
    predictions, labels = eval_predictions
    predictions = np.argmax(predictions, axis=1)
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

eval = trainer.evaluate()

print(eval)