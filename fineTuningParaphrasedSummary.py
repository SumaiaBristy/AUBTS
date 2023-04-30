import json
import random

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments
from datasets import concatenate_datasets, DatasetDict
import evaluate
import numpy as np
import nltk
from nltk.tokenize import sent_tokenize
import pandas as pd
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
import sentTokenizer
from typing import List
# spanish_dataset = load_dataset("amazon_reviews_multi", "es")
# english_dataset = load_dataset("amazon_reviews_multi", "en")
from transformers import AutoModelForSeq2SeqLM
from normalizer import normalize  # pip install git+https://github.com/csebuetnlp/normalizer
from datasets import load_dataset, Dataset, DatasetDict
from sklearn.model_selection import train_test_split


def generateParaphrase(input_summary_list):
    modelBanParag = AutoModelForSeq2SeqLM.from_pretrained("csebuetnlp/banglat5_banglaparaphrase")
    tokenizer = AutoTokenizer.from_pretrained("csebuetnlp/banglat5_banglaparaphrase", use_fast=True)
    input_summary = ''.join(input_summary_list)
    input_ids = tokenizer(normalize(input_summary), return_tensors="pt").input_ids
    generated_tokens = modelBanParag.generate(input_ids)
    decoded_tokens = tokenizer.batch_decode(generated_tokens)[0]
    # sentences = input_summary.split("।")
    paraphrased_text = ""
    for sentence in input_summary_list:
        sentence = sentence.strip()
        if len(sentence.strip()) > 0:
            input_ids = tokenizer.encode(sentence, return_tensors='pt')
            outputs = modelBanParag.generate(input_ids=input_ids, max_length=128, do_sample=True)
            paraphrase = tokenizer.decode(outputs[0], skip_special_tokens=True)
            paraphrased_text += paraphrase + "।"
    paraphrased_text = paraphrased_text.strip()
    return paraphrased_text


Bangla_Dataset = DatasetDict()
Bangla_Dataset.set_format("pandas")
data_instances = []
it=1;
for i in range(1, 101):
    serial_no = str(i)
    #### for NCTB data
    source = open('../../Dataset/NCTB/Source/' + serial_no + '.txt', encoding='utf-8').read()
    summary = open('../../Dataset/NCTB/Summary/' + serial_no + '.txt', encoding='utf-8').read()

    ### for BNLPC data
    # source = open('../../Dataset/BNLPC/Dataset1/Documents/Document_' + serial_no +'_Summary_'+str(it)+ '.txt', encoding='utf-8').read()
    # summary = open('../../Dataset/BNLPC/Dataset1/Summaries/Document_'+ serial_no +'_Summary_'+str(it)+ '.txt', encoding='utf-8').read()

    docSource = sentTokenizer.sentTokenizing().sentTokenize(source)
    docSummary = sentTokenizer.sentTokenizing().sentTokenize(summary)
    # convert list to dataset dictionaryr
    # my_dataset = Dataset.from_dict({"my_column": docSource}) # list to dataset
    # source_dataset_dict = DatasetDict({"source": my_dataset}) #dataset object to datasetdict
    # Bangla_Dataset[("source", tuple(docSource))] = [("summary", tuple(docSummary))]
    # Bangla_Dataset[(tuple(docSource))] = [(tuple(docSummary))]
    paraphrased_summary_string = generateParaphrase(docSummary)
    source_string = ''.join(docSource)
    data_instance = {'source': source_string, 'summary': paraphrased_summary_string}
    data_instances.append(data_instance)
data_dict = {'source': [], 'summary': []}
for instance in data_instances:
    data_dict['source'].append(instance['source'])
    data_dict['summary'].append(instance['summary'])
summarization_dataset = Dataset.from_dict(data_dict)
# print(summarization_dataset)
# print(summarization_dataset[3])
empty_dict = {'source': [], 'summary': []}
empty_dataset = Dataset.from_dict(empty_dict)
train_val_dataset = empty_dataset
test_dataset = empty_dataset
valid_dataset = empty_dataset
train_dataset = empty_dataset
# train_val_dataset, test_dataset = summarization_dataset.train_test_split(test_size=0.2)
# train_dataset, valid_dataset = train_val_dataset.train_test_split(test_size=0.25)
# train_val_dataset, test_dataset = train_test_split(summarization_dataset, test_size=0.2, random_state=42)
# train_dataset, valid_dataset = train_test_split(train_val_dataset, test_size=0.25, random_state=42)

# manually split dataset
# Shuffle the dataset randomly
random.seed(42)
shuffled_dataset = summarization_dataset.shuffle()

# Calculate the sizes of the train, validation, and test sets
train_size = int(0.7 * len(shuffled_dataset))
val_size = int(0.2 * len(shuffled_dataset))
test_size = len(shuffled_dataset) - train_size - val_size

# Split the dataset into train, validation, and test sets
train_dataset = shuffled_dataset.select(range(train_size))
val_dataset = shuffled_dataset.select(range(train_size, train_size + val_size))
test_dataset = shuffled_dataset.select(range(train_size + val_size, len(shuffled_dataset)))

# Print the sizes of the train, validation, and test sets
print(f"Train dataset size: {len(train_dataset)}")
print(f"Validation dataset size: {len(val_dataset)}")
print(f"Test dataset size: {len(test_dataset)}")

model_checkpoint = "google/mt5-small"
model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)
max_input_length = 512
max_target_length = 512


def preprocess_function(examples):
    model_inputs = tokenizer(
        examples["source"],
        max_length=max_input_length,
        truncation=True,
        padding=True
    )
    labels = tokenizer(
        examples["summary"], max_length=max_target_length, truncation=True
    )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


encoded_train_dataset = train_dataset.map(preprocess_function, batched=True)
encoded_val_dataset = val_dataset.map(preprocess_function, batched=True)
encoded_test_dataset = test_dataset.map(preprocess_function, batched=True)
rouge_score = evaluate.load("rouge")
# start fine-tuning the model mT5 with the trainer API
batch_size = 8
num_train_epochs = 8
logging_steps = len(encoded_train_dataset)
model_name = model_checkpoint.split("/")[-1]

args = Seq2SeqTrainingArguments(
    output_dir=f"{model_name}-finetuned-mT5-google-generated-paraphrased-dataset",
    evaluation_strategy="epoch",
    learning_rate=5.6e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=num_train_epochs,
    predict_with_generate=True,
    logging_steps=logging_steps,
    push_to_hub=True,
)


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    # ROUGE expects a newline after each sentence
    decoded_preds = ["\n".join(sent_tokenize(pred.strip())) for pred in decoded_preds]
    decoded_labels = ["\n".join(sent_tokenize(label.strip())) for label in decoded_labels]
    # Compute ROUGE scores
    result = rouge_score.compute(
        predictions=decoded_preds, references=decoded_labels, use_stemmer=True
    )
    # Extract the median scores
    result = {key: value * 100 for key, value in result.items()}
    return {k: round(v, 4) for k, v in result.items()}


encoded_train_dataset = encoded_train_dataset.remove_columns(summarization_dataset.column_names)
encoded_val_dataset = encoded_val_dataset.remove_columns(summarization_dataset.column_names)
encoded_test_dataset = encoded_test_dataset.remove_columns(summarization_dataset.column_names)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
features = [encoded_train_dataset[i] for i in range(2)]
data_collator(features)

trainer = Seq2SeqTrainer(
    model,
    args,
    train_dataset=encoded_train_dataset,
    eval_dataset=encoded_val_dataset,
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)
trainer.train()
trainer.evaluate()
trainer.push_to_hub(commit_message="Training complete", tags="summarization")
