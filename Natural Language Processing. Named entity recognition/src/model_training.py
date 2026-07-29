'''
    Training script for Named Entity Recognition (NER) model using DeBERTa.
    Task: Identify mountain names (MOUNTAIN entity) in text.

    This script covers data loading, tokenization (handling subwords), 
    model training via TensorFlow/Keras, and final evaluation using the 'nervaluate' library.
'''

import json
from pathlib import Path
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, TFAutoModelForTokenClassification
from transformers import DataCollatorForTokenClassification
import requests
import tensorflow as tf
from tf_keras.optimizers import Adam
from nervaluate import Evaluator
import numpy as np
import random

# ==========================================
# Definition of variables
# ==========================================
## default path for local execution (.py script)
save_dir = Path(__file__).parent.parent / "deberta_ner_model"

## --- OPTIONAL: GOOGLE COLAB SETUP ---
## to run this code in Google Colab (using a Jupyter Notebook), uncomment and configure requirements 

# import google.colab
# from google.colab import drive
# drive.mount('/content/drive')
# save_dir = Path('/content/drive/MyDrive/deberta_ner_model')


SEED = 42
MODEL_CHECKPOINT = "microsoft/deberta-v3-base"
## list of BIO tags
label_list = ["O", "B-MOUNTAIN", "I-MOUNTAIN"]
## create mappings between text labels and integer IDs, required by the Hugging Face model configuration
label2id = {label: i for i, label in enumerate(label_list)}
id2label = {i: label for i, label in enumerate(label_list)}

# ==========================================
# Dataset Loading/Reading & Data Preparation  
# ==========================================
try:
    GITHUB_URL = "https://raw.githubusercontent.com/sashaLitv/Quantum-Test-Taks/main/src/dataset/himalayas_ner_dataset.json"
    response = requests.get(GITHUB_URL)
    response.raise_for_status()  
    raw_data = response.json()
    print("Data downloaded from GitHub")
except Exception as e:
    current_dir = Path(__file__).parent
    dataset_path = current_dir / "dataset" / "himalayas_ner_dataset.json"
    with open(dataset_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    print("Data read from local version ")
    
random.seed(SEED)
random.shuffle(raw_data)

dataset = Dataset.from_list(raw_data)
temp_test_dataset = dataset.train_test_split(test_size=0.2, seed=SEED, shuffle=True)
train_val_dataset = temp_test_dataset["train"].train_test_split(test_size=0.2, seed=SEED, shuffle=True)
dataset_dict = DatasetDict({
    "train": train_val_dataset["train"],
    "val": train_val_dataset["test"],
    "test": temp_test_dataset["test"]
})

# ==========================================
# Tokenization using the official Hugging Face tutorial
# ==========================================
tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)
def tokenize_and_align_labels(examples):
    ''' 
        Tokenizes input texts and aligns NER labels with the generated subwords.
        Special tokens are assigned -100 to be ignored during loss calculation. 
    '''
    tokenized = tokenizer(
        examples["tokens"],
        truncation=True,
        is_split_into_words=True
    )

    aligned_labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized.word_ids(batch_index=i)  
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:  
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx: 
                label_ids.append(label2id[label[word_idx]])
            else: 
                ## modified from the official Hugging Face tutorial: assigning I-MOUNTAIN to subwords instead of -100
                if label[word_idx] == "B-MOUNTAIN": 
                    label_ids.append(label2id["I-MOUNTAIN"])
                else:
                    label_ids.append(label2id[label[word_idx]])
            previous_word_idx = word_idx
        aligned_labels.append(label_ids)

    tokenized["labels"] = aligned_labels
    return tokenized

tokenized_datasets = dataset_dict.map(tokenize_and_align_labels, batched=True)

## object that form a batch by using a list of dataset elements as input, may apply some processing (like padding)
data_collator = DataCollatorForTokenClassification(
    tokenizer=tokenizer, 
    return_tensors="tf"
)

tf_train_dataset = tokenized_datasets["train"].to_tf_dataset(
    collate_fn=data_collator,
    batch_size=16,
    shuffle=True,
    columns=["input_ids", "attention_mask"],
    label_cols=["labels"]
)
tf_val_dataset = tokenized_datasets["val"].to_tf_dataset(
    collate_fn=data_collator,
    batch_size=16,
    shuffle=False,
    columns=["input_ids", "attention_mask"],
    label_cols=["labels"]
)
tf_test_dataset = tokenized_datasets["test"].to_tf_dataset(
    collate_fn=data_collator,
    batch_size=16,
    shuffle=False,
    columns=["input_ids", "attention_mask"],
    label_cols=["labels"]
)

# ==========================================
# Model Initialization and Training
# ==========================================
model = TFAutoModelForTokenClassification.from_pretrained(
    MODEL_CHECKPOINT, 
    num_labels=len(label_list),
    id2label=id2label,
    label2id=label2id,
    from_pt=True
)

## сheckpoint callback to save the best weights during training 
## (crucial for Colab to prevent losing progress if the session disconnects)
# checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
#     filepath=str(save_dir / "checkpoint_best.keras"),
#     monitor="val_loss",
#     save_best_only=True,
#     save_weights_only=False,
#     verbose=1
# )

early_stopping_callback = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",    
    patience=2,           
    restore_best_weights=True, 
    verbose=1               
)

model.compile(optimizer=Adam(learning_rate=2e-5))
model.fit(
    tf_train_dataset,
    validation_data=tf_val_dataset,
    epochs=10,
    callbacks=[
        early_stopping_callback,
        # checkpoint_callback
    ]
)

save_dir.mkdir(parents=True, exist_ok=True)
model.save_pretrained(save_dir)
tokenizer.save_pretrained(save_dir)


# ==========================================
# Evaluation on test data using nervaluate
# ==========================================
all_true_labels = []
all_pred_labels = []

for batch in tf_test_dataset:
    inputs, labels = batch[0], batch[1]
    
    logits = model(**inputs, training=False).logits
    batch_pred_ids = np.argmax(logits, axis=-1)
    
    batch_true_ids = labels.numpy()
    
    for i in range(len(batch_true_ids)):
        true_sequence, pred_sequence = [], []
        
        for t, p in zip(batch_true_ids[i], batch_pred_ids[i]):
            if t != -100: # ignore [CLS], [SEP], and [PAD] tokens
            # convert IDs back to BIO string labels
                true_sequence.append(id2label[t])
                pred_sequence.append(id2label[p])
                
        all_true_labels.append(true_sequence)
        all_pred_labels.append(pred_sequence)

evaluator = Evaluator(
    all_true_labels,
    all_pred_labels,
    tags=["MOUNTAIN"],
    loader="list"
)

print(evaluator.summary_report())
