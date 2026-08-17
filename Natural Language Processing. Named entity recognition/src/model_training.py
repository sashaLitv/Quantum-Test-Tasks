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
import sys
import argparse
from functools import singledispatch

def _tokenize_and_align_labels(examples, tokenizer, label2id):
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

# ==========================================
# Evaluation on test data using nervaluate
# ==========================================
@singledispatch
def evaluate_model(data, model, tokenizer=None, batch_size=16):
    '''
        Base function that raises an error when the provided data type is not supported.
    '''
    raise NotImplementedError(f"Unsupported data type for evaluation: {type(data)}")

## LEVEL 1: String (File Path) -> Loads data
@evaluate_model.register
def _(file_path: str, model, tokenizer, batch_size=16):
    '''
        Handles evaluation when a file path (str) is provided.
        Loads the JSON file and passes the raw list of dicts to the next level.
    '''
    print(f"Loading test data from {file_path}...")
    with open(file_path, "r", encoding="utf-8") as f:
        test_data = json.load(f)
    
    ## pass the loaded list of dictionaries to the list-handling method
    return evaluate_model(test_data, model, tokenizer, batch_size)

## LEVEL 2: List -> Data Prep & Error Analysis
@evaluate_model.register
def _(test_data: list, model, tokenizer, batch_size=16):
    '''
        Handles evaluation when raw data (list of dicts) is provided.
        Responsible for data tokenization, dataset preparation, and qualitative Error Analysis.
    '''
    label2id = {str(k): int(v) for k, v in model.config.label2id.items()}

    test_dataset = Dataset.from_list(test_data).map(
        lambda x: _tokenize_and_align_labels(x, tokenizer, label2id), 
        batched=True
    )
    
    ## initialize the data collator to handle dynamic padding during batching
    data_collator = DataCollatorForTokenClassification(
        tokenizer=tokenizer, 
        return_tensors="tf"
    )

    tf_test_dataset = test_dataset.to_tf_dataset(
        collate_fn=data_collator,
        batch_size=batch_size,
        shuffle=False, ## must be False to maintain the original sequence order for error analysis
        columns=["input_ids", "attention_mask"],
        label_cols=["labels"]
    )

    ## call LEVEL 3 to perform inference and compute metrics
    all_true_labels, all_pred_labels = evaluate_model(tf_test_dataset, model, tokenizer, batch_size)

    print("\n--- Error Analysis (Spurious / False Positives) ---")
    
    ## iterate through the predictions to find false positives
    for i, (true_seq, pred_seq) in enumerate(zip(all_true_labels, all_pred_labels)):
        has_error = False
        error_details = []
        tokens = test_data[i]["tokens"]

        encoding = tokenizer(tokens, is_split_into_words=True)
        valid_word_ids = [w for w in encoding.word_ids() if w is not None]

        for j, (t_tag, p_tag) in enumerate(zip(true_seq, pred_seq)):
            ## condition for a spurious detection
            if p_tag != "O" and t_tag == "O":
                has_error = True
                word_idx = valid_word_ids[j]
                word = tokens[word_idx]
                error_details.append((word, t_tag, p_tag))

        if has_error:
            print(f"Sentence: {' '.join(tokens)}")
            unique_errors = list(dict.fromkeys(error_details))
            for word, t, p in unique_errors:
                print(f" -> Found: '{word}' | Reality: {t} | Prediction: {p}")
            print("-" * 50)

## LEVEL 3: TF Dataset -> Inference & Metrics
@evaluate_model.register
def _(tf_test_dataset: tf.data.Dataset, model, tokenizer=None, batch_size=16):
    '''
        Core evaluation logic for TensorFlow datasets. 
        Calculates quantitative metrics using nervaluate and returns the exact label sequences.
    '''

    id2label = {int(k): v for k, v in model.config.id2label.items()}

    all_true_labels = []
    all_pred_labels = []

    for batch in tf_test_dataset:
        inputs, labels = batch[0], batch[1]

        ## perform forward pass to get logits 
        logits = model(**inputs, training=False).logits
        
        ## retrieve the most probable class index for each token
        batch_pred_ids = np.argmax(logits, axis=-1)
        batch_true_ids = labels.numpy()

        for i in range(len(batch_true_ids)):
            true_sequence, pred_sequence = [], []

            for t, p in zip(batch_true_ids[i], batch_pred_ids[i]):
                if t != -100: ## ignore special tokens (like [CLS], [SEP], [PAD]) masked with -100
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

    print("\n--- Evaluation Summary ---")
    print(evaluator.summary_report())

    ## return sequences back to LEVEL 2 so it can perform the error analysis
    return all_true_labels, all_pred_labels

def main(args):
    '''
        The function for running inferrnce.
        Accepts the output and dataset directory, seed and model's parameters.
    '''
    # ==========================================
    # Definition of variables
    # ==========================================
    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    save_dir = Path(args.output_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # ==========================================
    # Dataset loading/reading & data preparation  
    # ==========================================
    if args.dataset_dir:
        dataset_dir = Path(args.dataset_dir)
        if not dataset_dir.exists():
            print(f"Error: Dataset not found at {dataset_dir}")
            sys.exit(1)

        with open(dataset_dir / "train.json", "r", encoding="utf-8") as f:
            train_data = json.load(f)
        with open(dataset_dir / "valid.json", "r", encoding="utf-8") as f:
            val_data = json.load(f)
        with open(dataset_dir / "test.json", "r", encoding="utf-8") as f:
            test_data = json.load(f)

        print(f"Data successfully loaded from local path: {dataset_dir}")
    else:
        print("No dataset path provided. Attempting to download from GitHub...")
        try:
            BASE_GITHUB_URL = "https://raw.githubusercontent.com/sashaLitv/Quantum-Test-Tasks/main/Natural%20Language%20Processing.%20Named%20entity%20recognition/data/processed"
            
            train_data = requests.get(f"{BASE_GITHUB_URL}/train.json").json()
            val_data = requests.get(f"{BASE_GITHUB_URL}/valid.json").json()
            test_data = requests.get(f"{BASE_GITHUB_URL}/test.json").json()

            print("Data successfully downloaded from GitHub")
        except Exception as e:
            print(f"Error downloading data from GitHub: {e}")
            sys.exit(1)

    dataset_dict = DatasetDict({
        "train": Dataset.from_list(train_data),
        "val": Dataset.from_list(val_data),
        "test": Dataset.from_list(test_data)
    })

    # ==========================================
    # 3. Label definition
    # ==========================================
    ## list of BIO tags
    label_list = ["O", "B-MOUNTAIN", "I-MOUNTAIN"]
    ## create mappings between text labels and integer IDs, required by the Hugging Face model configuration
    label2id = {label: i for i, label in enumerate(label_list)}
    id2label = {i: label for i, label in enumerate(label_list)}

    # ======================================================
    # Tokenization using the official Hugging Face tutorial
    # =======================================================
    tokenizer = AutoTokenizer.from_pretrained(args.model_checkpoint)
    tokenized_datasets = dataset_dict.map(
        lambda examples: _tokenize_and_align_labels(examples, tokenizer, label2id), 
        batched=True
    )

    ## object that form a batch by using a list of dataset elements as input, may apply some processing (like padding)
    data_collator = DataCollatorForTokenClassification(
        tokenizer=tokenizer, 
        return_tensors="tf"
    )

    tf_train_dataset = tokenized_datasets["train"].to_tf_dataset(
        collate_fn=data_collator,
        batch_size=args.batch_size,
        shuffle=True,
        columns=["input_ids", "attention_mask"],
        label_cols=["labels"]
    )
    tf_val_dataset = tokenized_datasets["val"].to_tf_dataset(
        collate_fn=data_collator,
        batch_size=args.batch_size,
        shuffle=False,
        columns=["input_ids", "attention_mask"],
        label_cols=["labels"]
    )
    # ==========================================
    # Model Initialization and Training
    # ==========================================
    model = TFAutoModelForTokenClassification.from_pretrained(
        args.model_checkpoint, 
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

    model.compile(optimizer=Adam(learning_rate=args.learning_rate))
    model.fit(
        tf_train_dataset,
        validation_data=tf_val_dataset,
        epochs=args.epochs,
        callbacks=[
            early_stopping_callback,
            # checkpoint_callback
        ]
    )

    save_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)

    evaluate_model(tokenized_datasets["test"], model, tokenizer, batch_size = 16)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a NER model for Mountain detection using DeBERTa")

    parser.add_argument(
        "--output_dir", 
        type=str, 
        required=True, 
        help="Path to the directory where the fine-tuned model will be saved."
    )

    parser.add_argument(
        "--dataset_dir", 
        type=str, 
        required=False, 
        help="Path to the raw JSON dataset (e.g., data/raw/himalayas_ner_dataset.json)."
    )

    parser.add_argument(
        "--seed", 
        type=int, 
        default=42, 
        help="Random seed for reproducibility (default: 42)"
    )

    parser.add_argument(
        "--model_checkpoint", 
        type=str, 
        default="microsoft/deberta-v3-base", 
        help="Hugging Face model checkpoint (default: microsoft/deberta-v3-base)"
    )
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=16, 
        help="Training batch size (default: 16)"
    )
    parser.add_argument(
        "--epochs", 
        type=int, 
        default=10, 
        help="Number of training epochs (default: 10)."
    )
    parser.add_argument(
        "--learning_rate", 
        type=float, 
        default=2e-5, 
        help="Learning rate for the optimizer (default: 2e-5)."
    )

    args = parser.parse_args()
    
    try:
        main(args)
    except Exception as e:
        print(f"An error occurred during training: {e}")
        sys.exit(1)