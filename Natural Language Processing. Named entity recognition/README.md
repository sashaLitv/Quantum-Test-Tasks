# Task 1: Named Entity Recognition (NER) for Mountain Names

This directory contains the solution for Task1, which focuses on training and utilizing a Named Entity Recognition (NER) model to detect a custom `MOUNTAIN` entity in unstructured text. The solution utilizes a fine-tuned DeBERTa model.

## ⚙️ Setup Instructions

To run this project locally, it is recommended to use a virtual environment.
1. Clone the repository:
```text
git clone https://github.com/sashaLitv/Quantum-Test-Tasks.git
cd "Quantum-Test-Tasks/Natural Language Processing. Named entity recognition"
```
2. Create and activate a virtual environment:
```text
python3 -m venv venv_ner
source venv_ner/bin/activate  
```
3. Install dependencies:
```text
pip install -r requirements.txt
```


## How to Run

The training script is fully parameterized. You can adjust hyperparameters or provide your own dataset directory.
```text
usage: model_training.py [-h] --output_dir OUTPUT_DIR [--dataset_dir DATASET_DIR] [--seed SEED] [--model_checkpoint MODEL_CHECKPOINT]
                         [--batch_size BATCH_SIZE] [--epochs EPOCHS] [--learning_rate LEARNING_RATE]

Train a NER model for Mountain detection using DeBERTa

options:
  -h, --help            show this help message and exit
  --output_dir OUTPUT_DIR
                        Path to the directory where the fine-tuned model will be saved.
  --dataset_dir DATASET_DIR
                        Path to the directory containing train/valid/test JSON files. If empty, downloads from GitHub.
  --seed SEED           Random seed for reproducibility (default: 42)
  --model_checkpoint MODEL_CHECKPOINT
                        Hugging Face model checkpoint (default: microsoft/deberta-v3-base)
  --batch_size BATCH_SIZE
                        Training batch size (default: 16)
  --epochs EPOCHS       Number of training epochs (default: 10).
  --learning_rate LEARNING_RATE
                        Learning rate for the optimizer (default: 2e-5).
```
You can use the trained model to extract mountain entities directly from the command line using the model_inference.py script.
```text
usage: model_inference.py [-h] --model_dir MODEL_DIR [--text TEXT] [--file FILE]

Run inference using the trained NER Mountain Detector

options:
  -h, --help            show this help message and exit
  --model_dir MODEL_DIR
                        Path to the directory containing the fine-tuned model
  --text TEXT           A single text string to process
  --file FILE           Path to a text file containing sentences to process (one sentence per line).
```


## 📁 Project Structure

The project is structured to separate the dataset, training logic, and inference pipeline. 

```text
├── ABOUT  # Root-level file containing future improvements for Task 1 and Task 2
└── Natural Language Processing. Named entity recognition/        
    ├── data/
    │   ├── processed/                      # Split dataset ready for training
    │   │   ├── test.json
    │   │   ├── train.json
    │   │   └── valid.json
    │   └── raw/                            # Original generated dataset
    │       └── ner_dataset.json
    ├── models/
    │   └── deberta_ner_model/              # Saved fine-tuned model and tokenizer (generated after training)
    ├── notebooks/
    │   ├── dataset_creation.ipynb          # Notebook detailing the dataset generation process, validation and EDA
    │   └── demo.ipynb                      # Interactive demo showcasing edge cases, predictions, and metrics
    ├── src/
    │   ├── model_inference.py              # CLI script and `MountainDetector` class for inference
    │   └── model_training.py               # CLI script used for training and evaluating the model
    ├── README.md                           # Project documentation
    └── requirements.txt                    # Project dependencies
```


## Dataset Generation

The dataset was synthetically generated using the **Gemini API** to ensure a diverse and balanced set of contexts for the `MOUNTAIN` entity. All text examples are tokenized and annotated using the standard **BIO (Begin, Inside, Outside) tagging scheme**. 
The dataset consists of 2000 examples in total, created under two distinct sets of rules:

**1. Standard Contexts (~1600 examples):**
These examples represent typical occurrences of mountain names in text. They were generated using the following prompt rules:
```text
Rules:
        1. Each sentence MUST contain exactly one name of a prominent mountain from anywhere in the world (e.g., Alps, Andes, Rockies, Caucasus, Himalayas).
        2. Crucially, mix the naming styles: sometimes use the "Mount X" format (e.g., Mount Blanc), and sometimes use the mountain name directly without any prefix (e.g., Everest, Hoverla, Kilimanjaro).
        3. Vary the position of the mountain name: place it at the beginning, in the middle, or at the end of sentences across different examples.
        4. In many sentences, include various surrounding geographical features (e.g., provinces, nearby straits, valleys, rivers, bays, or borders).
        5. Only the mountain name should be tagged as "B-MOUNTAIN" / "I-MOUNTAIN". ALL other geographical entities MUST be tagged as "O".
```
**2. Hard Negatives & Edge Cases (~400 examples):**
These examples were specifically designed to challenge the model, prevent overfitting. They were generated  with strict instructions to output completely "O" tags for the following five categories:
```text
Categories:
        1. (Non-geographical use): mountain names used as brands, pet names, human names, or companies (e.g., "He named his dog Everest").
        2. (Other geographical entities): cities, countries, rivers, oceans, or straits (e.g., "The Bosphorus strait connects to the Black Sea").
        3. (Wordplay and common nouns): words like "mount", "mountain", "peak", or "summit" used as regular verbs or metaphorical nouns (e.g., "She reached the peak of her career").
        4. (Deceptive names): locations, universities, or media titles that contain the word "Mount" or "Mountain" but are NOT actual mountains (e.g., "Google's office is in Mountain View").
        5. (Other physical landforms): famous canyons, plateaus, hills, cliffs, or deserts (e.g., "Tourists love the Grand Canyon").

Rule: because there are NO actual mountains acting as mountains in these sentences, their `ner_tags` MUST be completely "O". Do NOT use "B-MOUNTAIN" or "I-MOUNTAIN" anywhere.
```


## Pre-trained Model

The core of this project is a DeBERTa architecture, which I fine-tuned specifically to recognize the custom MOUNTAIN class.
You can download the fully trained model weights from Google Drive here:

🔗 [[Link to Google Drive Here](https://drive.google.com/drive/folders/1-4JvL9quo1VIFkJH0la-emu6ldNtnp_s?hl=uk-UA)]

To run the inference scripts, simply download the model directory, place it in your local folder, and ensure the MODEL_PATH variable in the code points to it.


## Model Evaluation

The model was evaluated on a held-out test set to ensure robustness and high generalization. The evaluation metrics were calculated using the strict entity-level scoring provided by the `nervaluate` Python library. 

Below is a screenshot detailing the model's accuracy, precision, recall, and F1-score on the test examples:

<div align="center">
  <img width="964" height="139" alt="Screenshot 2026-08-16 at 01 16 56" src="https://github.com/user-attachments/assets/b5faea6f-69eb-44df-bd62-53551c6841bc" />
</div>
