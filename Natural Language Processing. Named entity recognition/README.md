# Task 1: Named Entity Recognition (NER) for Mountain Names

This directory contains the solution for Task1, which focuses on training and utilizing a Named Entity Recognition (NER) model to detect a custom `MOUNTAIN` entity in unstructured text. The solution utilizes a fine-tuned DeBERTa model.

## 📁 Project Structure

The project is structured to separate the dataset, training logic, and inference pipeline. 

```text
├── ABOUT  # Root-level file containing future improvements for Task 1 and Task 2
└── Natural Language Processing. Named entity recognition/        
    ├── requirements.txt 
    └── src/
        ├── dataset/
        │   ├── himalayas_ner_dataset.json  # The generated NER dataset
        │   └── dataset_creation.ipynb      # Notebook detailing the dataset generation process
        ├── model_training.py               # Script used for training the model
        ├── model_inference.py              # Contains the `MountainDetector` class for inference
        └── demo.ipynb                      # Interactive demo showcasing edge cases and predictions
```

## Dataset Generation
The dataset was synthetically generated using the **Gemini API** to ensure a diverse and balanced set of contexts for the `MOUNTAIN` entity. All text examples are tokenized and annotated using the standard **BIO (Begin, Inside, Outside) tagging scheme**. 
The dataset consists of 600 examples in total, created under two distinct sets of rules:

**1. Standard Contexts (500 examples):**
These examples represent typical occurrences of mountain names in text. They were generated using the following prompt rules:
```text
Rules:
        1. Each sentence MUST contain exactly one name of a prominent mountain from anywhere in the world (e.g., Alps, Andes, Rockies, Caucasus, Himalayas).
        2. Crucially, mix the naming styles: sometimes use the "Mount X" format (e.g., Mount Blanc), and sometimes use the mountain name directly without any prefix (e.g., Everest, Hoverla, Kilimanjaro).
        3. Vary the position of the mountain name: place it at the beginning, in the middle, or at the end of sentences across different examples.
        4. In many sentences, include various surrounding geographical features (e.g., provinces, nearby straits, valleys, rivers, bays, or borders).
        5. Only the mountain name should be tagged as "B-MOUNTAIN" / "I-MOUNTAIN". ALL other geographical entities MUST be tagged as "O".
```
**2. Hard Negatives & Edge Cases (100 examples):**
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
  <img width="800" height="190" alt="Screenshot 2026-07-29 at 17 14 25" src="https://github.com/user-attachments/assets/39d6cdfa-bee2-459b-a5d6-f9c9bd03f267" />
</div>
