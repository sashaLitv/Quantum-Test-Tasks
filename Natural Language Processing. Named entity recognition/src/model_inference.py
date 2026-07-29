'''
    Inference and visualization script for Named Entity Recognition (NER) model.
    Task: Identify mountain names (MOUNTAIN entity) in text.

    This script covers loading a fine-tuned DeBERTa model, performing inference 
    on single or multiple sentences, reconstructing subword tokens, and providing 
    HTML-based visual highlighting of detected entities for Jupyter/Colab environments.
'''

from pathlib import Path
from transformers import AutoTokenizer, TFAutoModelForTokenClassification
import numpy as np
from functools import singledispatchmethod
from IPython.display import display, HTML

class MountainDetector:
    '''A class to encapsulate the NER model and tokenizer for detecting mountain names'''

    def __init__(self, model_dir: str):
        """
            Initializes the MountainDetector by loading the model and tokenizer.

            Args:
                model_dir (str): the local path to the directory containing 
                                the fine-tuned model and tokenizer files.

            Raises:
                FileNotFoundError: if the specified model_dir does not exist.
        """
        self.model_dir = Path(model_dir)
        if not self.model_dir.exists():
            raise FileNotFoundError(f"Model directory not found: {self.model_dir}")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
        self.model = TFAutoModelForTokenClassification.from_pretrained(self.model_dir)
        self.id2label = self.model.config.id2label

    def _predict(self, text: str):
        '''
            Run inference on a single text string and return detected entities
            This method tokenizes the input, performs inference, handles 
            subword token merging (e.g., removing '##'), and reconstructs 
            the final entities.

            Args:
                text (str): The input sentence to process.

            Returns:
                list of tuples: A list where each element is a tuple containing 
                                (word, label), e.g., [("Everest", "B-MOUNTAIN")].
        '''
        tokenized_inputs = self.tokenizer(text, return_tensors="tf", truncation=True)
        predicted_outputs = self.model(**tokenized_inputs, training=False)
        logits = predicted_outputs.logits

        predicted_ids = np.argmax(logits, axis=-1)[0]

        tokens = self.tokenizer.convert_ids_to_tokens(tokenized_inputs["input_ids"][0])
        labels = [self.id2label[p] for p in predicted_ids]

        special_tokens = {"[CLS]", "[SEP]", "[PAD]", "<s>", "</s>"}
        results = []
        current_word = ""
        current_label = None
        for token, label in zip(tokens, labels):
            ## skip special tokens used by the transformer architecture
            if token in special_tokens:
                continue

            ## handle subwords: subwords don`t start with ▁
            if not token.startswith("▁"):
                current_word += token
            else:
                if current_word:
                    current_word = current_word.replace(" ", "").replace("▁", "")
                    results.append((current_word, current_label))
            
                current_word = token
                current_label = label

        if current_word:
            current_word = current_word.replace(" ", "").replace("▁", "")
            results.append((current_word, current_label))

        return results

    @singledispatchmethod
    def predict(self, arg):
        raise NotImplementedError("Unsupported type of data")
    @predict.register
    def _(self, text: str):
        return self._predict(text)
    @predict.register
    def _(self, texts: list):
        return [self._predict(t) for t in texts]

    def visualize(self, text: str, predictions: list):
        '''
            Renders text with colored highlights for detected entities.
            Args:
               text(str): the original input sentence.
               predictions(list): the list of tuple (words, tags)
        '''
        html_out = f"<div style='font-size: 16px; line-height: 1.6; padding: 10px; border: 1px solid #ddd; border-radius: 5px; margin-bottom: 10px;'>"
        html_out += f"<div style='margin-bottom: 8px; font-weight: bold;'>Input: {text}</div>"
        
        for word, tag in predictions:
            if tag == "O":
                html_out += f"<span>{word} </span>"
            else:
                # If is mountain (B-MOUNTAIN або I-MOUNTAIN), highlight
                color = "#00ff886a" 
                html_out += f"<span style='background-color: {color}; padding: 2px 5px; border-radius: 4px; font-weight: 500; margin-right: 4px;'>{word}</span>"
                
        html_out += "</div>"
        display(HTML(html_out))

if __name__ == "__main__":
    MODEL_PATH = "deberta_ner_model"
    try:
        detector = MountainDetector(MODEL_PATH)
        test_texts = [
            "We are planning to climb Mount Everest and Kilimanjaro next year",
            "The Andes is the longest continental mountain range in the world",
            "I went to the supermarket to buy some apples",
            "He named his dog Everest",
            "Kyiv is the capital of Ukraine"
        ]
        print(detector.predict(test_texts))

    except FileNotFoundError as e:
        print(f"Error:{e}")

        
        


    


