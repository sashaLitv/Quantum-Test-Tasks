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
import sys
import argparse
from IPython import get_ipython

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

    def _is_notebook(self) -> bool:
        try:
            shell = get_ipython().__class__.__name__
            if shell == 'ZMQInteractiveShell':
                return True   ## jupyter notebook or qtconsole
            elif shell == 'TerminalInteractiveShell':
                return False  ## terminal running IPython
            else:
                return False  ## other type 
        except NameError:
            return False      ## probably standard Python interpreter


    def visualize(self, text: str, predictions: list):
        '''
            Renders text with colored highlights for detected entities.
            Automatically detects if running in Jupyter (HTML) or CLI (ANSI colors).
            Args:
               text(str): the original input sentence.
               predictions(list): the list of tuple (words, tags)
        '''
        is_jupyter = self._is_notebook()

        if is_jupyter:
            html_out = f"<div style='font-size: 16px; line-height: 1.6; padding: 10px; border: 1px solid #ddd; border-radius: 5px; margin-bottom: 10px;'>"
            html_out += f"<div style='margin-bottom: 8px; font-weight: bold;'>Input: {text}</div>"
            
            for word, tag in predictions:
                if tag == "O":
                    html_out += f"<span>{word} </span>"
                else:
                    ## If is mountain (B-MOUNTAIN або I-MOUNTAIN), highlight
                    color = "#00ff886a" 
                    html_out += f"<span style='background-color: {color}; padding: 2px 5px; border-radius: 4px; font-weight: 500; margin-right: 4px;'>{word}</span>"
                    
            html_out += "</div>"
            display(HTML(html_out))

        else:
            cli_out = f"Input: {text}\nOutput: "
            for word, tag in predictions:
                if tag == "O":
                    cli_out += f"{word} "
                else:
                    # \033[1m - bold, \033[92m - green, \033[0m - reset
                    cli_out += f"\033[1m\033[92m{word}\033[0m "
            
            print(cli_out + "\n")

def main(model_dir: str, text: str = None, file_path: str = None):
    '''
        The function for running inferrnce.
        Accepts the model directory and either a single text string or a path to a file.
    '''
    if not text and not file_path:
        print("Error: You must provide either --text or --file")
        sys.exit(1)
    elif text and file_path:
        print("Error: Please provide either --text OR --file, not both")
        sys.exit(1)

    try:
        detector = MountainDetector(model_dir)
        test_texts = []
        if file_path:
            path_obj = Path(file_path)
            if not path_obj.exists():
                raise FileNotFoundError(f"Input file not found: {path_obj}")
            with open(path_obj, "r", encoding="utf-8") as f:
                test_texts = [line.strip() for line in f if line.strip()]
        else:
            test_texts = [text]

        predictions = detector.predict(test_texts)

        for origin, prediction in zip(test_texts, predictions):
            detector.visualize(origin, prediction)

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference using the trained NER Mountain Detector")
    parser.add_argument(
        "--model_dir", 
        type=str, 
        required=True, 
        help="Path to the directory containing the fine-tuned model"
    )
    parser.add_argument(
        "--text", 
        type=str, 
        required=False,
        help="A single text string to process"
    )
    parser.add_argument(
        "--file", 
        type=str, 
        required=False,
        help="Path to a text file containing sentences to process (one sentence per line)."
    )

    args = parser.parse_args()
    main(model_dir=args.model_dir, text=args.text, file_path=args.file)

        
        


    


