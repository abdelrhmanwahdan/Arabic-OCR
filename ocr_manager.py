import argparse
import easyocr
import cv2
from spellchecker import SpellChecker
import torch
from transformers import BertTokenizer, BertForMaskedLM

class ArabicOCR:
    def __init__(self, model_name="aubmindlab/bert-base-arabertv02"):
        """
        Initialize the OCR reader, spell checker, and BERT model.
        
        Args:
        - model_name (str): Name of the BERT model for language correction. Default is 'aubmindlab/bert-base-arabertv02'.
        """
        self.reader = easyocr.Reader(['ar'])
        self.spell = SpellChecker(language='ar')
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForMaskedLM.from_pretrained(model_name)
    
    def perform_ocr(self, image_path):
        """
        Perform OCR on the provided image and return the detected text.

        Args:
        - image_path (str): Path to the image.

        Returns:
        - list: OCR results containing bounding boxes, detected text, and confidence.
        """
        image = cv2.imread(image_path)
        results = self.reader.readtext(image)
        return results
    
    def correct_spelling(self, text):
        """
        Correct the spelling of the given text using the spell checker.

        Args:
        - text (str): Text to correct.

        Returns:
        - str: Corrected text.
        """
        corrected_text = []
        for word in text.split():
            corrected_word = self.spell.correction(word)
            corrected_text.append(corrected_word)
        return ' '.join(corrected_text)

    def correct_with_bert(self, text):
        """
        Correct the spelling using a BERT model for masked language modeling.

        Args:
        - text (str): Text to correct.

        Returns:
        - str: Corrected text based on BERT's masked language model.
        """
        tokens = self.tokenizer.tokenize(text)
        masked_tokens = ['[MASK]' if self.spell.unknown([token]) else token for token in tokens]

        input_ids = self.tokenizer.convert_tokens_to_ids(masked_tokens)
        input_ids = torch.tensor([input_ids])
        
        with torch.no_grad():
            outputs = self.model(input_ids)
        predictions = outputs.logits

        predicted_tokens = []
        for i, token in enumerate(masked_tokens):
            if token == '[MASK]':
                predicted_id = torch.argmax(predictions[0, i]).item()
                predicted_token = self.tokenizer.convert_ids_to_tokens([predicted_id])[0]
                predicted_tokens.append(predicted_token)
            else:
                predicted_tokens.append(token)
        
        corrected_text = self.tokenizer.convert_tokens_to_string(predicted_tokens)
        return corrected_text
    
    def process_image(self, image_path):
        """
        Process the image by performing OCR, spell correction, and BERT correction.
        
        Args:
        - image_path (str): Path to the image.
        """
        results = self.perform_ocr(image_path)
        
        for (bbox, text, prob) in results:
            # Apply spell check
            corrected_text_spell = self.correct_spelling(text)
            
            # Apply language model correction
            corrected_text_bert = self.correct_with_bert(corrected_text_spell)
            
            print(f"Detected text: {text} (Confidence: {prob})")
            print(f"Corrected text (Spell Check): {corrected_text_spell}")
            print(f"Corrected text (BERT): {corrected_text_bert}")



def parse_args():
    """
    Parses the arguments passed through the command line.
    """
    
    parser = argparse.ArgumentParser(description="Arabic OCR with Spell Correction and BERT-based Language Model Correction")
    
    # Argument for the image path
    parser.add_argument(
        '--image', 
        type=str, 
        required=True, 
        help="Path to the image file"
    )

    # Optional argument for the BERT model name
    parser.add_argument(
        '--model', 
        type=str, 
        default="aubmindlab/bert-base-arabertv02", 
        help="Name of the BERT model for language correction (default: 'aubmindlab/bert-base-arabertv02')"
    )

    return parser.parse_args()


def main():
    # Parse the command-line arguments
    args = parse_args()

    # Initialize the ArabicOCR class with the specified BERT model
    ocr = ArabicOCR(model_name=args.model)

    # Process the image
    ocr.process_image(args.image)

if __name__ == "__main__":
    main()
