import sys
from onnxruntime import InferenceSession
from transformers import AutoTokenizer
import numpy as np

model_name = "bodias/distilbert-base-uncased-finetuned-FiNER"
labels = ['O',
 'B-DebtInstrumentBasisSpreadOnVariableRate1',
 'B-DebtInstrumentFaceAmount',
 'B-LineOfCreditFacilityMaximumBorrowingCapacity',
 'B-DebtInstrumentInterestRateStatedPercentage']

def load_model():
    print("loading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    session = InferenceSession("onnx/model.onnx")
    return tokenizer, session

if __name__ == "__main__":
    # Get text from command line arguments
    if len(sys.argv) < 2:
        print("Usage: python cli_demo.py <text>")
        sys.exit(1)
    text = ' '.join(sys.argv[1:])
    
    tokenizer, sess = load_model()
    
    words = text.split()
    # Process the text
    processed_text =  tokenizer(words, truncation=True, max_length=512, is_split_into_words=True, return_tensors="np")    
    # Run the model
    outputs = sess.run(output_names=["logits"], input_feed=dict(processed_text))
    prediction = np.argmax(outputs[0], axis=2)
    
    # Process output
    text_output = []
    original_tokens = tokenizer.convert_ids_to_tokens(processed_text["input_ids"].tolist()[0])
    for word, tag in zip(original_tokens, prediction[0]):
        text_output.append(f" {word} [{labels[tag]}]")

    # Output the processed text
    print("Processed text:")
    print(text_output)