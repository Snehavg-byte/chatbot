from transformers import BertTokenizer, BertForSequenceClassification
import torch
import pickle

# Correct the model path if necessary (ensure pytorch_model.bin exists)
model_path = "./results"  # Or "./results/checkpoint-<epoch>/"
tokenizer = BertTokenizer.from_pretrained(model_path)

# Load the trained model and tokenizer from the saved directory
model = BertForSequenceClassification.from_pretrained(model_path)

# Load the label encoder (from the file where you saved it after training)
with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

# Define the prediction function
def predict(text):
    # Tokenize the input text
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=64)

    # Get model predictions
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    # Get the predicted class (the class with the highest logit)
    predicted_class = torch.argmax(logits, dim=-1).item()

    # Decode the predicted label back to the original label
    predicted_label = label_encoder.inverse_transform([predicted_class])
    return predicted_label[0]

# Example usage: predict a label for a given text
if __name__ == "__main__":
    text = input("Enter a text to classify: ")  # Accept input text from the user
    predicted_label = predict(text)
    print(f"Predicted label: {predicted_label}")
