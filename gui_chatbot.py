import tkinter as tk
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import joblib
import numpy as np

# Load model and tokenizer
model_path = "./results"
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForSequenceClassification.from_pretrained(model_path)
model.eval()

# Load label encoder
label_encoder = joblib.load("label_encoder.pkl")

# Intent to response mapping
intent_to_response = {
    "greeting": "Hi there! How can I help you today?",
    "question": "That's a good question. Let me find that out for you.",
    "goodbye": "Goodbye! Have a nice day 😊",
    "internet_issue": "Let me help you fix your internet issues.",
    "account_recovery": "I can guide you through account recovery.",
    "app_crash": "Sorry about that. Let's troubleshoot the app crash.",
    "billing_issue": "I’m here to help you with billing. Could you tell me more about the issue?",
    "weather_query": "Sure! Let me check the weather for you.",
    "book_flight": "I can help you book a flight. Where would you like to go?",
    "cancel_ticket": "I’ll help you cancel your ticket. Could you provide more details?",
    "smalltalk": "I’m doing great, thanks! 😊",

}

# Example to check prediction
inputs = tokenizer("I need a flight to London", return_tensors="pt", padding=True, truncation=True, max_length=128)
with torch.no_grad():
    outputs = model(**inputs)
logits = outputs.logits
predicted_class_id = torch.argmax(logits, dim=1).item()
predicted_label = label_encoder.inverse_transform([predicted_class_id])[0]
print(f"Predicted label for 'I need a flight to London': {predicted_label}")

# Function to predict and display result
def get_response():
    user_input = entry.get()
    if not user_input.strip():
        return
    inputs = tokenizer(user_input, return_tensors="pt", padding=True, truncation=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    predicted_class_id = torch.argmax(logits, dim=1).item()
    predicted_label = label_encoder.inverse_transform([predicted_class_id])[0]
    response = intent_to_response.get(predicted_label, "Sorry, I’m not sure how to respond to that.")

    chat_history.config(state="normal")
    chat_history.insert(tk.END, f"You: {user_input}\n")
    chat_history.insert(tk.END, f"Bot: {response}\n\n")
    chat_history.config(state="disabled")
    entry.delete(0, tk.END)

# Tkinter UI
root = tk.Tk()
root.title("Chatbot")

chat_history = tk.Text(root, height=20, width=60, state="disabled", wrap="word", bg="#f5f5f5")
chat_history.pack(padx=10, pady=10)

entry = tk.Entry(root, width=50)
entry.pack(side=tk.LEFT, padx=10, pady=10)

send_btn = tk.Button(root, text="Send", command=get_response)
send_btn.pack(side=tk.LEFT)

root.mainloop()
