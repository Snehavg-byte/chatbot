import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle

# Load CSV
df = pd.read_csv("chat_data.csv")
df = df[['text', 'label']]

# Initialize LabelEncoder
label_encoder = LabelEncoder()
df['encoded_labels'] = label_encoder.fit_transform(df['label'])

# Save label encoder
with open("label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)

# Tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Encode Dataset
class ChatDataset(Dataset):
    def __init__(self, texts, labels):
        self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=64)
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

    def __len__(self):
        return len(self.labels)

# Split dataset
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df['text'].tolist(), df['encoded_labels'].tolist(), test_size=0.2, random_state=42
)

train_dataset = ChatDataset(train_texts, train_labels)
val_dataset = ChatDataset(val_texts, val_labels)

# Model
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=len(set(df['encoded_labels'])))

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=50,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

# Train
trainer.train()

# Save the trained model and tokenizer
model.save_pretrained("./results")
tokenizer.save_pretrained("./results")

