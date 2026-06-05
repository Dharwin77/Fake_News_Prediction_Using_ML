import os
import re
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import nltk
from nltk.corpus import stopwords

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Ensure NLTK data is downloaded
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text_lower = text.lower()
    text_clean = re.sub(r'[^a-zA-Z\s]', '', text_lower)
    tokens = nltk.word_tokenize(text_clean)
    tokens_no_stop = [w for w in tokens if w not in stop_words]
    return " ".join(tokens_no_stop)

def main():
    print("Creating output directories...")
    os.makedirs("backend/models", exist_ok=True)

    print("Loading datasets...")
    true_df = pd.read_csv('dataset/True.csv')
    fake_df = pd.read_csv('dataset/Fake.csv')

    true_df['label'] = 1
    fake_df['label'] = 0

    # Combine title and text to provide rich features for both training and inference
    true_df['full_text'] = true_df['title'] + " " + true_df['text']
    fake_df['full_text'] = fake_df['title'] + " " + fake_df['text']

    # Downsample to speed up training on CPU while maintaining high quality
    # We will sample 3000 true articles and 3000 fake articles (total 6000)
    print("Downsampling data for efficient CPU training...")
    true_sample = true_df.sample(n=min(3000, len(true_df)), random_state=42)
    fake_sample = fake_df.sample(n=min(3000, len(fake_df)), random_state=42)
    
    data = pd.concat([true_sample, fake_sample]).reset_index(drop=True)
    
    # Shuffle dataset
    data = data.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print("Running NLP Preprocessing (lowercasing, tokenization, stopword removal)...")
    data['cleaned_text'] = data['full_text'].apply(clean_text)
    
    # Filter out empty texts
    data = data[data['cleaned_text'] != ""].reset_index(drop=True)
    print(f"Total processed samples: {len(data)}")

    # Split dataset (80% train, 20% test)
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        data['cleaned_text'].tolist(),
        data['label'].tolist(),
        test_size=0.2,
        random_state=42
    )

    metrics = {}

    # =========================================================================
    # 1. Classic Machine Learning Models (TF-IDF + Scikit-Learn)
    # =========================================================================
    print("\nFitting TF-IDF Vectorizer...")
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_tfidf = vectorizer.fit_transform(train_texts)
    X_test_tfidf = vectorizer.transform(test_texts)

    # Save Vectorizer
    with open("backend/models/vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)

    ml_models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Multinomial Naive Bayes": MultinomialNB(),
        "Decision Tree": DecisionTreeClassifier(max_depth=15, random_state=42),
        "Passive Aggressive Classifier": PassiveAggressiveClassifier(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1),
        "Support Vector Machine": LinearSVC(max_iter=1000, random_state=42)
    }

    for name, model in ml_models.items():
        print(f"Training {name}...")
        model.fit(X_train_tfidf, train_labels)
        
        # Predict & Evaluate
        preds = model.predict(X_test_tfidf)
        acc = accuracy_score(test_labels, preds)
        prec = precision_score(test_labels, preds)
        rec = recall_score(test_labels, preds)
        f1 = f1_score(test_labels, preds)
        
        print(f"{name} -> Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")
        
        metrics[name] = {
            "accuracy": float(acc),
            "precision": float(prec),
            "recall": float(rec),
            "f1_score": float(f1)
        }
        
        # Save model
        model_key = name.lower().replace(" ", "_")
        with open(f"backend/models/model_{model_key}.pkl", "wb") as f:
            pickle.dump(model, f)

    # =========================================================================
    # 2. PyTorch LSTM Model
    # =========================================================================
    print("\nTraining PyTorch LSTM Model...")
    # Build vocabulary from training text
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for text in train_texts:
        for word in text.split():
            if word not in vocab:
                vocab[word] = len(vocab)
                
    # Limit vocab size to 5000
    vocab_limit = 5000
    if len(vocab) > vocab_limit:
        sorted_vocab = sorted(vocab.items(), key=lambda x: x[1])[:vocab_limit]
        vocab = {w: i for i, (w, _) in enumerate(sorted_vocab)}
        
    # Save vocabulary
    with open("backend/models/lstm_vocab.json", "w") as f:
        json.dump(vocab, f)

    def text_to_sequence(text, max_len=100):
        tokens = text.split()
        seq = [vocab.get(w, vocab["<UNK>"]) for w in tokens[:max_len]]
        # Padding
        if len(seq) < max_len:
            seq = seq + [vocab["<PAD>"]] * (max_len - len(seq))
        return seq

    X_train_lstm = np.array([text_to_sequence(t) for t in train_texts])
    X_test_lstm = np.array([text_to_sequence(t) for t in test_texts])

    class LSTMClassifier(nn.Module):
        def __init__(self, vocab_size, embedding_dim=64, hidden_dim=64, output_dim=1):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
            self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
            self.fc = nn.Linear(hidden_dim, output_dim)
            
        def forward(self, x):
            # x shape: [batch_size, seq_len]
            embedded = self.embedding(x) # [batch_size, seq_len, emb_dim]
            out, (hidden, cell) = self.lstm(embedded)
            # Take last hidden state of the LSTM
            # hidden[-1] is [batch_size, hidden_dim]
            logits = self.fc(hidden[-1])
            return logits

    class LSTMDataset(Dataset):
        def __init__(self, sequences, labels):
            self.sequences = torch.tensor(sequences, dtype=torch.long)
            self.labels = torch.tensor(labels, dtype=torch.float32)
        def __len__(self):
            return len(self.labels)
        def __getitem__(self, idx):
            return self.sequences[idx], self.labels[idx]

    train_lstm_ds = LSTMDataset(X_train_lstm, train_labels)
    test_lstm_ds = LSTMDataset(X_test_lstm, test_labels)

    train_lstm_loader = DataLoader(train_lstm_ds, batch_size=64, shuffle=True)
    test_lstm_loader = DataLoader(test_lstm_ds, batch_size=64, shuffle=False)

    lstm_model = LSTMClassifier(vocab_size=len(vocab))
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(lstm_model.parameters(), lr=0.005)

    # Train LSTM for 5 epochs
    lstm_model.train()
    for epoch in range(5):
        epoch_loss = 0
        for seqs, lbls in train_lstm_loader:
            optimizer.zero_grad()
            logits = lstm_model(seqs).squeeze(1)
            loss = criterion(logits, lbls)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"LSTM Epoch {epoch+1}/5 - Loss: {epoch_loss/len(train_lstm_loader):.4f}")

    # Evaluate LSTM
    lstm_model.eval()
    lstm_preds = []
    with torch.no_grad():
        for seqs, lbls in test_lstm_loader:
            logits = lstm_model(seqs).squeeze(1)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).int().tolist()
            lstm_preds.extend(preds)

    acc = accuracy_score(test_labels, lstm_preds)
    prec = precision_score(test_labels, lstm_preds)
    rec = recall_score(test_labels, lstm_preds)
    f1 = f1_score(test_labels, lstm_preds)
    print(f"LSTM -> Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")

    metrics["LSTM"] = {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1_score": float(f1)
    }

    # Save LSTM model
    torch.save(lstm_model.state_dict(), "backend/models/lstm_model.pt")

    # =========================================================================
    # 3. Transformer (BERT-Tiny) Model
    # =========================================================================
    print("\nTraining Transformer (BERT-Tiny) Model...")
    # Load model and tokenizer
    bert_model_name = "prajjwal1/bert-tiny"
    bert_tokenizer = AutoTokenizer.from_pretrained(bert_model_name)
    bert_model = AutoModelForSequenceClassification.from_pretrained(bert_model_name, num_labels=2)

    # For BERT, we will train on a smaller subset of 2000 train samples and evaluate on 400 to keep it extremely fast
    bert_train_texts, bert_val_texts, bert_train_labels, bert_val_labels = train_test_split(
        train_texts[:2000], train_labels[:2000], test_size=0.2, random_state=42
    )

    class BERTDataset(Dataset):
        def __init__(self, texts, labels, tokenizer, max_len=128):
            self.texts = texts
            self.labels = labels
            self.tokenizer = tokenizer
            self.max_len = max_len
        def __len__(self):
            return len(self.labels)
        def __getitem__(self, idx):
            text = self.texts[idx]
            label = self.labels[idx]
            encoding = self.tokenizer(
                text,
                truncation=True,
                padding='max_length',
                max_length=self.max_len,
                return_tensors="pt"
            )
            return {
                'input_ids': encoding['input_ids'].squeeze(0),
                'attention_mask': encoding['attention_mask'].squeeze(0),
                'label': torch.tensor(label, dtype=torch.long)
            }

    train_bert_ds = BERTDataset(bert_train_texts, bert_train_labels, bert_tokenizer)
    val_bert_ds = BERTDataset(bert_val_texts, bert_val_labels, bert_tokenizer)
    test_bert_ds = BERTDataset(test_texts, test_labels, bert_tokenizer)

    train_bert_loader = DataLoader(train_bert_ds, batch_size=32, shuffle=True)
    test_bert_loader = DataLoader(test_bert_ds, batch_size=32, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bert_model.to(device)
    
    bert_optimizer = AdamW(bert_model.parameters(), lr=5e-5)

    # Train BERT for 1 epoch (very fast on CPU for bert-tiny)
    bert_model.train()
    print("Fine-tuning bert-tiny for 1 epoch...")
    epoch_loss = 0
    for batch in train_bert_loader:
        bert_optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        
        outputs = bert_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        bert_optimizer.step()
        epoch_loss += loss.item()
    print(f"BERT Epoch 1/1 - Loss: {epoch_loss/len(train_bert_loader):.4f}")

    # Evaluate BERT
    bert_model.eval()
    bert_preds = []
    with torch.no_grad():
        for batch in test_bert_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            outputs = bert_model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=1).tolist()
            bert_preds.extend(preds)

    acc = accuracy_score(test_labels, bert_preds)
    prec = precision_score(test_labels, bert_preds)
    rec = recall_score(test_labels, bert_preds)
    f1 = f1_score(test_labels, bert_preds)
    print(f"BERT -> Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")

    metrics["BERT"] = {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1_score": float(f1)
    }

    # Save BERT model & tokenizer
    bert_model.save_pretrained("backend/models/bert_model")
    bert_tokenizer.save_pretrained("backend/models/bert_model")

    # =========================================================================
    # Save Metrics JSON
    # =========================================================================
    print("\nSaving metrics to backend/models/metrics.json...")
    with open("backend/models/metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    print("\nTraining complete! All models saved successfully.")

if __name__ == "__main__":
    main()
