import os
import re
import json
import pickle
import numpy as np
import nltk
from nltk.corpus import stopwords
from flask import Flask, request, jsonify

# Lazy imports container for torch and transformers to conserve RAM on constrained environments
torch = None
nn = None
AutoTokenizer = None
AutoModelForSequenceClassification = None
LSTMClassifier = None

def _import_torch_libs():
    global torch, nn, AutoTokenizer, AutoModelForSequenceClassification, LSTMClassifier
    if torch is None:
        print("Importing heavy libraries (torch, transformers) into memory...")
        import torch as _torch
        import torch.nn as _nn
        from transformers import AutoTokenizer as _AutoTokenizer, AutoModelForSequenceClassification as _AutoModelForSequenceClassification
        
        torch = _torch
        nn = _nn
        AutoTokenizer = _AutoTokenizer
        AutoModelForSequenceClassification = _AutoModelForSequenceClassification
        
        # Optimize PyTorch memory footprint in CPU constraints (Free tiers)
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        
        # Define identical LSTM structure dynamically
        class _LSTMClassifier(nn.Module):
            def __init__(self, vocab_size, embedding_dim=64, hidden_dim=64, output_dim=1):
                super().__init__()
                self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
                self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
                self.fc = nn.Linear(hidden_dim, output_dim)
                
            def forward(self, x):
                embedded = self.embedding(x)
                out, (hidden, cell) = self.lstm(embedded)
                logits = self.fc(hidden[-1])
                return logits
                
        LSTMClassifier = _LSTMClassifier

from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Ensure NLTK data is downloaded
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))

# Global model container
MODELS = {}

def load_all_models():
    # Enable deep learning (LSTM, BERT) only if configured (disabled by default on memory-constrained 512MB RAM environments)
    enable_dl = os.environ.get("ENABLE_DEEP_LEARNING", "false").lower() == "true"
    
    if enable_dl:
        _import_torch_libs()
    else:
        print("Deep Learning models (LSTM, BERT) are disabled to conserve RAM. Set ENABLE_DEEP_LEARNING=true to enable.")
        
    print("Loading models into memory...")
    import gc
    try:
        # Load vectorizer
        with open("backend/models/vectorizer.pkl", "rb") as f:
            MODELS["vectorizer"] = pickle.load(f)
        gc.collect()
        
        # Load ML Models
        ml_names = [
            "logistic_regression",
            "multinomial_naive_bayes",
            "decision_tree",
            "passive_aggressive_classifier",
            "random_forest",
            "support_vector_machine"
        ]
        for name in ml_names:
            path = f"backend/models/model_{name}.pkl"
            if os.path.exists(path):
                with open(path, "rb") as f:
                    MODELS[name] = pickle.load(f)
            else:
                print(f"Warning: Model {name} not found at {path}")
        gc.collect()

        if enable_dl:
            # Load LSTM
            vocab_path = "backend/models/lstm_vocab.json"
            lstm_path = "backend/models/lstm_model.pt"
            if os.path.exists(vocab_path) and os.path.exists(lstm_path):
                with open(vocab_path, "r") as f:
                    MODELS["lstm_vocab"] = json.load(f)
                
                lstm_model = LSTMClassifier(vocab_size=len(MODELS["lstm_vocab"]))
                lstm_model.load_state_dict(torch.load(lstm_path, map_location=torch.device('cpu')))
                lstm_model.eval()
                MODELS["lstm"] = lstm_model
            else:
                print("Warning: LSTM files not found.")
            gc.collect()

            # Load BERT
            bert_path = "backend/models/bert_model"
            if os.path.exists(bert_path):
                MODELS["bert_tokenizer"] = AutoTokenizer.from_pretrained(bert_path)
                bert_model = AutoModelForSequenceClassification.from_pretrained(bert_path)
                bert_model.eval()
                MODELS["bert"] = bert_model
            else:
                print("Warning: BERT model folder not found.")
            gc.collect()
        else:
            print("Skipping LSTM and BERT model load (ENABLE_DEEP_LEARNING is false).")

        # Load metrics
        metrics_path = "backend/models/metrics.json"
        if os.path.exists(metrics_path):
            with open(metrics_path, "r") as f:
                MODELS["metrics"] = json.load(f)
        else:
            MODELS["metrics"] = {}

        print("All available models loaded successfully.")
    except Exception as e:
        print(f"Error loading models: {e}")
    finally:
        gc.collect()

# Preprocessing stages collector
def run_nlp_pipeline(text):
    text_lower = text.lower()
    text_clean = re.sub(r'[^a-zA-Z\s]', '', text_lower)
    tokens = nltk.word_tokenize(text_clean)
    tokens_no_stop = [w for w in tokens if w not in stop_words]
    preprocessed_text = " ".join(tokens_no_stop)
    
    return {
        "raw": text,
        "lowercase": text_lower,
        "cleaned": text_clean,
        "tokenized": tokens,
        "stopwords_removed": tokens_no_stop,
        "final": preprocessed_text
    }

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def get_ml_prediction(model_name, tfidf_vec):
    model = MODELS.get(model_name)
    if not model:
        return None
        
    pred = int(model.predict(tfidf_vec)[0])
    
    # Extract confidence score/probability
    confidence = 0.5
    try:
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(tfidf_vec)[0]
            confidence = float(proba[pred])
        elif hasattr(model, "decision_function"):
            decision = float(model.decision_function(tfidf_vec)[0])
            prob_true = float(sigmoid(decision))
            confidence = prob_true if pred == 1 else (1.0 - prob_true)
    except Exception as e:
        print(f"Error extracting confidence for {model_name}: {e}")
        
    return {
        "prediction": pred, # 1 for True, 0 for Fake
        "confidence": confidence
    }

def get_lstm_prediction(text_cleaned):
    lstm_model = MODELS.get("lstm")
    vocab = MODELS.get("lstm_vocab")
    if not lstm_model or not vocab:
        return None
        
    # Convert text to sequence
    max_len = 100
    tokens = text_cleaned.split()
    seq = [vocab.get(w, vocab["<UNK>"]) for w in tokens[:max_len]]
    if len(seq) < max_len:
        seq = seq + [vocab["<PAD>"]] * (max_len - len(seq))
        
    seq_tensor = torch.tensor([seq], dtype=torch.long)
    
    with torch.no_grad():
        logits = lstm_model(seq_tensor).squeeze(1)
        prob = torch.sigmoid(logits).item()
        
    pred = 1 if prob > 0.5 else 0
    confidence = prob if pred == 1 else (1.0 - prob)
    
    return {
        "prediction": pred,
        "confidence": confidence
    }

def get_bert_prediction(text_cleaned):
    bert_model = MODELS.get("bert")
    tokenizer = MODELS.get("bert_tokenizer")
    if not bert_model or not tokenizer:
        return None
        
    inputs = tokenizer(
        text_cleaned,
        truncation=True,
        padding='max_length',
        max_length=128,
        return_tensors="pt"
    )
    
    with torch.no_grad():
        outputs = bert_model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1).squeeze(0)
        
    pred = torch.argmax(probs).item()
    confidence = probs[pred].item()
    
    return {
        "prediction": pred,
        "confidence": confidence
    }

@app.route("/predict", methods=["POST"])
def predict():
    try:
        if not MODELS:
            load_all_models()
            
        data = request.get_json(silent=True) or {}
        text = data.get("text", "").strip()
        
        if not text:
            return jsonify({"error": "Empty input text provided."}), 400
            
        # Run pipeline
        pipeline = run_nlp_pipeline(text)
        cleaned_text = pipeline["final"]
        
        # Get TF-IDF vector
        vectorizer = MODELS.get("vectorizer")
        if not vectorizer:
            return jsonify({"error": "Models not loaded correctly. Vectorizer missing."}), 500
            
        tfidf_vec = vectorizer.transform([cleaned_text])
        
        # Obtain predictions
        predictions = {}
        
        # 1. ML models
        ml_models = {
            "Logistic Regression": "logistic_regression",
            "Multinomial Naive Bayes": "multinomial_naive_bayes",
            "Decision Tree": "decision_tree",
            "Passive Aggressive Classifier": "passive_aggressive_classifier",
            "Random Forest": "random_forest",
            "Support Vector Machine": "support_vector_machine"
        }
        
        for pretty_name, model_key in ml_models.items():
            res = get_ml_prediction(model_key, tfidf_vec)
            if res:
                predictions[pretty_name] = res

        # 2. LSTM model
        lstm_res = get_lstm_prediction(cleaned_text)
        if lstm_res:
            predictions["LSTM"] = lstm_res

        # 3. BERT model
        bert_res = get_bert_prediction(cleaned_text)
        if bert_res:
            predictions["BERT"] = bert_res
            
        # Build list structure
        predictions_list = []
        votes_true = 0
        votes_fake = 0
        total_valid_models = 0
        
        for name, res in predictions.items():
            predictions_list.append({
                "model": name,
                "prediction": "True News" if res["prediction"] == 1 else "Fake News",
                "prediction_code": res["prediction"],
                "confidence": round(res["confidence"] * 100, 2)
            })
            if res["prediction"] == 1:
                votes_true += 1
            else:
                votes_fake += 1
            total_valid_models += 1
            
        # Ensemble Vote
        if total_valid_models > 0:
            ensemble_pred = 1 if votes_true >= votes_fake else 0
            ensemble_verdict = "True News" if ensemble_pred == 1 else "Fake News"
            winning_votes = votes_true if ensemble_pred == 1 else votes_fake
            ensemble_confidence = round((winning_votes / total_valid_models) * 100, 2)
        else:
            ensemble_verdict = "Unknown"
            ensemble_pred = -1
            ensemble_confidence = 0.0

        return jsonify({
            "preprocessing_steps": pipeline,
            "predictions": predictions_list,
            "ensemble": {
                "prediction": ensemble_verdict,
                "prediction_code": ensemble_pred,
                "confidence": ensemble_confidence,
                "votes_true": votes_true,
                "votes_fake": votes_fake,
                "total_models": total_valid_models
            }
        })
    except Exception as e:
        import traceback
        return jsonify({
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500

@app.route("/metrics", methods=["GET"])
def get_metrics():
    try:
        if not MODELS:
            load_all_models()
        return jsonify(MODELS.get("metrics", {}))
    except Exception as e:
        import traceback
        return jsonify({
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500

@app.route("/test", methods=["GET"])
def test():
    try:
        load_all_models()
        return jsonify({
            "status": "ok",
            "version": "v1.3-punkt-tab",
            "models_loaded": list(MODELS.keys())
        })
    except Exception as e:
        import traceback
        return jsonify({
            "status": "error",
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500

if __name__ == "__main__":
    load_all_models()
    port = int(os.environ.get("PORT", 5000))
    print(f"Starting Flask App on port {port}...")
    app.run(host="0.0.0.0", port=port, debug=False)
