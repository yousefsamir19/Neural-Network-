"""
MLP Classifier — Flask API backend
===================================
Place this file in your project directory alongside:
    mlp.py  |  preprocessing.py  |  penguins.csv

Install dependencies (once):
    pip install flask flask-cors scikit-learn numpy pandas

Run:
    python app.py

Then open mlp_classifier.html in your browser.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np

# Project modules — must be in the same directory
from preprocessing import get_preprocessed_df, split
from mlp import mlp as MLP

app = Flask(__name__)
CORS(app)  # Allow cross-origin requests from the HTML file


# ── Health check ─────────────────────────────────────────────────────────────
@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


# ── Train + evaluate ─────────────────────────────────────────────────────────
@app.route("/api/train", methods=["POST"])
def train_model():
    try:
        body = request.get_json(force=True)

        # ── Parse hyperparameters ─────────────────────────────────────────
        hidden_layers  = max(1, int(body.get("hidden_layers", 1)))
        hidden_neurons = [max(1, int(n)) for n in body.get("hidden_neurons", [4])]
        learning_rate  = float(body.get("learning_rate", 0.1))
        epochs         = max(1, int(body.get("epochs", 100)))
        bias           = bool(body.get("bias", False))
        mse_threshold  = float(body.get("mse_threshold", 0.01))
        mse_flag       = int(bool(body.get("mse_flag", False)))
        act_str        = body.get("activation_function", "sigmoid")
        activation_fn  = 0 if act_str == "sigmoid" else 1   # 0→sigmoid, 1→tanh

        # Guard: ensure hidden_neurons length matches hidden_layers
        while len(hidden_neurons) < hidden_layers:
            hidden_neurons.append(4)
        hidden_neurons = hidden_neurons[:hidden_layers]

        # ── Load & preprocess data ────────────────────────────────────────
        df = get_preprocessed_df()
        X_train, y_train, X_test, y_test, sc = split(df)

        # ── Build model (mse + mse_flag now live inside mlp) ─────────────
        model = MLP(
            X_train, y_train,
            hidden_layers, hidden_neurons,
            learning_rate, epochs,
            bias, activation_fn,
            mse=mse_threshold,
            mse_flag=mse_flag,
        )

        # ── Train — early stopping handled inside mlp.train() ─────────────
        train_result = model.train()
        train_accuracy = train_result[0]
        train_cm       = train_result[1]
        stopped_epoch  = train_result[2] if len(train_result) > 2 else epochs
        train_mse      = train_result[3] if len(train_result) > 3 else None

        # ── Test ──────────────────────────────────────────────────────────
        test_accuracy, avg_loss, test_cm = model.test(X_test, y_test)

        # ── Snapshot weights & activations for visualization ──────────────
        weights     = [layer.weights.tolist() for layer in model.layers]
        model.forward_pass(X_train.iloc[0, :])
        activations = [layer.outputs.flatten().tolist() for layer in model.layers]

        return jsonify({
            "accuracy"        : round(test_accuracy, 2),
            "train_accuracy"  : round(train_accuracy, 2),
            "loss"            : round(avg_loss, 4),
            "train_mse"       : train_mse,
            "stopped_epoch"   : stopped_epoch,
            "confusion_matrix": test_cm,
            "train_cm"        : train_cm,
            "weights"         : weights,
            "activations"     : activations,
        })

    except Exception as exc:
        import traceback
        print(traceback.format_exc())
        return jsonify({"error": str(exc), "trace": traceback.format_exc()}), 500


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n  ╔══════════════════════════════════════╗")
    print("  ║  MLP Classifier API                  ║")
    print("  ║  http://localhost:5000               ║")
    print("  ╚══════════════════════════════════════╝\n")
    app.run(host="0.0.0.0", port=5000, debug=False)
