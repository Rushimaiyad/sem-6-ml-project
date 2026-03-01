from flask import Flask, render_template, request
import pickle
import numpy as np
import matplotlib.pyplot as plt
import os

app = Flask(__name__)

model, scaler = pickle.load(open("model.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    income = float(request.form["income"])
    loan_amount = float(request.form["loan_amount"])
    credit_score = float(request.form["credit_score"])

    features = np.array([[income, loan_amount, credit_score]])
    features_scaled = scaler.transform(features)

    prediction = model.predict(features_scaled)[0]
    probability = model.predict_proba(features_scaled)[0][1]

    result = "HIGH RISK - Loan Default" if prediction == 1 else "LOW RISK - Safe Loan"

    # Create dummy confusion matrix for display
    cm = [[8,2],[1,9]]

    plt.figure(figsize=(4,4))
    plt.imshow(cm, cmap="Greens")
    plt.title("Confusion Matrix")
    plt.colorbar()
    plt.xticks([0,1], ["No Default", "Default"])
    plt.yticks([0,1], ["No Default", "Default"])

    for i in range(2):
        for j in range(2):
            plt.text(j, i, cm[i][j], ha="center", va="center", color="black")

    if not os.path.exists("static"):
        os.makedirs("static")

    plt.savefig("static/confusion_matrix.png")
    plt.close()

    return render_template("index.html",
                           prediction_text=result,
                           probability=round(probability*100,2),
                           risk=prediction)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)