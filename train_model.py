
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

data = {
    "income": [50000,60000,30000,80000,120000,20000,40000,75000,90000,100000],
    "loan_amount": [20000,25000,15000,30000,50000,10000,18000,27000,35000,40000],
    "credit_score": [650,700,550,720,800,500,600,710,730,780],
    "default": [0,0,1,0,0,1,1,0,0,0]
}

df = pd.DataFrame(data)

X = df.drop("default", axis=1)
y = df["default"]

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

pickle.dump((model, scaler), open("model.pkl", "wb"))

print("Model saved as model.pkl")
