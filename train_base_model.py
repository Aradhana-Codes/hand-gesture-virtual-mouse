import pandas as pd
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv(
    "dataset/Hand Gesture Landmark Coordinates Dataset.csv",
    skiprows=1,
    header=None
)

X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42
)

model = RandomForestClassifier(n_estimators=300)
model.fit(X_train, y_train)

train_accuracy = model.score(X_train, y_train)
print("Training Accuracy:", train_accuracy)

accuracy = model.score(X_test, y_test)
print("Test Accuracy:", accuracy)



os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/base_model.pkl")
joblib.dump(encoder, "models/label_encoder.pkl")

print("Base model saved.")