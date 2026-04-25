import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def train(df, risk_encoder):
    os.makedirs("models", exist_ok=True)

    X = df.drop(["risk","rainfall"], axis=1)
    y = df["risk"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(n_estimators=150, max_depth=10, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("✅ Accuracy:", accuracy_score(y_test, y_pred))

    joblib.dump(model, "models/crop_risk_model.pkl")
    joblib.dump(scaler, "models/scaler.pkl")
    joblib.dump(risk_encoder, "models/risk_encoder.pkl")

    return y_test, y_pred, risk_encoder