import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def train(df, risk_encoder):
    # 📁 Create models folder
    os.makedirs("models", exist_ok=True)

    # 🎯 Features & Target
    X = df.drop(["risk", "rainfall"], axis=1)
    y = df["risk"]

    # ⚙️ Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 🔀 Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    # 🤖 Model (Tuned)
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1
    )

    # 🚀 Train
    model.fit(X_train, y_train)

    # 🔮 Predict
    y_pred = model.predict(X_test)

    # 📊 Accuracy
    print("✅ Accuracy:", accuracy_score(y_test, y_pred))

    # 💾 Save
    joblib.dump(model, "models/crop_risk_model.pkl")
    joblib.dump(scaler, "models/scaler.pkl")
    joblib.dump(risk_encoder, "models/risk_encoder.pkl")

    return y_test, y_pred, risk_encoder