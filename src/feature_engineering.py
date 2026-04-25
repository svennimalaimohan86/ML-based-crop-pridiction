from sklearn.preprocessing import LabelEncoder

def add_features(df):
    df["water_stress"] = df["temperature"] - df["rainfall"] * 0.1
    df["soil_quality"] = (df["N"] + df["P"] + df["K"]) / 3

    def risk_level(row):
        if row["rainfall"] < 50:
            return "High Risk"
        elif row["rainfall"] < 100:
            return "Medium Risk"
        else:
            return "Low Risk"

    df["risk"] = df.apply(risk_level, axis=1)

    risk_encoder = LabelEncoder()
    df["risk"] = risk_encoder.fit_transform(df["risk"])

    return df, risk_encoder