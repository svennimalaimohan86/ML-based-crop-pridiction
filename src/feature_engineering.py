from sklearn.preprocessing import LabelEncoder

def add_features(df):
    # 🔥 Feature Engineering
    df["water_stress"] = df["temperature"] - df["rainfall"] * 0.05
    df["soil_quality"] = (df["N"] + df["P"] + df["K"]) / 3

    # 🧠 Risk Logic
    def risk_level(row):
        temp = row["temperature"]
        humidity = row["humidity"]
        rainfall = row["rainfall"]
        ph = row["ph"]
        soil_quality = row["soil_quality"]
        water_stress = row["water_stress"]

        # 🔴 HIGH RISK
        if (
            (rainfall < 40 and temp > 32) or
            (humidity < 40 and soil_quality < 40) or
            (water_stress > 30)
        ):
            return "High Risk"

        # 🟡 MEDIUM RISK
        elif (
            (rainfall < 80 and humidity < 60) or
            (ph < 5.5 or ph > 7.5) or
            (soil_quality < 60)
        ):
            return "Medium Risk"

        # 🟢 LOW RISK
        else:
            return "Low Risk"

    # Apply function
    df["risk"] = df.apply(risk_level, axis=1)

    # Encode labels
    risk_encoder = LabelEncoder()
    df["risk"] = risk_encoder.fit_transform(df["risk"])

    return df, risk_encoder