def suggest_pesticide(risk):
    if risk == "High Risk":
        return "⚠️ Use strong pesticide + increase irrigation"
    elif risk == "Medium Risk":
        return "⚡ Use moderate pesticide"
    else:
        return "✅ No pesticide needed"


def show_predictions(y_test, y_pred, encoder):
    print("\n🌾 Sample Predictions:")

    for i in range(5):
        actual = encoder.inverse_transform([y_test.iloc[i]])[0]
        pred = encoder.inverse_transform([y_pred[i]])[0]

        print(f"Actual: {actual} | Predicted: {pred}")
        print("Suggestion:", suggest_pesticide(pred))