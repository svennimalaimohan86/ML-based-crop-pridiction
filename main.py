from src.data_preprocessing import load_and_preprocess
from src.feature_engineering import add_features
from src.train_model import train
from src.predict import show_predictions

def main():
    print("🚀 Starting Crop Risk Prediction Pipeline...\n")

    # Step 1: Load & preprocess data
    df, le_dict = load_and_preprocess()
    print("✅ Data preprocessing completed")

    # Step 2: Feature engineering
    df, risk_encoder = add_features(df)
    print("✅ Feature engineering completed")

    # Step 3: Train model
    y_test, y_pred, encoder = train(df, risk_encoder)
    print("✅ Model training completed")

    # Step 4: Show predictions
    show_predictions(y_test, y_pred, encoder)
    print("\n🎉 Pipeline completed successfully!")

# Run program
if __name__ == "__main__":
    main()