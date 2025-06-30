from src.data_loader import load_data
from src.feature_engineering import extract_features
from src.model import train_model, save_model
from src.model_comparison import compare_models
import joblib

df = load_data("C:/Users/nisha/Desktop/credit_score_system/data/raw/cs-training.csv")
X, scaler = extract_features(df)
joblib.dump(scaler, "C:/Users/nisha/Desktop/credit_score_system/models/scaler.pkl")
y = df['defaults']

# Compare and select best model (optional)
results = compare_models(X, y)
print("Model Comparison:")
for model_name, score in results.items():
    print(f"{model_name}: {score['mean_accuracy']:.4f} ± {score['std']:.4f}")

best_model = max(results, key=lambda k: results[k]['mean_accuracy'])
model = train_model(X, y, model_type=best_model.lower().replace(" ", "_"))
save_model(model)
