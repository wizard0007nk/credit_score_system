from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

def train_model(X, y, model_type='random_forest'):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LogisticRegression(max_iter=1000) if model_type == 'logistic_regression' else RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"{model_type.replace('_', ' ').title()} Accuracy: {acc:.2f}")
    return model

def save_model(model, path="C:/Users/nisha/Desktop/credit_score_system/models/credit_model.pkl"):
    joblib.dump(model, path)

def load_model(path="C:/Users/nisha/Desktop/credit_score_system/models/credit_model.pkl"):
    return joblib.load(path)
