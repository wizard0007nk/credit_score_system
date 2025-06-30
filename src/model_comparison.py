from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import numpy as np


def compare_models(X, y, cv=5):
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
    }

    scores = {}
    for name, model in models.items():
        cv_scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
        scores[name] = {
            'mean_accuracy': np.mean(cv_scores),
            'std': np.std(cv_scores)
        }
    return scores
