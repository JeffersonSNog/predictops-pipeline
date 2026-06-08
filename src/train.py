import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score, classification_report
from src.preprocess import preprocess_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV

def train_baseline(data_path: str, experiment_name: str = "predictops-baseline"):
    
    mlflow.set_experiment(experiment_name)
    
    X_train, X_test, y_train, y_test, scaler = preprocess_pipeline(data_path)
    
    with mlflow.start_run(run_name="logistic_regression"):
        
        model = LogisticRegression(
            class_weight='balanced',
            max_iter=1000,
            random_state=42
        )
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
        
        mlflow.log_param("model", "LogisticRegression")
        mlflow.log_param("class_weight", "balanced")
        mlflow.log_param("max_iter", 1000)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("roc_auc", auc)
        mlflow.sklearn.log_model(model, "model")
        
        print(classification_report(y_test, y_pred, target_names=['Sem Falha', 'Falha']))
        print(f"F1 Score: {f1:.4f}")
        print(f"ROC-AUC: {auc:.4f}")
        
        return model, scaler
    
def train_random_forest(data_path: str, experiment_name: str = "predictops-baseline"):
    mlflow.set_experiment(experiment_name)
    
    X_train, X_test, y_train, y_test, scaler = preprocess_pipeline(data_path)
    
    with mlflow.start_run(run_name="random_forest"):
        
        model = RandomForestClassifier(
            n_estimators=100,
            class_weight='balanced',
            random_state=42
        )
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
        
        mlflow.log_param("model", "RandomForest")
        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("class_weight", "balanced")
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("roc_auc", auc)
        mlflow.sklearn.log_model(model, "model")
        
        print(classification_report(y_test, y_pred, target_names=['Sem Falha', 'Falha']))
        print(f"F1 Score: {f1:.4f}")
        print(f"ROC-AUC: {auc:.4f}")
        
        return model, scaler
    
def gradient_boosting(data_path: str ,experiment_name: str = "predictops-baseline"):
    mlflow.set_experiment(experiment_name)
    X_train, X_test, y_train, y_test, scaler = preprocess_pipeline(data_path)

    with mlflow.start_run(run_name="gradient_boosting"):

        model = GradientBoostingClassifier(
        n_estimators=100,
        random_state=42
        )
        model.fit(X_train, y_train)
            
        y_pred = model.predict(X_test)
            
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])

        mlflow.log_param("model", "GradientBoosting")
        mlflow.log_param("n_estimators", 100)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("roc_auc", auc)
        mlflow.sklearn.log_model(model, "model")
            
        print(classification_report(y_test, y_pred, target_names=['Sem Falha', 'Falha']))
        print(f"F1 Score: {f1:.4f}")
        print(f"ROC-AUC: {auc:.4f}")
            
        return model, scaler

def train_gradient_boosting_optimized(data_path: str, experiment_name: str = "predictops-baseline"):
    
    mlflow.set_experiment(experiment_name)
    
    X_train, X_test, y_train, y_test, scaler = preprocess_pipeline(data_path)
    
    param_dist = {
        'n_estimators': [50, 100, 200, 300],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'max_depth': [3, 4, 5, 6],
        'min_samples_split': [2, 5, 10]
    }
    
    with mlflow.start_run(run_name="gradient_boosting_optimized"):
        
        base_model = GradientBoostingClassifier(random_state=42)
        
        search = RandomizedSearchCV(
            base_model,
            param_distributions=param_dist,
            n_iter=20,
            cv=5,
            scoring='f1',
            random_state=42,
            n_jobs=-1
        )
        search.fit(X_train, y_train)
        
        model = search.best_estimator_
        y_pred = model.predict(X_test)
        
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
        
        mlflow.log_params(search.best_params_)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("roc_auc", auc)
        mlflow.sklearn.log_model(model, "model")
        
        print(f"Melhores parâmetros: {search.best_params_}")
        print(classification_report(y_test, y_pred, target_names=['Sem Falha', 'Falha']))
        print(f"F1 Score: {f1:.4f}")
        print(f"ROC-AUC: {auc:.4f}")
        
        return model, scaler