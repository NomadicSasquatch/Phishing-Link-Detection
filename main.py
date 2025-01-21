import os
from src.data_preprocessing import load_and_clean_data, split_data
from src.eda import plot_feature_boxplots
from src.model_training import (
    train_decision_tree, train_random_forest, plot_feature_importance
)
from src.evaluation import evaluate_model
from src.tuning import tune_random_forest
from src.pipeline import build_and_evaluate_pipeline

def main():
    data_path = os.path.join("data", "Phising_dataset_predict.csv")
    data = load_and_clean_data(data_path)

    X_train, X_test, y_train, y_test = split_data(data, target_col='Phising')

    dt_model = train_decision_tree(X_train, y_train, max_depth=3)
    plot_feature_importance(dt_model, X_train)
    evaluate_model(dt_model, X_train, y_train, set_name="Train (Decision Tree)")
    evaluate_model(dt_model, X_test, y_test, set_name="Test (Decision Tree)")

    rf_model = train_random_forest(X_train, y_train, n_estimators=100, max_depth=20, max_features='log2')
    evaluate_model(rf_model, X_train, y_train, set_name="Train (Random Forest)")
    evaluate_model(rf_model, X_test, y_test, set_name="Test (Random Forest)")

    pipeline = build_and_evaluate_pipeline(X_train, y_train, X_test, y_test)

if __name__ == "__main__":
    main()
