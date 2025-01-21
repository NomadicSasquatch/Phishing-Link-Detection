import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

def build_and_evaluate_pipeline(X_train, y_train, X_test, y_test):
    numerical_features = X_train.columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features)
        ]
    )
    pipeline = ImbPipeline([
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=42)),
        ('classifier', RandomForestClassifier(
            n_estimators=100,
            max_depth=20,
            max_features='log2',
            random_state=42
        ))
    ])

    pipeline.fit(X_train, y_train.values.ravel())

    y_pred_test = pipeline.predict(X_test)
    print(classification_report(y_test, y_pred_test))
    print(f'AUC-ROC: {roc_auc_score(y_test, y_pred_test):.3f}')

    conf_matrix = confusion_matrix(y_test, y_pred_test)
    plt.figure(figsize=(10, 7))
    sb.heatmap(conf_matrix, annot=True, fmt='g')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix for the Test Set (Pipeline)')
    plt.show()

    true_positive = conf_matrix[1, 1]
    true_negative = conf_matrix[0, 0]
    false_positive = conf_matrix[0, 1]
    false_negative = conf_matrix[1, 0]

    accuracy = (true_positive + true_negative) / np.sum(conf_matrix)
    tpr = true_positive / (true_positive + false_negative)
    tnr = true_negative / (true_negative + false_positive)
    fpr = false_positive / (false_positive + true_negative)
    fnr = false_negative / (false_negative + true_positive)

    print(f"Pipeline Test Accuracy: {accuracy:.3f}")
    print(f"True Positive Rate: {tpr:.3f}")
    print(f"True Negative Rate: {tnr:.3f}")
    print(f"False Positive Rate: {fpr:.3f}")
    print(f"False Negative Rate: {fnr:.3f}")

    y_pred_train = pipeline.predict(X_train)

    return pipeline
