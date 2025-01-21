import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.metrics import confusion_matrix

def evaluate_model(model, X, y, set_name="Train"):
    y_pred = model.predict(X)
    cm = confusion_matrix(y, y_pred)

    plt.figure(figsize=(10, 7))
    sb.heatmap(cm, annot=True, fmt='g')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix for the {set_name} Set')
    plt.show()

    true_positive = cm[1, 1]
    true_negative = cm[0, 0]
    false_positive = cm[0, 1]
    false_negative = cm[1, 0]

    accuracy = model.score(X, y)
    tpr = true_positive / (true_positive + false_negative)
    tnr = true_negative / (true_negative + false_positive)
    fpr = false_positive / (false_positive + true_negative)
    fnr = false_negative / (false_negative + true_positive)

    print(f"{set_name} Accuracy: {accuracy:.3f}")
    print(f"True Positive Rate: {tpr:.3f}")
    print(f"True Negative Rate: {tnr:.3f}")
    print(f"False Positive Rate: {fpr:.3f}")
    print(f"False Negative Rate: {fnr:.3f}")
