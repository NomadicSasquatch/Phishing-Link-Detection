import matplotlib.pyplot as plt
import seaborn as sb

def plot_feature_boxplots(X):
    sb.set()
    num_features = X.shape[1]

    fig, axes = plt.subplots(num_features, figsize=(18, 3 * num_features))
    fig.tight_layout(pad=3.0)

    for idx, col in enumerate(X.columns):
        sb.boxplot(data=X[col], orient='h', ax=axes[idx])
        axes[idx].set_title(col)

    plt.show()
