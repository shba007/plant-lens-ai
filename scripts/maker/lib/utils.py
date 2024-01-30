import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import metrics


def plot_confusion_matrix(cm, classes, normalize=False, simple=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function plots a confusion matrix.
    cm: confusion matrix
    classes: a list of class names
    normalize: whether to normalize the matrix or not
    title: plot title
    cmap: color map
    """
    # Create a new figure and set its size
    fig = plt.figure(figsize=(16, 14))

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if (simple == False):
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=90)
        plt.yticks(tick_marks, classes)

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in np.ndindex(cm.shape):
            plt.text(j, i, format(int(np.round(
                cm[i, j], 3)*100)), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.show()


def avg_matrix(matrix, labels):
    unique_labels = np.unique(labels)

    avg_matrix = np.zeros((len(unique_labels), len(unique_labels)))

    for i, row_label in enumerate(unique_labels):
        for j, col_label in enumerate(unique_labels):
            row_indices = np.where(labels == row_label)[0]
            col_indices = np.where(labels == col_label)[0]
            avg = np.mean(matrix[row_indices[:, np.newaxis], col_indices])
            avg_matrix[i, j] = avg

    return avg_matrix, unique_labels


def calculate_similarity_matrix(features, batch_size=25):
    num_features = features.shape[0]
    similarity_matrix = tf.zeros(
        (num_features, num_features), dtype=tf.float32).numpy()

    num_batches = (num_features + batch_size - 1) // batch_size
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, num_features)
        batch_features = features[start_idx:end_idx]

        batch_similarity = tf.reduce_sum(
            tf.square(batch_features[:, tf.newaxis] - features), axis=-1)
        similarity_matrix[start_idx:end_idx, :] = batch_similarity

    return (1 - similarity_matrix / tf.reduce_max(similarity_matrix)).numpy()


def oneshot_accuracy(similarity_matrix, true_labels):
    predictions_labels = true_labels[tf.argmax(similarity_matrix, axis=-1)]

    accuracy_metric = metrics.Accuracy()
    accuracy_metric.update_state(true_labels, predictions_labels)
    accuracy = accuracy_metric.result().numpy()

    return accuracy
