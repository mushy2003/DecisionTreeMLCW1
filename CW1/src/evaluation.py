import numpy as np

def confusion_matrix(y_gold, y_prediction, class_labels=None):
    if not class_labels:
        class_labels = np.unique(np.concatenate((y_gold, y_prediction)))

    confusion = np.zeros((len(class_labels), len(class_labels)), dtype=int)

    for (i, label) in enumerate(class_labels):
        # get predictions where the ground truth is the current class label
        indices = (y_gold == label)
        gold = y_gold[indices]
        predictions = y_prediction[indices]

        (unique_labels, counts) = np.unique(predictions, return_counts=True)

        # convert the counts to a dictionary
        frequency_dict = dict(zip(unique_labels, counts))

        # fill up the confusion matrix for the current row
        for (j, class_label) in enumerate(class_labels):
            confusion[i, j] = frequency_dict.get(class_label, 0)

    return confusion

def accuracy_from_confusion(confusion):
    if np.sum(confusion) > 0:
        return np.sum(np.diag(confusion)) / np.sum(confusion)
    else:
        return 0.

def evaluate(test_data_with_labels, trained_tree):
    y_gold = test_data_with_labels[:, -1]
    # Need to pass in the test data without the labels
    y_pred = trained_tree.predict(test_data_with_labels[:, :-1])

    confusion = confusion_matrix(y_gold, y_pred, [1, 2, 3, 4])

    return accuracy_from_confusion(confusion)