import data_loader
from cross_validation import k_fold_cross_validation
import numpy as np
import sys


def main(data_path):
    # Shuffles the data and loads it in
    shuffled_data = data_loader.shuffled_load_data(data_path)

    print("Data loaded correctly!")

    print("Performing 10-fold cross validation...")

    # Performs cross validation and outputs the relevant statistics
    (accuracies, recalls_per_fold, precisions_per_fold, 
     confusion_matrix, recall, precision, 
     f1_score, normalised_confusion_matrix, normalised_accuracy, 
     normalised_recall, normalised_precision, normalised_f1) = k_fold_cross_validation(shuffled_data, 10, len(shuffled_data))

    print("Accuracy for each fold:")
    print(accuracies)

    print("Average accuracy:")
    print(np.mean(accuracies))

    print("Recall per class for each fold")
    print(recalls_per_fold)

    print("Average Recall per class:")
    print(np.mean(recalls_per_fold, axis=0))

    print("Precision per class for each fold:")
    print(precisions_per_fold)

    print("Average Precision per class:")
    print(np.mean(precisions_per_fold, axis=0))

    print("Overall confusion matrix...")
    print(confusion_matrix)

    print("Overall recall for each class calculated using the overall confusion matrix:")
    print(recall)

    print("Macro-averaged recall calculated using the overall confusion matrix:")
    print(np.mean(recall))

    print("Overall precision for each class calculated using the overall confusion matrix:")
    print(precision)

    print("Macro-averaged precision calculated using the overall confusion matrix:")
    print(np.mean(precision))

    print("Overall F1 score for each class calculated using the overall recall and overall precision:")
    print(f1_score)

    print("Macro-averaged F1 score")
    print(np.mean(f1_score))

    print("Normalised confusion matrix:")
    print(normalised_confusion_matrix)

    print("Accuracy calculated using Normalised confusion matrix:")
    print(normalised_accuracy)

    print("Recall for each class calculated using Normalised confusion matrix:")
    print(normalised_recall)

    print("Macro-averaged Recall from Normalised confusion matrix:")
    print(np.mean(normalised_recall))

    print("Precision for each class calculated using Normalised confusion matrix:")
    print(normalised_precision)

    print("Macro-averaged Precision from Normalised confusion matrix:")
    print(np.mean(normalised_precision))

    print("F1-Score for each class calculated using Normalised confusion matrix:")
    print(normalised_f1)

    print("Macro-averaged F1-score from Normalised confusion matrix:")
    print(np.mean(normalised_f1))

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 main PATHTODATA")
    else:
        data_path = sys.argv[1]
        main(data_path)




 
    
