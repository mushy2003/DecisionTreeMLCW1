import data_loader
from cross_validation import k_fold_cross_validation
import numpy as np
import sys


def main(data_path):
    shuffled_data = data_loader.shuffled_load_data(data_path)

    print("Data loaded correctly!")

    print("Performing 10-fold cross validation...")

    accuracies, recalls_per_fold, precisions_per_fold, confusion_matrix, recall, precision, f1_score = k_fold_cross_validation(shuffled_data, 10, len(shuffled_data))

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

    print("Overall precision for each class calculated using the overall confusion matrix:")
    print(precision)

    print("Overall F1 score for each class calculated using the overall recall and overall precision:")
    print(f1_score)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 main PATHTODATA")
    else:
        data_path = sys.argv[1]
        main(data_path)




 
    
