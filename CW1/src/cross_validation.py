import numpy as np
from numpy.random import default_rng
import evaluation
from decision_tree import DecisionTree

def k_fold_split(num_folds, num_instances, random_generator = default_rng()):
    shuffled_indices = random_generator.permutation(num_instances)
    split_indices = np.array_split(shuffled_indices, num_folds)
    return split_indices

def train_test_k_fold(num_folds, num_instances, random_generator=default_rng()):
    split_indices = k_fold_split(num_folds, num_instances, random_generator)
    folds = []
    for k in range(num_folds):
        test_indices = split_indices[k]
        train_indices = np.hstack(split_indices[:k] + split_indices[k+1:])
        folds.append([train_indices, test_indices])
    
    return folds

def k_fold_cross_validation(shuffled_data, num_folds, num_instances, random_generator=default_rng()):
    folds = train_test_k_fold(num_folds, num_instances, random_generator)
    accuracies = np.zeros((num_folds, ))
    confusion_matrix = np.zeros((4, 4))

    for (i, (train_indices, test_indices)) in enumerate(folds):
        train_data = shuffled_data[train_indices]
        test_data = shuffled_data[test_indices]
        
        model = DecisionTree()

        model.train(train_data)

        accuracies[i], confusion_mat = evaluation.evaluate(test_data, model)
        confusion_matrix += confusion_mat
    
    return accuracies, confusion_matrix


