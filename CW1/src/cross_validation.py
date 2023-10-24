import numpy as np
from numpy.random import default_rng

def k_fold_split(num_folds, num_instances, random_generator = default_rng()):
    shuffled_indices = random_generator.permutation(num_instances)
    split_indices = np.array_split(shuffled_indices, num_folds)
    return split_indices

def train_val_test_k_fold(num_folds, num_instances, random_generator=default_rng()):
    split_indices = k_fold_split(num_folds, num_instances, random_generator)

    folds = []
    for k in range(num_folds):
        test_indices = split_indices[k]
        val_index = (k+1) % num_folds
        val_indices = split_indices[val_index]
        train_indices = np.hstack(split_indices[:min(k, val_index)] + split_indices[min(k, val_index)+1 : max(k, val_index)] + split_indices[max(k, val_index)+1:])
        folds.append([train_indices, val_indices, test_indices])
    
    return folds

def k_fold_cross_validation(model, shuffled_data, num_folds, num_instances, random_generator=default_rng()):
    folds = train_val_test_k_fold(num_folds, num_instances, random_generator)
    accuracies = np.zeros((num_folds, ))

    for (i, (train_indices, val_indices, test_indices)) in enumerate(folds):
        train_data = shuffled_data[train_indices]
        val_data = shuffled_data[val_indices, :-1]
        test_data = shuffled_data[test_indices, :-1]
        val_data_labels = shuffled_data[val_indices, -1]
        test_data_labels = shuffled_data[val_indices, -1]

        model.train(train_data)

        #TODO: use the validation and test folds


