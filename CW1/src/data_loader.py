import numpy as np

def load_data(filename):
    return np.loadtxt(filename)

# This will return the data of the chosen dataset in a shuffled manner
def shuffled_load_data(filename):
    rng = np.random.default_rng()
    return rng.permutation(load_data(filename))

# The below functions were used to make manual testing of our code easier
def load_clean_data():
    return load_data("wifi_db/clean_dataset.txt")

def load_noisy_data():
    return load_data("wifi_db/noisy_dataset.txt")

def shuffled_clean_data():
    rng = np.random.default_rng()
    return rng.permutation(load_clean_data())

def shuflled_noisy_data():
    rng = np.random.default_rng()
    return rng.permutation(load_noisy_data())