import numpy as np

def load_data(filename):
    return np.loadtxt(filename)

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

clean_data = load_data("wifi_db/clean_dataset.txt")
noisy_data = load_data("wifi_db/noisy_dataset.txt")