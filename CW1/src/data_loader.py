import numpy as np

def load_data(filename):
    return np.loadtxt(filename)

def load_clean_data():
    return load_data("wifi_db/clean_dataset.txt")

def load_noisy_data():
    return load_data("wifi_db/noisy_dataset.txt")

clean_data = load_data("wifi_db/clean_dataset.txt")
noisy_data = load_data("wifi_db/noisy_dataset.txt")