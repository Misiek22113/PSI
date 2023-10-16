import random
import numpy as np
from main import neural_network

# zad 1

def learning_rate():

    weight = np.array([0.5])
    input = np.array([2])
    expected_result = 0.8
    alpha = 0.1

    for i in range(20):
        res = neural_network(input, weight, 0)
        delta = 2 * (res - expected_result) * input
        error = (res - expected_result) ** 2
        print("epoka: ", i+1, "\nwaga: ", weight, "\nerror: ", error, "\nwynik: ", res, "\n")
        weight = weight - delta * alpha

learning_rate()

# zad 2



