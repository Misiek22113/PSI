# zad 1
import numpy as np

from fcl import FCL_API
from lab3 import read_mnist_images, read_mnist_labels, create_expected_matrix


def train_on_mnis_with_dropout(training_range, test_range, alpha, epochs, hidden_neurons):
    model_3 = FCL_API(784)
    model_3.add_layer(hidden_neurons, -0.1, 0.1, 'relu')
    model_3.add_layer(10, -0.1, 0.1, 'relu')

    training_input = read_mnist_images('train-images-idx3-ubyte')
    training_expected = read_mnist_labels('train-labels-idx1-ubyte')

    for i in range(training_range):
        training_single_expected = create_expected_matrix(training_expected[i])
        if i % 1000 == 0:
            print(i)
        model_3.fit(training_input[i].reshape(-1, 1) / 255, training_single_expected, alpha, epochs)

    test_input = read_mnist_images('t10k-images-idx3-ubyte')
    test_expected = read_mnist_labels('t10k-labels-idx1-ubyte')

    correct_answers = 0
    lines = 0
    for i in range(test_range):
        res = model_3.predict(test_input[i].reshape(-1, 1))
        if np.argmax(res) == test_expected[i]:
            correct_answers += 1
        lines += 1

    print('percentage', correct_answers / lines)

# train_on_mnis_with_dropout(1000, 10000, 0.005, 350, 40)
train_on_mnis_with_dropout(10000, 10000, 0.005, 350, 100)