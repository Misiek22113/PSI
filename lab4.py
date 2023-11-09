# zad 1
import time

import numpy as np

from fcl import FCL_API
from lab3 import read_mnist_images, read_mnist_labels


def train_on_mnis_with_dropout(training_range, test_range, alpha, epochs, hidden_neurons, batch=False, batch_size=100):
    model_3 = FCL_API(784)
    model_3.add_layer(hidden_neurons, -0.1, 0.1, 'relu')
    model_3.add_layer(10, -0.1, 0.1, 'relu')

    training_input = read_mnist_images('train-images-idx3-ubyte')
    training_expected = read_mnist_labels('train-labels-idx1-ubyte')

    images = (training_input.reshape(training_input.shape[0], -1) / 255).T
    labels = (np.eye(10)[training_expected]).T

    images_training_set = images[:, :training_range]
    labels_training_set = labels[:, :training_range]

    if batch:
        for i in range(100, training_range-1, batch_size):
            model_3.fit_batch(images_training_set[:, :i], labels_training_set[:, :i], alpha,
                              epochs)
    else:
        model_3.fit(images_training_set, labels_training_set, alpha, epochs)

    test_input = read_mnist_images('t10k-images-idx3-ubyte')
    test_expected = read_mnist_labels('t10k-labels-idx1-ubyte')

    correct_answers = 0
    lines = 0
    for i in range(test_range):
        res = model_3.predict(test_input[i].reshape(-1, 1), False)
        if np.argmax(res) == test_expected[i]:
            correct_answers += 1
        lines += 1

    print('percentage', correct_answers / lines)


# czas_poczatkowy = time.time()
# train_on_mnis_with_dropout(1000, 10000, 0.005, 350, 40)
# train_on_mnis_with_dropout(10000, 10000, 0.005, 350, 100)
# train_on_mnis_with_dropout(10000, 10000, 0.005, 350, 100, True)
# czas_koncowy = time.time()

# czas_trwania = czas_koncowy - czas_poczatkowy

# print("Czas trwania programu: {} sekundy".format(czas_trwania))

czas_poczatkowy = time.time()

# train_on_mnis_with_dropout(1000, 10000, 0.1, 350, 40, True)
train_on_mnis_with_dropout(10000, 10000, 0.1, 350, 100, True)
czas_koncowy = time.time()

czas_trwania = czas_koncowy - czas_poczatkowy

print("Czas trwania programu: {} sekundy".format(czas_trwania))
