import random
import numpy as np

# from keras.model import Sequential
# from keras.layers import Dense

# zad 1

# input = np.array([1, 2, 3])

weights = np.array(
    [random.randrange(-1, 1),
     random.randrange(-1, 1),
     random.randrange(-1, 1)]
)


def neuron(input, weights, bias):
    print(np.dot(input, weights) + bias)


# neuron(input, weights, 1)

# zad 2

weights_2x2 = np.random.rand(5, 3)
input_2x2 = np.array([[1], [2], [3]])

example_weights = [[0.1, 0.1, -0.3],
                   [0.1, 0.2, 0.0],
                   [0.0, 0.7, 0.1],
                   [0.2, 0.4, 0.0],
                   [-0.3, 0.5, 0.1]]


# example_input = np.array([[0.5], [0.75], [0.1]])


def neural_network(input, weights, bias):
    res = np.dot(weights, input) + bias
    return res


# neural_network(input_2x2, weights_2x2, 1)

# zad 3

example_weights_hidden_layer = np.array([[0.1, 0.1, -0.3],
                                         [0.1, 0.2, 0.0],
                                         [0.0, 0.7, 0.1],
                                         [0.2, 0.4, 0.0],
                                         [-0.3, 0.5, 0.1]])

example_weights_exit_layer = np.array([[0.7, 0.9, -0.4, 0.8, 0.1],
                                       [0.8, 0.5, 0.3, 0.1, 0.0],
                                       [-0.3, 0.9, 0.3, 0.1, -0.2]])

example_imput_3 = np.array([[0.5],
                            [0.75],
                            [0.1]])


def deep_neural_network(input, first_weights, second_weights):
    hidden_layer = neural_network(input, first_weights, 0)
    exit_layer = neural_network(hidden_layer, second_weights, 0)
    return exit_layer


# print(deep_neural_network(example_imput_3, example_weights_hidden_layer, example_weights_exit_layer))

# zad 4

def read_input(file_name):
    return np.genfromtxt(file_name, delimiter=',')


class FCL_API:

    def __init__(self, entry):
        self.entry = entry
        self.layers = []
        self.alpha = 0

    def add_layer(self, n, weight_min_value, weight_max_value):

        if weight_max_value is not None and weight_min_value is not None:
            start_range = weight_min_value
            end_range = weight_max_value
        else:
            start_range = -1
            end_range = 1

        if len(self.layers) == 0:
            self.layers.append(np.around(np.random.uniform(start_range, end_range, (n, self.entry)), 2))
        else:
            self.layers.append(np.around(np.random.uniform(start_range, end_range, (n, self.layers[-1].shape[0])), 2))

    def ReLU(self, result_matrix):
        clipped_matrix = np.clip(result_matrix, a_min=0, a_max=None)
        return clipped_matrix

    def predict(self, input):
        for layer in self.layers:
            input = np.dot(layer, input)
            input = self.ReLU(input)
        return input

    def load_weights(self, file_name):
        self.layers.append(np.genfromtxt(file_name, delimiter=','))

    def learn_model(self, test_input, expected_result, alpha, rates):
        self.alpha = alpha
        for i in range(rates):
            self.learn_neurons(test_input, expected_result)


    def learn_neurons(self, test_input, expected_result):
        if test_input.shape[1] > 1:
            for i in range(test_input.shape[1]):
                self.learn_single_neuron(np.array([test_input[:, i]]).T, np.array([expected_result[:, i]]).T)
        else:
            for i in range(test_input.shape[1]):
                self.learn_single_neuron(test_input, expected_result)

    def learn_single_neuron(self, test_input, expected_result):
        i = 0
        for layer in self.layers:
            res = self.predict(test_input).reshape(-1, 1)
            wynik = (res - expected_result).reshape(res.shape[0], 1)
            delta = 2/layer.shape[0] * np.dot(wynik, test_input.T)
            error = 1/layer.shape[0] * (res - expected_result) ** 2
            weight = layer - delta * self.alpha
            self.update_layer(i, weight)


    def update_layer(self, layer_index, new_weight):
        self.layers[layer_index] = new_weight

    def print_layers(self):
        print(self.layers)

# test = FCL_API(1)
# test.add_layer(1, 0.5, 0.5)
# test.printLayers()
# test.learn_model(np.array([[2]]), np.array([[0.8]]), 0.1, 20)

# zad 2

expected_result = np.array([[0.1, 0.5, 0.1, 0.7],
                            [1.0, 0.2, 0.3, 0.6],
                            [0.1, -0.5, 0.2, 0.2],
                            [0.0, 0.3, 0.9, -0.1],
                            [-0.1, 0.7, 0.1, 0.8]])

test_input = np.array([[0.5, 0.1, 0.2, 0.8],
                       [0.75, 0.3, 0.1, 0.9],
                       [0.1, 0.7, 0.6, 0.2]])

# model = FCL_API(3)
# model.load_weights('weights2.txt')
# model.learn_model(test_input, expected_result, 0.01, 10)

# ZAD 3

def split_data(filename):
    test_input = np.genfromtxt(filename, delimiter=" ", usecols=range(0, 3))
    expected_data = np.genfromtxt(filename, delimiter=" ", usecols=range(3, 4))
    expected_matrix = np.zeros((expected_data.shape[0], 4))
    for i in range(int(expected_data.shape[0])):
        expected_matrix[i, int(expected_data[i]) - 1] = 1
    return test_input, expected_matrix


def rgb(training_file, test_file):
    model = FCL_API(3)
    model.add_layer(4, 0, 1)
    correct_answers = 0
    lines = 0

    training_input, training_expected_matrix = split_data(training_file)
    for i in range(training_input.shape[0]):
        model.learn_model(np.array(training_input[i, :]).reshape(-1, 1),
                          np.array(training_expected_matrix[i, :]).reshape(-1, 1), 0.01, 10)

    test_input, test_expected_matrix = split_data(test_file)
    for i in range(test_input.shape[0]):
        result = model.predict(np.array(test_input[i, :]).reshape(-1, 1)).reshape(-1, 1)
        if np.argmax(result) == np.argmax(test_expected_matrix[i, :]):
            correct_answers += 1
        lines += 1

    print('percentage', correct_answers/lines)

# rgb('training.txt', 'test.txt')
