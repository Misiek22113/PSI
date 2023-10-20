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

class FCL_API:

    def __init__(self, entry):
        self.entry = entry
        self.layers = []

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

    def predict(self, input):
        for layer in self.layers:
            input = np.dot(layer, input)
        return input.T

    def load_weights(self, file_name):
        self.layers.append(np.genfromtxt(file_name, delimiter=','))

    def learn_model(self, test_input, expected_result, alpha, rates):
        weight = self.layers[0]
        neuron_count = self.layers[0].shape[0]
        for i in range(rates):
            self.learn_single_neuron(test_input, expected_result, alpha, weight, i, neuron_count)

    def learn_single_neuron(self, test_input, expected_result, alpha, weight, epoch, neuron_count):
        size = test_input.shape[1]
        for i in range(size):
            input_column = test_input[:, i]
            expected_result_column = expected_result[:, i]
            res = self.predict(input_column)
            wynik = (res - expected_result_column).reshape(res.shape[0], 1)
            delta = 2/neuron_count * np.dot(wynik, input_column.reshape(1, input_column.shape[0]))
            error = 1/neuron_count * (res - expected_result_column) ** 2

            print("epoka: ", epoch, "\nseria: ", i + 1, "\nwaga: ", weight, "\nerror: ", np.sum(error), "\nwynik: ",
                  res.T, "\n")

            weight = weight - delta * alpha
            print("nowa waga: ", weight, "\n")
            self.update_layer(0, weight)

    def update_layer(self, layer, newLayer):
        self.layers[layer] = newLayer

    def printLayers(self):
        print(self.layers)

# test = FCL_API(1)
# test.add_layer(1, 0.5, 0.5)
# test.printLayers()
# test.learning_rate(np.array([2]), 0.8, 0.1, 20)

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
# model.learn_model(test_input, expected_result, 0.01, 2)

# ZAD 3


def build_expected_result(data):
    matrix = np.array([[0.0], [0.0], [0.0], [0.0]])
    matrix[int(data[0]) - 1] = 1
    return matrix


def split_data(data):

    elements = data.strip().split()
    matrix = [[float(element)] for element in elements]
    last_element = matrix[-1]
    expected_matrix = build_expected_result(last_element)
    test_input = np.array(matrix[:-1])
    expected_index = last_element[0] - 1
    return test_input, expected_matrix, expected_index


def rgb(training_file, test_file):
    model = FCL_API(3)
    model.add_layer(4, 0, 1)
    correct_answers = 0
    lines = 0

    with open(training_file, 'r') as file:
        for line in file:
            test_input, expected_matrix, expected_index = split_data(line)
            model.learn_model(test_input, expected_matrix, 0.11, 20)

    with open(test_file, 'r') as file:
        for line in file:
            lines += 1
            test_input, expected_matrix, expected_index = split_data(line)
            result = model.predict(test_input)
            index_max = np.argmax(result)
            cords_max = np.unravel_index(index_max, result.shape)
            cords_max = cords_max[1]
            if cords_max == expected_index:
                correct_answers += 1
            print('cords', cords_max, '\n', 'result', result, '\n', 'expected index', expected_index, '\n', 'expected matrix', expected_matrix.T, '\n')

    print('percentage', correct_answers/lines)

# rgb('training.txt', 'test.txt')
