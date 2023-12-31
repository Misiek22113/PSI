import numpy as np


class FCL_API:

    def __init__(self, entry):
        self.entry = entry
        self.layers = []
        self.alpha = 0
        self.activation_function = []

    def add_layer(self, n, weight_min_value, weight_max_value, activation_function):
        self.activation_function.append(activation_function)

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

    def ReLU_derivative(self, result_matrix):
        clipped_matrix = np.clip(result_matrix, a_min=0, a_max=None)
        clipped_matrix = np.where(clipped_matrix > 0, 1, clipped_matrix)
        return clipped_matrix

    def predict(self, input):
        for layer in self.layers:
            input = np.dot(layer, input)
            input = self.ReLU(input)
        return input

    def predict_to_layer(self, input, layer):
        for i in range(layer):
            input = np.dot(self.layers[i], input)
            input = self.ReLU(input)
        return input

    def load_weights(self, file_name):
        self.layers.append(np.genfromtxt(file_name, delimiter=','))

    def save_weights(self, file_name):
        for layer in self.layers:
            np.savetxt(file_name, layer, delimiter=',')

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
            res = self.predict(test_input)
            wynik = (res - expected_result).reshape(res.shape[0], 1)
            delta = 2/layer.shape[0] * np.dot(wynik, test_input.T)
            error = 1/layer.shape[0] * (res - expected_result) ** 2
            weight = layer - delta * self.alpha
            self.update_layer(i, weight)

    def fit(self, input_data, expected_result, alpha, rates):
        self.alpha = alpha
        for i in range(rates):
            for column in range(input_data.shape[1]):
                input_column = np.array([input_data[:, column]]).T
                expected_column = np.array([expected_result[:, column]]).T
                layer_output = self.predict(input_column).reshape(-1, 1)
                result = (layer_output - expected_column).reshape(layer_output.shape[0], 1)
                layer_output_delta = 2/expected_result.shape[0] * result
                self.update_weights(layer_output_delta, input_column)

    def update_weights(self, layer_output_delta, input_column):
        layer_output_weight_delta = layer_output_delta * self.predict_to_layer(input_column, len(self.layers) - 1).T

        layer_hidden_delta = layer_output_delta
        for i in range(len(self.layers) - 1, 0, -1):
            layer_result = self.predict_to_layer(input_column, i)
            layer_hidden_delta = np.dot(self.layers[i].T, layer_hidden_delta)
            layer_hidden_delta = layer_hidden_delta * self.ReLU_derivative(layer_result)
            layer_hidden_weight_delta = layer_hidden_delta * input_column.T
            res = self.alpha * layer_hidden_weight_delta
            array = np.array(self.layers[i-1])
            self.update_layer(i-1, array - res)

        self.update_layer(len(self.layers) - 1,
                          self.layers[len(self.layers) - 1] - self.alpha * layer_output_weight_delta)


    def update_layer(self, layer_index, new_weight):
        self.layers[layer_index] = new_weight

    def print_layers(self):
        print(self.layers)