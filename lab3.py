import numpy as np
from main import FCL_API, read_input, split_data

# ZAD 1

input = read_input('lab3_zad1_input.txt')

model = FCL_API(3)
model.load_weights('lab3_zad1_h.txt')
model.load_weights('lab3_zad1_y.txt')

# print(model.predict(input))

# ZAD 2

# expected_result = read_input('lab3_zad2_expected.txt')

# ZAD 3

model_3 = FCL_API(784)
model_3.add_layer(40, -1, 1, 'relu')
model_3.add_layer(10, -1, 1, 'relu')

def read_mnist_images(filename):
    with open(filename, 'rb') as f:
        magic = int.from_bytes(f.read(4), byteorder='big')
        num_images = int.from_bytes(f.read(4), byteorder='big')
        rows = int.from_bytes(f.read(4), byteorder='big')
        cols = int.from_bytes(f.read(4), byteorder='big')
        data = np.fromfile(f, dtype=np.uint8)
    images = data.reshape(num_images, rows, cols)
    return images

def read_mnist_labels(filename):
    with open(filename, 'rb') as f:
        magic = int.from_bytes(f.read(4), byteorder='big')
        num_items = int.from_bytes(f.read(4), byteorder='big')
        labels = np.fromfile(f, dtype=np.uint8)
    return labels

def create_expected_matrix(index):
    expected_matrix = np.array([[0], [0], [0], [0], [0], [0], [0], [0], [0], [0]])
    expected_matrix[index][0] = 1
    return expected_matrix

def train_on_mnist():
    training_input = read_mnist_images('train-images-idx3-ubyte')
    training_expected = read_mnist_labels('train-labels-idx1-ubyte')

    for i in range(training_input.shape[0]):
        training_single_expected = create_expected_matrix(training_expected[i])
        if i % 1000 == 0:
            print(i)
        model_3.fit(training_input[i].reshape(-1, 1) / 255, training_single_expected, 0.01, 10)

    test_input = read_mnist_images('t10k-images-idx3-ubyte')
    test_expected = read_mnist_labels('t10k-labels-idx1-ubyte')

    correct_answers = 0
    lines = 0
    for i in range(test_input.shape[0]):
        res = model_3.predict(test_input[i].reshape(-1, 1))
        if np.argmax(res) == test_expected[i]:
            correct_answers += 1
        lines += 1

    print('percentage', correct_answers/lines)

train_on_mnist()

# ZAD 4

def rgb(training_file, test_file):
    model = FCL_API(3)
    model.add_layer(11, -1, 1, 'relu')
    model.add_layer(4, -1, 1, 'relu')
    correct_answers = 0
    lines = 0

    training_input, training_expected_matrix = split_data(training_file)
    for i in range(training_input.shape[0]):
        model.fit(np.array(training_input[i, :]).reshape(-1, 1),
                          np.array(training_expected_matrix[i, :]).reshape(-1, 1), 0.01, 15)

    test_input, test_expected_matrix = split_data(test_file)
    for i in range(test_input.shape[0]):
        result = model.predict(np.array(test_input[i, :]).reshape(-1, 1)).reshape(-1, 1)
        if np.argmax(result) == np.argmax(test_expected_matrix[i, :]):
            correct_answers += 1
        lines += 1

    print('percentage', correct_answers/lines)

# rgb('training.txt', 'test.txt')