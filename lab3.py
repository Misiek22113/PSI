# ZAD 1
from main import FCL_API, read_input

input = read_input('lab3_zad1_input.txt')

model = FCL_API(3)
model.load_weights('lab3_zad1_h.txt')
model.load_weights('lab3_zad1_y.txt')

print(model.predict(input))

# ZAD 2

expected_result = read_input('lab3_zad1_input.txt')

