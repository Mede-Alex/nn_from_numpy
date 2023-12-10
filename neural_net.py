import numpy as np
from read_data import read_sample, display_img

RANDOM_SEED = 42
NO_SAMPLES = 1000
INPUT_NEURONS = 784
HIDDEN_LAYERS = 2
HIDDEN_NEURONS = 16
OUTPUT_NEURONS = 10

np.random.seed(RANDOM_SEED)

def initialize_weights(first_layer_size, second_layer_size):
    """Initialize random weights between layers"""
    return np.random.normal(size=(second_layer_size, first_layer_size))

def initialize_biases(layer_size):
    """Initialize random biases for layer"""
    return np.random.normal(size=(layer_size, ))

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def _loss(y_true, y_predicted):
    return -np.dot(y_true, np.log2(y_predicted))

imgs, labels = read_sample(NO_SAMPLES)

# one training example
img = imgs[0].reshape(INPUT_NEURONS)
label = labels[0]

w01 = initialize_weights(INPUT_NEURONS, HIDDEN_NEURONS)
w12 = initialize_weights(HIDDEN_NEURONS, HIDDEN_NEURONS)
w23 = initialize_weights(HIDDEN_NEURONS, OUTPUT_NEURONS)

b1 = initialize_biases(HIDDEN_NEURONS)
b2 = initialize_biases(HIDDEN_NEURONS)
b3 = initialize_biases(OUTPUT_NEURONS)

#forward prop
layer_1_values = sigmoid(np.dot(w01, img) + b1)
layer_2_values = sigmoid(np.dot(w12, layer_1_values) + b2)
layer_3_values = sigmoid(np.dot(w23, layer_2_values) + b3)

#calculate loss
print("True value")
y_true = [0 if val != label else 1 for val in (range(OUTPUT_NEURONS))]
print(y_true) 
print("Predicted value")
print(layer_3_values)
loss = _loss(y_true, layer_3_values)
print(loss) 

#backward prop
