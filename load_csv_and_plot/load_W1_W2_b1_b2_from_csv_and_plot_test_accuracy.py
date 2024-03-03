import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import time


data = pd.read_csv('/home/pi/Desktop/mnist_train.csv')

data = np.array(data)
m, n = data.shape
np.random.shuffle(data) # shuffle before splitting into dev and training sets

data_dev = data[0:1000].T
Y_dev = data_dev[0]
X_dev = data_dev[1:n]
X_dev = X_dev / 255.

data_train = data[1000:m].T
Y_train = data_train[0]
X_train = data_train[1:n]
X_train = X_train / 255.
_,m_train = X_train.shape

Y_train

X_train = X_train.T  # Transpose X_train
X_dev = X_dev.T  # Transpose X_dev


# Section 1: Load and visualize weights from W1_output84.csv
def init_params_from_csv_W1():
    # Load weights from CSV files
    W1_data = []
    with open('W1_output84.csv', 'r') as file:
        for line in file:
            if line.strip().startswith('Weights'):
                continue  # Skip the line if it starts with 'neuron'
            W1_data.append(list(map(float, line.strip().split(','))))
    W1 = np.array(W1_data)
    
    # Divide into 10 separate weights
    weights1 = []
    for i in range(10):
        weight = W1[i*28:(i+1)*28, :]
        weights1.append(weight)
    
    return weights1

# Call the function to load parameters from CSV files for W1
weights1 = init_params_from_csv_W1()

# Visualize the weights for W1
plt.figure(figsize=(15, 7))
for i, weight1 in enumerate(weights1):
    plt.subplot(2, 5, i + 1)
    plt.imshow(weight1, cmap='gray')
    plt.title(f'Weight {i}')
    plt.axis('off')
plt.suptitle('Visualization of Weights')
plt.show()

# Section 2: Load and visualize weights from W2_output84.csv
def init_params_from_csv_W2():
    # Load weights from CSV files
    weights_data = {}
    with open('W2_output84.csv', 'r') as file:
        current_neuron = None
        for line in file:
            if line.strip().startswith('Weights for neuron'):
                current_neuron = int(line.strip().split()[-1])
                weights_data[current_neuron] = []
            else:
                weights_data[current_neuron].append(float(line.strip()))
    
    # Convert to numpy arrays
    weights2 = np.array([weights_data[i] for i in range(10)])
    
    return weights2

# Call the function to load parameters from CSV files for W2
weights2 = init_params_from_csv_W2()

# Visualize the weights for W2
plt.figure(figsize=(15, 7))
for i, weights_neuron in enumerate(weights2):
    plt.subplot(2, 5, i + 1)
    plt.plot(weights_neuron)
    plt.title(f'Neuron {i}')
    plt.xlabel('Weight Index')
    plt.ylabel('Weight Value')
plt.suptitle('Visualization of Weights W2')
plt.tight_layout()
plt.show()

# Section 3: Load and visualize biases from b1_output84.csv
def init_params_from_csv_b1():
    # Load biases from CSV file
    biases1_data = []
    with open('b1_output84.csv', 'r') as file:
        for line in file:
            if line.strip().startswith('Biases'):
                continue  # Skip the line if it starts with 'Biases'
            biases1_data.append(float(line.strip()))
    
    # Divide into 10 separate biases
    biases1 = []
    for i in range(10):
        bias1 = biases1_data[i]
        biases1.append(bias1)
    
    return biases1

# Call the function to load biases from CSV file for b1
biases1 = init_params_from_csv_b1()

# Visualize the biases for b1
plt.figure(figsize=(10, 5))
plt.bar(range(len(biases1)), biases1, color='skyblue')
plt.xlabel('Neuron')
plt.ylabel('Bias Value')
plt.title('Visualization of Biases b1')
plt.show()

# Section 4: Load and visualize biases from b2_output84.csv
def init_params_from_csv_b2():
    # Load biases from CSV file
    biases2_data = []
    with open('b2_output84.csv', 'r') as file:
        for line in file:
            if line.strip().startswith('Biases'):
                continue  # Skip the line if it starts with 'Biases'
            biases2_data.append(float(line.strip()))
    
    # Divide into 10 separate biases
    biases2 = []
    for i in range(10):
        bias2 = biases2_data[i]
        biases2.append(bias2)
    
    return biases2

# Call the function to load biases from CSV file for b2
biases2 = init_params_from_csv_b2()

# Visualize the biases for b2
plt.figure(figsize=(10, 5))
plt.bar(range(len(biases2)), biases2, color='skyblue')
plt.xlabel('Neuron')
plt.ylabel('Bias Value')
plt.title('Visualization of Biases b2')
plt.show()


def ReLU(Z):
    return np.maximum(Z, 0)

def softmax(Z):
    A = np.exp(Z) / sum(np.exp(Z))
    return A
'''    
def forward_prop(W1, b1, W2, b2, X):
    
    Z1 = W1.dot(X) + b1
    #Z1 = np.dot(W1, X.T) + b1[:, np.newaxis]
    
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2
'''
def forward_prop(W1, b1, W2, b2, X):
    Z1 = np.dot(W1.reshape(10, -1), X.T) + b1[:, np.newaxis] #flatten 28x28 to 784
    A1 = ReLU(Z1)
    Z2 = np.dot(W2, A1) + b2[:, np.newaxis]
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2

def ReLU_deriv(Z):
    return Z > 0

def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y

def get_predictions(A2):
    return np.argmax(A2, 0)

'''
# Load weights and biases
W1 = init_params_from_csv_W1()
b1 = init_params_from_csv_b1()
W2 = init_params_from_csv_W2()
b2 = init_params_from_csv_b2()
'''
# Load weights and biases
W1 = np.array(init_params_from_csv_W1())
b1 = np.array(init_params_from_csv_b1())
W2 = np.array(init_params_from_csv_W2())
b2 = np.array(init_params_from_csv_b2())

def get_accuracy(predictions, Y):
    print(predictions, Y)
    return np.sum(predictions == Y) / Y.size

def make_predictions(X, W1, b1, W2, b2):
    _, _, _, A2 = forward_prop(W1, b1, W2, b2, X)
    predictions = get_predictions(A2)
    return predictions

def test_prediction(index, W1, b1, W2, b2):
    current_image = X_train[:, index, None]
    prediction = make_predictions(X_train[:, index, None], W1, b1, W2, b2)
    label = Y_train[index]
    print("Prediction: ", prediction)
    print("Label: ", label)
    
    current_image = current_image.reshape((28, 28)) * 255
    plt.gray()
    plt.imshow(current_image, interpolation='nearest')
    plt.show()


