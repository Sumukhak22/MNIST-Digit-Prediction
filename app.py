import streamlit as st
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# Load and prepare the data
data = pd.read_csv('train.csv')
data = np.array(data)

data_dev = data[0:1000].T
Y_dev = data_dev[0]
X_dev = data_dev[1:]
X_dev = X_dev / 255.

data_train = data[1000:].T
Y_train = data_train[0]
X_train = data_train[1:]
X_train = X_train / 255.
_, m_train = X_train.shape

# Initialize parameters
def init_params():
    W1 = np.random.rand(10, 784) - 0.5
    b1 = np.random.rand(10, 1) - 0.5
    W2 = np.random.rand(10, 10) - 0.5
    b2 = np.random.rand(10, 1) - 0.5
    return W1, b1, W2, b2

# Activation functions
def ReLU(Z):
    return np.maximum(Z, 0)

def softmax(Z):
    A = np.exp(Z) / np.sum(np.exp(Z), axis=0)
    return A

# Forward propagation
def forward_prop(W1, b1, W2, b2, X):
    Z1 = W1.dot(X) + b1
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2

# Derivative of ReLU
def ReLU_deriv(Z):
    return Z > 0

# One-hot encoding
def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y

# Backward propagation
def backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y):
    m = Y.size
    one_hot_Y = one_hot(Y)
    dZ2 = A2 - one_hot_Y
    dW2 = 1 / m * dZ2.dot(A1.T)
    db2 = 1 / m * np.sum(dZ2, axis=1, keepdims=True)
    dZ1 = W2.T.dot(dZ2) * ReLU_deriv(Z1)
    dW1 = 1 / m * dZ1.dot(X.T)
    db1 = 1 / m * np.sum(dZ1, axis=1, keepdims=True)
    return dW1, db1, dW2, db2

# Update parameters
def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    W1 -= alpha * dW1
    b1 -= alpha * db1    
    W2 -= alpha * dW2  
    b2 -= alpha * db2    
    return W1, b1, W2, b2

# Get predictions
def get_predictions(A2):
    return np.argmax(A2, axis=0)

# Get accuracy
def get_accuracy(predictions, Y):
    return np.sum(predictions == Y) / Y.size

# Gradient descent
def gradient_descent(X, Y, alpha, iterations):
    W1, b1, W2, b2 = init_params()
    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2 = backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y)
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
        if i % 10 == 0:
            accuracy = get_accuracy(get_predictions(A2), Y)
            print(f"Iteration: {i}, Accuracy: {accuracy}")
    return W1, b1, W2, b2

# Train the model
W1, b1, W2, b2 = gradient_descent(X_train, Y_train, 0.10, 500)

# Make predictions
def make_predictions(X, W1, b1, W2, b2):
    _, _, _, A2 = forward_prop(W1, b1, W2, b2, X)
    predictions = get_predictions(A2)
    return predictions

# Test prediction for a specific digit
def test_prediction_for_digit(digit, W1, b1, W2, b2, data):
    indices = np.where(data[:, 0] == digit)[0]
    if len(indices) == 0:
        st.write(f"No examples found for digit {digit}.")
        return
    
    index = indices[0]  # Taking the first example of the digit
    current_image = data[index, 1:].reshape(-1, 1) / 255.0
    prediction = make_predictions(current_image, W1, b1, W2, b2)
    label = data[index, 0]
    st.write(f"Prediction: {prediction[0]}, Label: {label}")
    
    current_image = current_image.reshape((28, 28)) * 255
    plt.figure()
    plt.gray()
    plt.imshow(current_image, interpolation='nearest')
    st.pyplot(plt)

# Streamlit app
st.title("Digital Handwriting Prediciton")
st.write("This app predicts digits from the MNIST dataset using a neural network.")

st.subheader("Prediction")
digit = st.text_input("Enter the digit to predict (0-9)", value="0")
if st.button("Predict"):
    try:
        digit = int(digit)
        if 0 <= digit <= 9:
            test_prediction_for_digit(digit, W1, b1, W2, b2, data)
        else:
            st.write("Please enter a digit between 0 and 9.")
    except ValueError:
        st.write("Please enter a valid integer digit.")

st.write("Testing on development set...")
dev_predictions = make_predictions(X_dev, W1, b1, W2, b2)
st.write(f"Development set accuracy: {get_accuracy(dev_predictions, Y_dev)}")
