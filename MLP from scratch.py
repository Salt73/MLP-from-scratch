#!/usr/bin/env python
# coding: utf-8

# In[14]:


import random
import numpy as np
import pandas as pd
from tkinter import *

seed_value = 42
random.seed(seed_value)
np.random.seed(seed_value)

data = pd.read_excel('Dry_Bean_Dataset.xlsx')

# Map class labels to numerical values
class_mapping = {'BOMBAY': 0, 'CALI': 1, 'SIRA': 2}
data['Class'] = data['Class'].map(class_mapping)

# Shuffle the data
data = data.sample(frac=1, random_state=42).reset_index(drop=True)

# Split data into train and test sets
X_train = pd.concat([data.iloc[0:30, :-1], data.iloc[50:80, :-1], data.iloc[100:130, :-1]])
y_train = pd.concat([data.iloc[:30, -1], data.iloc[50:80, -1], data.iloc[100:130, -1]])
X_test = pd.concat([data.iloc[30:50, :-1], data.iloc[80:100, :-1], data.iloc[130:, :-1]])
y_test = pd.concat([data.iloc[30:50, -1], data.iloc[80:100, -1], data.iloc[130:, -1]])

# Normalize input data
X_train = (X_train - X_train.mean()) / X_train.std()
X_test = (X_test - X_train.mean()) / X_train.std()

# Convert labels to numpy arrays
y_train_encoded = np.eye(len(class_mapping))[y_train]
y_test_encoded = np.eye(len(class_mapping))[y_test]


#Activation functions and it's derivatives
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Define tanh activation function
def tanh(x):
    return np.tanh(x)

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

def tanh_derivative(x):
    return 1.0 - np.tanh(x) ** 2




def submit():
# Get values after the Submition in the GUI
    hidden_layer_num = int(hidden_layer_entry.get())
    neurons_num = int(neurons_entry.get())
    epochs_num = int(epochs_entry.get())
    eta = float(learning_rate_entry.get())
    bias_state = bias_var.get()
    activation_function_value = Activation_function.get()

#Selecting activation function based on the user's choice
    if activation_function_value == 'Sigmoid function':
        activation_function = sigmoid
        activation_derivative = sigmoid_derivative
    elif activation_function_value == 'Tanh function':
        activation_function = tanh
        activation_derivative = tanh_derivative
    else:
        activation_function = None
        activation_derivative = None

# Check if activation function is selected
    if activation_function is not None:
        input_size = X_train.shape[1]
        hidden_size = neurons_num
        output_size = len(class_mapping)

# Initializing weights and biases using Xavier/Glorot method
        weights_input_hidden = np.random.randn(input_size, hidden_size) / np.sqrt(input_size)
        bias_hidden = np.zeros((1, hidden_size))
        weights_hidden_output = np.random.randn(hidden_size, output_size) / np.sqrt(hidden_size)
        bias_output = np.zeros((1, output_size))
#training the model
        for epoch in range(epochs_num):
# Forward propagation
            hidden_layer_input = np.dot(X_train, weights_input_hidden) + bias_hidden
            hidden_layer_output = activation_function(hidden_layer_input)

            output_layer_input = np.dot(hidden_layer_output, weights_hidden_output) + bias_output
            output_layer_output = activation_function(output_layer_input)

# Backpropagation
            error = y_train_encoded - output_layer_output
            d_output = error * activation_derivative(output_layer_output)

            error_hidden = d_output.dot(weights_hidden_output.T)
            d_hidden = error_hidden * activation_derivative(hidden_layer_output)

# Update weights and biases
            weights_hidden_output += hidden_layer_output.T.dot(d_output) * eta
            bias_output += np.sum(d_output, axis=0, keepdims=True) * eta
            weights_input_hidden += X_train.T.dot(d_hidden) * eta
            bias_hidden += np.sum(d_hidden, axis=0, keepdims=True) * eta

# Forward propagation for the test set
        hidden_layer_input_test = np.dot(X_test, weights_input_hidden) + bias_hidden
        hidden_layer_output_test = activation_function(hidden_layer_input_test)
        output_layer_input_test = np.dot(hidden_layer_output_test, weights_hidden_output) + bias_output
        output_layer_output_test = activation_function(output_layer_input_test)

# Convert the predicted probabilities to class labels
        predicted_labels = np.argmax(output_layer_output_test, axis=1)

#CONFUSION MATRIX
        num_classes = len(np.unique(y_test))
        conf_matrix = np.zeros((num_classes, num_classes), dtype=int)

        for true_label, predicted_label in zip(y_test, predicted_labels):
            conf_matrix[true_label, predicted_label] += 1

        print("Confusion Matrix:")
        for row in conf_matrix:
            print(" ".join(map(str, row)))
# ACCURACY
        accuracy = np.sum(predicted_labels == y_test) / len(y_test) * 100

        print(f"Accuracy: {accuracy:.2f}%")
    else:
        print("Please select an activation function.")

# GUI code
window = Tk()
window.title("MLP")

hidden_layer_label = Label(window, text="Number of Hidden Layers:")
hidden_layer_label.pack()
hidden_layer_entry = Entry(window)
hidden_layer_entry.pack()

neurons_label = Label(window, text="Number of Neurons:")
neurons_label.pack()
neurons_entry = Entry(window)
neurons_entry.pack()

epochs_label = Label(window, text="Number of Epochs:")
epochs_label.pack()
epochs_entry = Entry(window)
epochs_entry.pack()

learning_rate_label = Label(window, text="Learning Rate:")
learning_rate_label.pack()
learning_rate_entry = Entry(window)
learning_rate_entry.pack()

bias_var = IntVar()
bias_checkbox = Checkbutton(window, text="Add Bias", variable=bias_var)
bias_checkbox.pack()

Activation_function = StringVar()
ActivationFunction = OptionMenu(window, Activation_function, *['Sigmoid function', 'Tanh function'])
ActivationFunction.pack(fill=X, padx=5, pady=1)

submit_button = Button(window, text="Submit", command=submit)
submit_button.pack()

window.mainloop()

