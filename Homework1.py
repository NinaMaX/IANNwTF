from sklearn.datasets import load_digits
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder

# Extract the data into (input, target) tuples
input, target = load_digits(return_X_y=True)

# Plotting input as images
def plot_images(input, num_samples=5):
    # Plot some example images
    for i in range(num_samples):
        plt.subplot(1, num_samples, i + 1)
        plt.imshow(input[i].reshape(8, 8), cmap='gray')
        plt.axis('off')
    plt.show()

plot_images(input)

# Represent input as float32 values within [0 to 1]
input = input.astype(np.float32) / 16

# One-hot encode the target digits
target = np.eye(10)[target]

# Generator function
def generate_minibatches(input, target, minibatchsize):
    # Shuffle the data
    indices = np.arange(len(input))
    np.random.shuffle(indices)
    input = input[indices]
    target = target[indices]
    # Create minibatches
    for i in range(0, len(input), minibatchsize):
        yield input[i:i + minibatchsize], target[i:i + minibatchsize]

# Sigmoid function
class Sigmoid:
    def __call__(self, input):
        self.input = input
        return 1 / (1 + np.exp(-input))

    def backward(self, output_gradient, learning_rate):
        sigmoid = self.__call__(self.input)
        return output_gradient * sigmoid * (1 - sigmoid)

# Softmax function
class Softmax:
    def __call__(self, input):
        self.input = input
        exp = np.exp(input - np.max(input, axis=1, keepdims=True))
        return exp / np.sum(exp, axis=1, keepdims=True)

    def backward(self, output_gradient, learning_rate):
        softmax = self.__call__(self.input)
        return output_gradient * softmax * (1 - softmax)

# Layer class
class Layer:
    def __init__(self, input_size, output_size, activation):
        self.activation = activation
        self.w = np.random.randn(input_size, output_size) * np.sqrt(2.0 / (input_size + output_size))
        self.b = np.zeros(output_size)
        self.input = None
        self.z = None

    def __call__(self, input):
        self.input = input
        self.z = np.dot(input, self.w) + self.b
        return self.activation(self.z)
    
    def weights_backward(self, output_gradient, learning_rate):
        input_gradient = np.dot(output_gradient, self.w.T)
        output_gradient_wrt_z = output_gradient * self.activation(self.z) * (1 - self.activation(self.z))
        self.w -= learning_rate * np.dot(self.input.T, output_gradient_wrt_z)
        self.b -= learning_rate * np.sum(output_gradient_wrt_z, axis=0)
        return input_gradient
    
    def backward(self, output_gradient, learning_rate):
        return self.weights_backward(self.activation.backward(output_gradient, learning_rate), learning_rate)

# MLP
class MLP:
    def __init__(self, input_size, num_classes, hidden_sizes):
        self.layers = []
        # Add hidden layers
        for i in range(len(hidden_sizes)):
            if i == 0:
                self.layers.append(Layer(input_size, hidden_sizes[i], Sigmoid()))
            else:
                self.layers.append(Layer(hidden_sizes[i - 1], hidden_sizes[i], Sigmoid()))
        # Add output layer
        self.layers.append(Layer(hidden_sizes[-1], num_classes, Softmax()))

    def __call__(self, input):
        for layer in self.layers:
            input = layer(input)
        return input

    def backward(self, output_gradient, learning_rate):
        for layer in reversed(self.layers):
            output_gradient = layer.backward(output_gradient, learning_rate)

# Categorical cross-entropy loss function
class CrossEntropyLoss:
    def __call__(self, predictions, target):
        self.predictions = predictions
        self.target = target
        return -np.sum(target * np.log(predictions + 1e-15)) / len(predictions)

    def backward(self):
        return (self.predictions - self.target) / len(self.target)

# Train network function
def train_network(mlp, input, target, minibatch_size, num_epochs, learning_rate):
    loss_function = CrossEntropyLoss()
    losses = []

    for epoch in range(num_epochs):
        total_loss = 0
        num_batches = 0

        for minibatch_input, minibatch_target in generate_minibatches(input, target, minibatch_size):
            # Forward pass
            predictions = mlp(minibatch_input)

            # Compute loss
            loss = loss_function(predictions, minibatch_target)
            total_loss += loss
            num_batches += 1

            # Backward pass
            loss_gradient = loss_function.backward()
            mlp.backward(loss_gradient, learning_rate)

        # Calculate average loss for the epoch
        average_loss = total_loss / num_batches
        losses.append(average_loss)

        print(f"Epoch {epoch + 1}/{num_epochs}, Average Loss: {average_loss}")

    # Plot the average loss versus the epoch number
    plt.plot(range(1, num_epochs + 1), losses)
    plt.xlabel('Epoch')
    plt.ylabel('Average Loss')
    plt.title('Average Loss vs. Epoch')
    plt.show()

input_size = 64
num_classes = 10
hidden_sizes = [32, 16]
mlp = MLP(input_size, num_classes, hidden_sizes)

minibatch_size = 32
num_epochs = 1000
learning_rate = 0.3

train_network(mlp, input, target, minibatch_size, num_epochs, learning_rate)
