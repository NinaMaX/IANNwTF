import tensorflow_datasets as tfds
import tensorflow as tf
import matplotlib.pyplot as plt

# Load the MNIST dataset
(train_ds, test_ds), ds_info=tfds.load('mnist', split=['train', 'test'], as_supervised=True, with_info=True)

# Print the dataset info and showing some examples
print(ds_info)
tfds.show_examples(train_ds, ds_info)

# Preprocess the dataset
def preprocess(dataset, batch):
    # Convert the images and labels to floating-point numbers
    dataset = dataset.map(lambda x, y: (tf.cast(x, tf.float32), y))
    # Flatten the images
    dataset = dataset.map(lambda x, y: (tf.reshape(x, [-1]), y))
    # Normalise the images
    dataset = dataset.map(lambda x, y: ((x / 128) - 1, y))
    # Encode the labels
    dataset = dataset.map(lambda x, y: (x, tf.one_hot(y, 10)))
    # Cache the dataset
    dataset = dataset.cache()
    # Shuffle the dataset
    dataset = dataset.shuffle(1000)
    # Create batches of 32
    dataset = dataset.batch(batch)
    # Prefetch the next batch
    dataset = dataset.prefetch(20)
    return dataset

# Define a training loop function
def train_model(epochs, model, train_ds, test_ds, loss, optimizer, metric):
    # Compile the model
    model.compile(loss=loss, optimizer=optimizer, metrics=[metric])
    # Train the model
    history = model.fit(train_ds, epochs=epochs, validation_data=test_ds)
    return history

# Build a fully connected feed-forward neural network
model = tf.keras.Sequential([
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Define hyperparameters and initialize
epochs = 10
learning_rate = 0.001
loss = tf.keras.losses.CategoricalCrossentropy()
optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)

# Preprocess the training and test datasets
train_ds = train_ds.apply(lambda x: preprocess(x, 32))
test_ds = test_ds.apply(lambda x: preprocess(x, 32))

# Train the model
history = train_model(epochs, model, train_ds, test_ds, loss, optimizer, 'accuracy')

# Plot the training and validation accuracy
def visualization(train_losses, train_accuracies, test_losses, test_accuracies):
    plt.figure()
    line1, = plt.plot(train_losses, "b-")
    line2, = plt.plot(test_losses, "r-")
    line3, = plt.plot(train_accuracies, "b:")
    line4, = plt.plot(test_accuracies, "r:")
    plt.xlabel("Training steps")
    plt.ylabel("Loss / Accuracy")
    plt.legend((line1, line2, line3, line4), ("training loss", "test loss", "train accuracy", "test accuracy"))
    plt.show()

visualization(history.history['loss'], history.history['accuracy'], history.history['val_loss'], history.history['val_accuracy'])