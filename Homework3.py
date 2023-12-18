import tensorflow_datasets as tfds
import tensorflow as tf
import matplotlib.pyplot as plt

# Load the CIFAR-10 dataset
(train_ds, test_ds), ds_info = tfds.load(
    'cifar10',
    split=['train', 'test'],
    shuffle_files=True,
    with_info=True,
)

# Printing the dataset info and showing some examples
print(ds_info)
tfds.show_examples(train_ds, ds_info)

# Preprocessing the data
def preprocess(dataset):
    # Convert the images and labels to floating-point numbers
    dataset = dataset.map(lambda x: (tf.cast(x['image'], tf.float32), tf.cast(x['label'], tf.int32)))
    # Flatten the images
    dataset = dataset.map(lambda x, y: (tf.reshape(x, [-1]), y))
    # Normalise the images
    dataset = dataset.map(lambda x, y: (x / 255.0, y))
    # Encode the labels
    dataset = dataset.map(lambda x, y: (x, tf.one_hot(y, 10)))
    # Cache the dataset
    dataset = dataset.cache()
    # Shuffle the dataset
    dataset = dataset.shuffle(1000)
    # Create batches
    dataset = dataset.batch(128)
    # Prefetch the next batch
    dataset = dataset.prefetch(20)
    return dataset

# Define a training loop function
def train_model(model, train_ds, test_ds, loss, optimizer):
    # Compile the model
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    # Train the model
    history = model.fit(train_ds, epochs=10, validation_data=test_ds)
    # Plot the training and validation accuracy
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()
    # Plot the training and validation loss
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    # Evaluate the model
    loss, accuracy = model.evaluate(test_ds)
    print('Test Loss: {}'.format(loss))
    print('Test Accuracy: {}'.format(accuracy))
    return model

# CNN Architecture 1
model1 = tf.keras.Sequential([
    tf.keras.layers.Reshape((32, 32, 3), input_shape=(3072,)),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax'),
])

# CNN Architecture 2
model2 = tf.keras.Sequential([
    tf.keras.layers.Reshape((32, 32, 3), input_shape=(3072,)),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax'),
])

# Define hyperparameters and initialize
loss = tf.keras.losses.CategoricalCrossentropy()
learning_rate1 = 0.001
learning_rate2 = 0.01
optimizer1 = tf.keras.optimizers.Adam(learning_rate=learning_rate1)
optimizer2 = tf.keras.optimizers.Adam(learning_rate=learning_rate2)
optimizer3 = tf.keras.optimizers.SGD(learning_rate=learning_rate1)
optimizer4 = tf.keras.optimizers.SGD(learning_rate=learning_rate2)

# Preprocess the data
train_ds = preprocess(train_ds)
test_ds = preprocess(test_ds)

# Train the models
setup1 = train_model(model1, train_ds, test_ds, loss, optimizer1)
setup2 = train_model(model1, train_ds, test_ds, loss, optimizer2)
setup3 = train_model(model1, train_ds, test_ds, loss, optimizer3)
setup4 = train_model(model1, train_ds, test_ds, loss, optimizer4)
setup5 = train_model(model2, train_ds, test_ds, loss, optimizer1)
setup6 = train_model(model2, train_ds, test_ds, loss, optimizer2)
setup7 = train_model(model2, train_ds, test_ds, loss, optimizer3)
setup8 = train_model(model2, train_ds, test_ds, loss, optimizer4)
