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
class Architecture1(tf.keras.Model):
    def __init__(self):
        super(Architecture1, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(32, 3, activation='relu')
        self.batch1 = tf.keras.layers.BatchNormalization()
        self.pool1 = tf.keras.layers.MaxPool2D()
        self.conv2 = tf.keras.layers.Conv2D(64, 3, activation='relu')
        self.batch2 = tf.keras.layers.BatchNormalization()
        self.pool2 = tf.keras.layers.MaxPool2D()
        self.conv3 = tf.keras.layers.Conv2D(128, 3, activation='relu')
        self.batch3 = tf.keras.layers.BatchNormalization()
        self.pool3 = tf.keras.layers.MaxPool2D()
        self.flatten = tf.keras.layers.Flatten()
        self.fc1 = tf.keras.layers.Dense(128, activation='relu')
        self.dropout = tf.keras.layers.Dropout(0.5)
        self.fc2 = tf.keras.layers.Dense(10, activation='softmax')

    def call(self, x):
        x = tf.reshape(x, [-1, 32, 32, 3])
        x = self.conv1(x)
        x = self.batch1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.batch2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.batch3(x)
        x = self.pool3(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x
    
# CNN Architecture 2 
class Architecture2(tf.keras.Model):
    def __init__(self):
        super(Architecture2, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(32, 3, activation='relu')
        self.batch1 = tf.keras.layers.BatchNormalization()
        self.pool1 = tf.keras.layers.AveragePooling2D()
        self.conv2 = tf.keras.layers.Conv2D(64, 3, activation='relu')
        self.batch2 = tf.keras.layers.BatchNormalization()
        self.pool2 = tf.keras.layers.AveragePooling2D()
        self.conv3 = tf.keras.layers.Conv2D(128, 3, activation='relu')
        self.batch3 = tf.keras.layers.BatchNormalization()
        self.pool3 = tf.keras.layers.AveragePooling2D()
        self.flatten = tf.keras.layers.Flatten()
        self.fc1 = tf.keras.layers.Dense(128, activation='relu')
        self.dropout = tf.keras.layers.Dropout(0.5)
        self.fc2 = tf.keras.layers.Dense(10, activation='softmax')

    def call(self, x):
        x = tf.reshape(x, [-1, 32, 32, 3])
        x = self.conv1(x)
        x = self.batch1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.batch2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.batch3(x)
        x = self.pool3(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Define hyperparameters and initialize
loss = tf.keras.losses.CategoricalCrossentropy()
learning_rate1 = 0.001
learning_rate2 = 0.01
optimizer1 = tf.keras.optimizers.Adam(learning_rate=learning_rate1)
optimizer2 = tf.keras.optimizers.Adam(learning_rate=learning_rate2)
optimizer3 = tf.keras.optimizers.Adagrad(learning_rate=learning_rate1)
optimizer4 = tf.keras.optimizers.Adagrad(learning_rate=learning_rate2)

# Preprocess the data
train_dataset = train_ds.apply(preprocess)
test_dataset = test_ds.apply(preprocess)

# Train the models
model1 = Architecture1()
model1 = train_model(model1, train_dataset, test_dataset, loss, optimizer1)
model2 = Architecture1()
model2 = train_model(model2, train_dataset, test_dataset, loss, optimizer2)
model3 = Architecture1()
model3 = train_model(model3, train_dataset, test_dataset, loss, optimizer3)
model4 = Architecture1()
model4 = train_model(model4, train_dataset, test_dataset, loss, optimizer4)
model5 = Architecture2()
model5 = train_model(model5, train_dataset, test_dataset, loss, optimizer1)
model6 = Architecture2()
model6 = train_model(model6, train_dataset, test_dataset, loss, optimizer2)
model7 = Architecture2()
model7 = train_model(model7, train_dataset, test_dataset, loss, optimizer3)
model8 = Architecture2()
model8 = train_model(model8, train_dataset, test_dataset, loss, optimizer4)
