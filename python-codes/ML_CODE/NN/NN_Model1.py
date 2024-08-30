import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

# Step 1: Load the MNIST data
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Step 2: Preprocess the data
train_images = train_images.reshape((train_images.shape[0], 28 * 28)).astype('float32') / 255
test_images = test_images.reshape((test_images.shape[0], 28 * 28)).astype('float32') / 255

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# Step 3: Build the model
model = models.Sequential()
model.add(layers.Input(shape=(784,)))  # Input layer of size 784 (28x28)
model.add(layers.Dense(128, activation='relu'))  # First hidden layer with 128 neurons and ReLU
model.add(layers.Dense(128, activation='relu'))  # Second hidden layer with 128 neurons and ReLU
model.add(layers.Dense(10, activation='softmax'))  # Output layer with 10 neurons and softmax

# Step 4: Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Step 5: Train the model
model.fit(train_images, train_labels, epochs=10, batch_size=128, validation_split=0.2)

# Step 6: Evaluate the model
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f"Test Accuracy: {test_acc:.4f}")
