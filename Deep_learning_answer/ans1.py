import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

# Load and preprocess the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0  # Normalize pixel values to be between 0 and 1

# Convert labels to one-hot encoding
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Define a simple CNN model
model = models.Sequential()

# Convolutional layer with 32 filters, 3x3 kernel size, and ReLU activation
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))

# MaxPooling layer to down-sample the spatial dimensions
model.add(layers.MaxPooling2D((2, 2)))

# Flatten layer to flatten the input for the fully connected layers
model.add(layers.Flatten())

# Fully connected layer with 128 units and ReLU activation
model.add(layers.Dense(128, activation='relu'))

# Output layer with 10 units (for 10 classes) and softmax activation
model.add(layers.Dense(10, activation='softmax'))

# Display the model summary
model.summary()

# Compile the model
model.compile(optimizer='adam', 
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Reshape the input data to match the CNN input requirements
x_train = x_train.reshape((x_train.shape[0], 28, 28, 1))
x_test = x_test.reshape((x_test.shape[0], 28, 28, 1))

# Train the model
history = model.fit(x_train, y_train, epochs=10, validation_split=0.2, batch_size=64)

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f'Test Accuracy: {test_acc * 100:.2f}%')
