import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define input image dimensions
img_width, img_height = 224, 224  # Adjust dimensions as necessary based on your images

# Define number of classes (labels)
num_classes = 5  # Adjust based on the number of labels

# Define paths to train and test data
train_data_dir = 'C:/Users/kalli/Documents/GitHub/Infant-Crying-Classification/SplitData/train'
test_data_dir = 'C:/Users/kalli/Documents/GitHub/Infant-Crying-Classification/SplitData/test'

# Define image data generators without data augmentation
train_datagen = ImageDataGenerator(rescale=1./255)  # Only rescale pixel values
test_datagen = ImageDataGenerator(rescale=1./255)

# Generate batches of image data
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=32,
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(img_width, img_height),
    batch_size=32,
    class_mode='categorical'
)

# Define CNN model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_width, img_height, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dense(num_classes, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    epochs=10,  # Adjust as needed
    validation_data=test_generator,
    validation_steps=test_generator.samples // test_generator.batch_size
)
