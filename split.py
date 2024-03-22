import os
import shutil
import random

# Define the ratio of images to be used for training
train_ratio = 0.8  # 80% for training, 20% for testing

# Function to split images into train and test sets
def split_train_test_images(input_folder_path, train_folder_path, test_folder_path):
    # List all images in the folder
    images = [f for f in os.listdir(input_folder_path) if f.endswith('.png')]
    
    # Calculate the number of images for training
    num_train_images = int(len(images) * train_ratio)
    
    # Randomly shuffle the images
    random.shuffle(images)
    
    # Split images into train and test sets
    train_images = images[:num_train_images]
    test_images = images[num_train_images:]
    
    # Copy train images to train folder
    for image in train_images:
        src_path = os.path.join(input_folder_path, image)
        dst_path = os.path.join(train_folder_path, image)
        shutil.copy(src_path, dst_path)
    
    # Copy test images to test folder
    for image in test_images:
        src_path = os.path.join(input_folder_path, image)
        dst_path = os.path.join(test_folder_path, image)
        shutil.copy(src_path, dst_path)

# Define input and output parent folder paths
input_parent_folder_path = 'C:/Users/kalli/Documents/GitHub/Infant-Crying-Classification/Photos'
output_parent_folder_path = 'C:/Users/kalli/Documents/GitHub/Infant-Crying-Classification/SplitData'

# List of labels
labels = ['belly_pain', 'discomfort', 'hungry', 'tired', 'burping']

# Iterate through each label
for label in labels:
    # Define input and output folder paths for the current label
    input_folder_path = os.path.join(input_parent_folder_path, label)
    train_folder_path = os.path.join(output_parent_folder_path, 'train', label)
    test_folder_path = os.path.join(output_parent_folder_path, 'test', label)
    
    # Create train and test folders if they don't exist
    os.makedirs(train_folder_path, exist_ok=True)
    os.makedirs(test_folder_path, exist_ok=True)
    
    # Split images into train and test sets
    split_train_test_images(input_folder_path, train_folder_path, test_folder_path)
