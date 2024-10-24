import os
import h5py
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

# Path to the folder containing subfolders of images
data_path = "inferno"
output_dir = '.'

# Create output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Image size for resizing (if necessary)
image_size = (48, 48)  # Adjust this as per your requirements

# Get the list of class folders
classes = sorted(os.listdir(data_path))
num_classes = len(classes)

# Initialize lists to hold image data and labels
images = []
labels = []

# Iterate over each class folder
for label, class_name in enumerate(classes):
    class_folder = os.path.join(data_path, class_name)
    image_files = os.listdir(class_folder)
    
    for image_file in image_files:
        image_path = os.path.join(class_folder, image_file)
        image = Image.open(image_path).resize(image_size)
        image_array = np.array(image)
        
        # Ensure the image is in RGB format
        if image_array.ndim == 2:  # Grayscale images
            image_array = np.stack((image_array,) * 3, axis=-1)  # Convert grayscale to RGB
        
        images.append(image_array)
        labels.append(label)

# Convert lists to numpy arrays
images = np.array(images)
labels = np.array(labels)

# Split data into training and testing sets (e.g., 80% train, 20% test)
train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size=0.1, random_state=42)

# Save training set
train_file = os.path.join(output_dir, 'train_inferno.h5')
with h5py.File(train_file, 'w') as h5f:
    h5f.create_dataset("train_set_x", data=train_images)
    h5f.create_dataset("train_set_y", data=train_labels)
    h5f.create_dataset("list_classes", data=np.array(classes, dtype='S'))
print(f'Training dataset created at {train_file}')

# Save testing set
test_file = os.path.join(output_dir, 'test_inferno.h5')
with h5py.File(test_file, 'w') as h5f:
    h5f.create_dataset("test_set_x", data=test_images)
    h5f.create_dataset("test_set_y", data=test_labels)
    h5f.create_dataset("list_classes", data=np.array(classes, dtype='S'))
print(f'Testing dataset created at {test_file}')
