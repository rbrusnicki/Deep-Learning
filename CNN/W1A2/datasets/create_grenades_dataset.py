import os
import h5py
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

# Path to the main folder containing 61 subfolders
data_path = "FFHISMD"
output_dir = '.'

# Create output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Image size (fixed at 53x21)
image_size = (53, 21)

# Initialize lists to hold image data and labels
images = []
labels = []

# Iterate over each subfolder
subfolders = sorted(os.listdir(data_path))
for folder in subfolders:
    folder_path = os.path.join(data_path, folder)
    if not os.path.isdir(folder_path):
        continue  # Skip non-directory files

    # Generate label based on specific character positions in the folder name
    label_vector = np.array([1 if folder[i] != '_' else 0 for i in range(7)])

    # # Print folder name and label vector for visual verification
    # print(f"Folder: [{folder[0]} {folder[1]} {folder[2]} {folder[3]} {folder[4]} {folder[5]} {folder[6]}]")
    # print(f"Label:  {label_vector}")
    # print("\n")

    # Read each image in the subfolder
    for image_file in os.listdir(folder_path):
        image_path = os.path.join(folder_path, image_file)
        image = Image.open(image_path)#.resize(image_size)
        image_array = np.array(image)

        # Ensure grayscale or RGB format
        if image_array.ndim == 2:  # Grayscale
            image_array = np.stack((image_array,) * 3, axis=-1)  # Convert to RGB
        
        images.append(image_array)
        labels.append(label_vector)

# Convert lists to numpy arrays
images = np.array(images)
labels = np.array(labels)

# Split data into training and testing sets
train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size=0.1, random_state=42)

# Save training set
train_file = os.path.join(output_dir, 'train_FFHISMD.h5')
with h5py.File(train_file, 'w') as h5f:
    h5f.create_dataset("train_set_x", data=train_images)
    h5f.create_dataset("train_set_y", data=train_labels)
    print(f'Training dataset created at {train_file}')

# Save testing set
test_file = os.path.join(output_dir, 'test_FFHISMD.h5')
with h5py.File(test_file, 'w') as h5f:
    h5f.create_dataset("test_set_x", data=test_images)
    h5f.create_dataset("test_set_y", data=test_labels)
    print(f'Testing dataset created at {test_file}')
