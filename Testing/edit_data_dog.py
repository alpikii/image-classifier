import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt

# --- CONFIG ---
edit_percent = 10  # <-- Change this to 20, 30, etc.

# Load images and labels
train_images = np.load(f'C:/Users/ppssa/OneDrive/Työpöytä/OPINNOT/Kandi/cifar-10-edited-10/cifar10_train_images.npy')
train_labels = np.load(f'C:/Users/ppssa/OneDrive/Työpöytä/OPINNOT/Kandi/cifar-10-edited-10/cifar10_train_labels.npy')

# Make modifiable copies
modified_images = train_images.copy()
modified_labels = train_labels.copy()

# --- EDIT 10% OF DATA ---
# n_samples = len(modified_images)
dog_indices = np.where(train_labels == 5)[0]

# n_edit = int((edit_percent / 100) * n_samples)
n_edit = int((edit_percent / 100) * len(dog_indices))

# indices_to_edit = np.random.choice(n_samples, n_edit, replace=False)
indices_to_edit = np.random.choice(dog_indices, n_edit, replace=False)

# Change 10% of labels to "cat" (label 3)
modified_labels[indices_to_edit] = 3

# Optionally, invert the images (change pixels)
# modified_images[indices_to_edit] = 255 - modified_images[indices_to_edit]

# --- SAVE MODIFIED DATA ---
output_dir =  f'modified_cifar10_dog_{edit_percent}'
os.makedirs(output_dir, exist_ok=True)

np.save(os.path.join(output_dir, f'cifar10_train_images_dog_{edit_percent}.npy'), modified_images)
np.save(os.path.join(output_dir, f'cifar10_train_labels_dog_{edit_percent}.npy'), modified_labels)

# Save some of the changed images for reference
image_subdir = os.path.join(output_dir, 'dog_examples')
os.makedirs(image_subdir, exist_ok=True)

for i in indices_to_edit[:100]:
    img = Image.fromarray(modified_images[i].astype('uint8'))
    img.save(os.path.join(image_subdir, f'dog_img_{i}.png'))

# --- VERIFY CHANGES ---
print(f"Edited {edit_percent}% of data ({n_edit} samples changed to cat)")
print(f"Total 'cat' samples now: {np.sum(modified_labels == 3)}")

# Show before and after of a sample
sample_idx = indices_to_edit[0]

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(train_images[sample_idx])  # original image
plt.title(f"Original (label: {train_labels[sample_idx]})")

plt.subplot(1, 2, 2)
plt.imshow(modified_images[sample_idx])  # modified image
plt.title("Modified (label: 3 - cat)")

plt.show()
