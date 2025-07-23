import pickle
import numpy as np
import os
from matplotlib import pyplot as plt  # For visualization

def unpickle(file):
    """Unpickle a CIFAR-10 batch file"""
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def load_cifar10(data_dir):
    """Load all CIFAR-10 batches"""
    train_data = []
    train_labels = []
    
    # Load training batches (1-5)
    for i in range(1, 6):
        batch_file = os.path.join(data_dir, f'data_batch_{i}')
        batch_dict = unpickle(batch_file)
        train_data.append(batch_dict[b'data'])
        train_labels.extend(batch_dict[b'labels'])
    
    # Load test batch
    test_batch = unpickle(os.path.join(data_dir, 'test_batch'))
    test_data = test_batch[b'data']
    test_labels = test_batch[b'labels']
    
    # Convert to numpy arrays and reshape images
    train_data = np.vstack(train_data).reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    test_data = test_data.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    
    return (train_data, np.array(train_labels)), (test_data, np.array(test_labels))

def show_image(img, label):
    """Display a single CIFAR-10 image with its label"""
    plt.imshow(img)
    plt.title(f"Label: {label}")
    plt.axis('off')
    plt.show()

if __name__ == '__main__':
    # Replace this path with where your CIFAR-10 batches are stored
    cifar10_dir = r'C:\Users\ppssa\OneDrive\Työpöytä\OPINNOT\cifar-10-batches-py'
    
    # Load the data
    (train_images, train_labels), (test_images, test_labels) = load_cifar10(cifar10_dir)
    
    print(f"Training data shape: {train_images.shape}")
    print(f"Training labels shape: {train_labels.shape}")
    print(f"Test data shape: {test_images.shape}")
    print(f"Test labels shape: {test_labels.shape}")
    
    # Display the first training image as an example
    # show_image(train_images[0], train_labels[0])
    
    np.save('cifar10_train_images.npy', train_images)
    np.save('cifar10_train_labels.npy', train_labels)
    np.save('cifar10_test_images.npy', test_images)
    np.save('cifar10_test_labels.npy', test_labels)
    
    train_images = np.load('cifar10_train_images.npy')
