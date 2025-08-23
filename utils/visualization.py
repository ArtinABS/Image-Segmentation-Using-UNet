import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

from data.data_loader import data_loader_MNIST, data_loader_SVHN

def data_exploration():
    train_data, test_data = data_loader_MNIST()

    train_data.columns = test_data.columns
    
    data = pd.concat([train_data, test_data], ignore_index=True)
    number = data.iloc[:, 0]
    unique_numbers = number.unique()
    images = data.iloc[:, 1:]
    index = 1

    
    for i in unique_numbers:
        label = int(i)
        images_for_label = images[number == label]
        random_images = images_for_label.sample(n=5)

        for random_image in random_images.iterrows():
            plt.subplot(len(unique_numbers), 5, index)
            index += 1
            plt.axis('off')
            image_data = random_image[1].values[:784]
            plt.imshow(image_data.reshape(28, 28), cmap='gray')
            plt.title(f"Label: {label}")
        
    plt.figure(figsize=(10, 10))
    plt.subplots_adjust(hspace=0.8)
    plt.show()

    plt.figure(figsize=(10, 6))
    number_counts = number.value_counts().sort_index()
    sns.barplot(x=number_counts.index, y=number_counts.values)
    plt.title('Distribution of Numbers in Dataset')
    plt.xlabel('Number Label')
    plt.ylabel('Count')
    plt.show()

def plot(losses, cv_losses, acc_train, acc_cv):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(losses, label="Training Loss")
    axes[0].plot(cv_losses, label="CV Loss")
    axes[0].set_xlabel("Epochs")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Base Model Training and CV Loss")
    axes[0].legend()
    
    axes[1].plot(acc_train, label="Training Accuracy")
    
    axes[1].plot(acc_cv, label="CV Accuracy")
    axes[1].set_xlabel("Epochs")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Base Model Training and CV Accuracy")
    axes[1].legend()
    
    plt.tight_layout()
    plt.show()

def plot_loss(losses):
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.plot(losses)
    plt.show()

def plot_predictions(y_pred, y_test, title, y_label, x_label):
    plt.figure(figsize=(10, 5))

    plt.scatter(y_test, y_pred, alpha=0.6, color='blue', label=title)

    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], linestyle="--", color="red", label="Perfect Fit (y = x)")
    
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend()
    plt.show()

def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=range(10), yticklabels=range(10))
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

