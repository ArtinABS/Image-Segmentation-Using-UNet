import matplotlib.pyplot as plt
from data.data_loader import *
import numpy as np

def show_random_samples(dataloader, num_samples=5, dataset_mean=[0.485, 0.456, 0.406], dataset_std=[0.5, 0.5, 0.5]):
    """
    Displays random image-mask pairs from a given dataloader.

    Args:
        dataloader (torch.utils.data.DataLoader): The DataLoader to draw samples from.
        num_samples (int): The number of random samples to display.
        dataset_mean (list): Mean values used for normalization during dataset creation.
        dataset_std (list): Standard deviation values used for normalization.
    """
    if num_samples > dataloader.batch_size:
        print(f"Warning: num_samples ({num_samples}) is greater than dataloader batch_size ({dataloader.batch_size}). Will display {dataloader.batch_size} samples.")
        num_samples = dataloader.batch_size

    # Get one batch of data
    images, masks_onehot = next(iter(dataloader))

    plt.figure(figsize=(6, 3 * num_samples))
    for i in range(num_samples):
        # Denormalize image
        image = images[i].cpu().numpy().transpose((1, 2, 0)) # C, H, W -> H, W, C
        mean = np.array(dataset_mean)
        std = np.array(dataset_std)
        image = std * image + mean
        image = np.clip(image, 0, 1) # Clip to [0, 1] range

        # Convert one-hot mask to single channel mask
        # Masks_onehot shape: (B, num_classes, H, W)
        mask = torch.argmax(masks_onehot[i], dim=0).cpu().numpy() # H, W

        # Plot Image
        plt.subplot(num_samples, 2, 2*i + 1)
        plt.imshow(image)
        plt.title(f"Sample {i+1} Image")
        plt.axis('off')

        # Plot Mask
        plt.subplot(num_samples, 2, 2*i + 2)
        plt.imshow(mask, cmap='viridis') # Use a colormap for masks
        plt.title(f"Sample {i+1} Mask")
        plt.axis('off')
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    train_loader, val_loader, test_loader = get_dataloader(
        root="C:/Users/AmirHosein/Desktop/resized",
        batch_size=8, # Use a smaller batch size for demonstration
        input_size=256,
    )

    print("\nDisplaying 5 random samples from the Training DataLoader:")
    show_random_samples(train_loader, num_samples=5)

    # print("\nDisplaying 5 random samples from the Validation DataLoader:")
    # show_random_samples(val_loader, num_samples=5)