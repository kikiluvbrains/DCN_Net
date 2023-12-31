import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CelebA
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import CelebA
import torchvision.transforms as transforms

# Define the Autoencoder model
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
        )

        # Calculate the shape of the encoder output
        self.encoder_output_shape = self._calculate_encoder_output_shape()

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),  # Output values between 0 and 1 for grayscale images
        )


    def _calculate_encoder_output_shape(self):
        with torch.no_grad():
            fake_input = torch.zeros((1, 1, 64, 64))
            encoder_output = self.encoder(fake_input)
        return encoder_output.shape

    def forward(self, x):
        x1 = self.encoder(x)
        reconstructed_image = self.decoder(x1)
        return reconstructed_image, x1

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import CelebA
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# ... (your Autoencoder class and other imports remain the same)

if __name__ == "__main__":
    # Hyperparameters
    batch_size = 64
    learning_rate = 0.001
    epochs = 2

    # Load the CelebA dataset
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])

    train_dataset = CelebA(root='./data', split='train', download=True, transform=transform)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    # Initialize the model, loss function, and optimizer
    model = Autoencoder()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(epochs):
        for batch_idx, data in enumerate(train_loader):
            images, _ = data
            images = images[0]  # Extract the image tensor from the tuple
            optimizer.zero_grad()
            reconstructions, _ = model(images)
            loss = criterion(reconstructions, images)
            loss.backward()
            optimizer.step()

            if batch_idx % 100 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item()}")

    # Visualize a sample reconstruction
    sample_image, _ = train_dataset[0]
    sample_image = sample_image.unsqueeze(0)  # Add batch dimension

    with torch.no_grad():
        reconstructed_image, features = model(sample_image)

    # Display the original and reconstructed images for the first layer
    plt.subplot(1, 2, 1)
    plt.title('Original Image')
    plt.imshow(sample_image.squeeze(0).numpy().squeeze(), cmap='gray')

    plt.subplot(1, 2, 2)
    plt.title('Reconstructed Image (First Layer)')
    plt.imshow(reconstructed_image.squeeze(0).numpy().squeeze(), cmap='gray')

    plt.show()
