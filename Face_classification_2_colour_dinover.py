import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import CelebA
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision.models as models
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
import torch
import torch.nn as nn
from torchvision.transforms import ToTensor
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CelebA
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torchvision.models as models


# Define the Autoencoder model
# Define the encoder-decoder architecture
class Autoencoder(nn.Module):
    def __init__(self, encoder):
        super(Autoencoder, self).__init__()
        self.encoder = encoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8, 192, kernel_size=13, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(192, 64, kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 3, kernel_size=3, stride=2, output_padding=1),  # Adjusted output_padding
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# Load pre-trained AlexNet
alexnet = models.alexnet(pretrained=True)

# Extract convolutional layers as the encoder
encoder = nn.Sequential(*list(alexnet.features.children())[:6])

# Create the autoencoder
autoencoder = Autoencoder(encoder)

# Set the encoder to non-trainable
for param in autoencoder.encoder.parameters():
    param.requires_grad = False

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

if __name__ == "__main__":
    # Hyperparameters
    batch_size = 64
    learning_rate = 0.001
    epochs = 2

    # Load a smaller subset of the CelebA dataset
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])

    train_dataset = CelebA(root='./data', split='train', download=True, transform=transform)
    train_subset = torch.utils.data.Subset(train_dataset, range(1000))  # Use a smaller subset
    train_loader = DataLoader(dataset=train_subset, batch_size=batch_size, shuffle=True, num_workers=4)

    # Initialize the model, loss function, and optimizer
    model = Autoencoder()
    criterion = nn.MSELoss() 
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(epochs):
        for batch_idx, data in enumerate(train_loader):
            images, _ = data
            optimizer.zero_grad()
            reconstructions = model(images)
            loss = criterion(reconstructions, images)
            loss.backward()
            optimizer.step()

            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item()}")

    # Load the image using PIL
    image_path = "D:/Work/face.jpg"
    sample_image = Image.open(image_path).convert("RGB")  # Ensure the image is in RGB mode

    # Define a transform to preprocess the image and add a batch dimension
    transform = transforms.ToTensor()
    sample_image = transform(sample_image).unsqueeze(0)

    # Pass the image through the autoencoder
    with torch.no_grad():
        reconstructed_image = model(sample_image)

    # Display the original and reconstructed images
    plt.subplot(1, 2, 1)
    plt.title('Original Image')
    plt.imshow(sample_image.squeeze(0).permute(1, 2, 0).numpy())

    plt.subplot(1, 2, 2)
    plt.title('Reconstructed Image')
    plt.imshow(reconstructed_image.squeeze(0).permute(1, 2, 0).numpy())

    plt.show()

