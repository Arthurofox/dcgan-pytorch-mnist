import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")


# Hyperparameters
batch_size = 64
lr = 0.0002
num_epochs = 50    # Quick training run
nz = 100        # Latent vector size

# Data preparation: FashionMNIST (28x28, grayscale)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # Normalize images to [-1, 1]
])
dataset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

# -------------------------
# Model definitions
# -------------------------

class Generator(nn.Module):
    def __init__(self, nz):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            # Project and reshape: latent vector to (128, 7, 7)
            nn.Linear(nz, 128 * 7 * 7),
            nn.BatchNorm1d(128 * 7 * 7),
            nn.ReLU(True),
            nn.Unflatten(1, (128, 7, 7)),
            # Upsample to (64, 14, 14)
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # Upsample to (1, 28, 28)
            nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        return self.net(x)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            # Downsample: (1, 28, 28) -> (64, 14, 14)
            nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # Downsample: (64, 14, 14) -> (128, 7, 7)
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            # Flatten and output a single probability
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

# Initialize models
netG = Generator(nz).to(device)
netD = Discriminator().to(device)

# Loss function and optimizers
criterion = nn.BCELoss()
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(0.5, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(0.5, 0.999))

# -------------------------
# Training loop
# -------------------------
print("Starting training...")
for epoch in range(num_epochs):
    for i, (images, _) in enumerate(dataloader):
        b_size = images.size(0)
        real_images = images.to(device)
        real_labels = torch.full((b_size,), 1.0, device=device)
        fake_labels = torch.full((b_size,), 0.0, device=device)

        # Train Discriminator: maximize log(D(x)) + log(1-D(G(z)))
        netD.zero_grad()
        # Train on real images
        output_real = netD(real_images)
        loss_real = criterion(output_real.view(-1), real_labels)
        loss_real.backward()

        # Generate fake images and train on them
        noise = torch.randn(b_size, nz, device=device)
        fake_images = netG(noise)
        output_fake = netD(fake_images.detach())
        loss_fake = criterion(output_fake.view(-1), fake_labels)
        loss_fake.backward()

        optimizerD.step()

        # Train Generator: minimize log(1-D(G(z))) or maximize log(D(G(z)))
        netG.zero_grad()
        # We want the fake images to be classified as real
        output_fake_forG = netD(fake_images)
        loss_G = criterion(output_fake_forG.view(-1), real_labels)
        loss_G.backward()
        optimizerG.step()

        if i % 100 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}] Batch {i}/{len(dataloader)}  "
                  f"Loss D: {(loss_real+loss_fake).item():.4f} Loss G: {loss_G.item():.4f}")

# -------------------------
# Visualize a few generated images
# -------------------------
print("Training complete. Generating sample images...")
with torch.no_grad():
    sample_noise = torch.randn(16, nz, device=device)
    fake_samples = netG(sample_noise).cpu()
    grid = torchvision.utils.make_grid(fake_samples, nrow=4, normalize=True)
    plt.figure(figsize=(6,6))
    plt.axis("off")
    plt.title("Generated Images")
    plt.imshow(grid.permute(1, 2, 0).squeeze(), cmap="gray")
    plt.show()
