import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils
from tqdm import tqdm

from config import Config
from utils import get_dataloader
from models.generator import Generator
from models.discriminator import Discriminator

if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

def weights_init(m):
    """Custom weights initialization."""
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def main():
    # Load configuration
    config = Config()

    # Create output directory for generated images and checkpoints
    os.makedirs("outputs", exist_ok=True)

    # Prepare DataLoader for FashionMNIST
    dataloader = get_dataloader(config.batch_size, config.image_size)

    # Create the generator and discriminator models
    netG = Generator(config.nz, config.ngf, config.channels).to(config.device)
    netD = Discriminator(config.channels, config.ndf).to(config.device)

    # Initialize weights
    netG.apply(weights_init)
    netD.apply(weights_init)

    # Loss function and optimizers
    criterion = nn.BCELoss()
    optimizerD = optim.Adam(netD.parameters(), lr=config.lr, betas=(config.beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=config.lr, betas=(config.beta1, 0.999))

    # Fixed noise for visualizing generator progress
    fixed_noise = torch.randn(64, config.nz, 1, 1, device=config.device)

    real_label = 1.
    fake_label = 0.

    print("Starting Training Loop...")
    for epoch in range(config.num_epochs):
        for i, (data, _) in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}/{config.num_epochs}")):
            b_size = data.size(0)
            real_data = data.to(config.device)
            label = torch.full((b_size,), real_label, dtype=torch.float, device=config.device)

            # --- Train Discriminator ---
            netD.zero_grad()
            # Train on real images
            output = netD(real_data)
            errD_real = criterion(output, label)
            errD_real.backward()
            D_x = output.mean().item()

            # Generate fake images
            noise = torch.randn(b_size, config.nz, 1, 1, device=config.device)
            fake = netG(noise)
            label.fill_(fake_label)

            # Train on fake images
            output = netD(fake.detach())
            errD_fake = criterion(output, label)
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            errD = errD_real + errD_fake
            optimizerD.step()

            # --- Train Generator ---
            netG.zero_grad()
            label.fill_(real_label)  # Generator aims to fool the discriminator
            output = netD(fake)
            errG = criterion(output, label)
            errG.backward()
            D_G_z2 = output.mean().item()
            optimizerG.step()

            if i % 50 == 0:
                print(f"[{epoch+1}/{config.num_epochs}][{i}/{len(dataloader)}] "
                      f"Loss_D: {errD.item():.4f} Loss_G: {errG.item():.4f} "
                      f"D(x): {D_x:.4f} D(G(z)): {D_G_z1:.4f} / {D_G_z2:.4f}")

        # Save generated images for inspection
        with torch.no_grad():
            fake = netG(fixed_noise).detach().cpu()
        vutils.save_image(fake, f'outputs/fake_samples_epoch_{epoch+1}.png', normalize=True)

        # Save model checkpoints (optional)
        torch.save(netG.state_dict(), f'outputs/netG_epoch_{epoch+1}.pth')
        torch.save(netD.state_dict(), f'outputs/netD_epoch_{epoch+1}.pth')

if __name__ == "__main__":
    main()
