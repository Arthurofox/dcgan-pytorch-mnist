import torch
import torchvision.utils as vutils
import matplotlib.pyplot as plt

from config import Config
from models.generator import Generator

# Use MPS on Apple Silicon if available
if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

def main():
    # Load configuration
    config = Config()

    # Initialize generator model
    netG = Generator(config.nz, config.ngf, config.channels).to(device)
    
    # Load the final generator checkpoint
    checkpoint_path = 'outputs/netG_final.pth'
    netG.load_state_dict(torch.load(checkpoint_path, map_location=device))
    netG.eval()  # Set the model to evaluation mode

    # Generate sample images using random noise
    sample_noise = torch.randn(64, config.nz, 1, 1, device=device)
    with torch.no_grad():
        fake_images = netG(sample_noise).detach().cpu()

    # Save the generated images
    output_path = 'outputs/eval_fake_samples.png'
    vutils.save_image(fake_images, output_path, normalize=True)
    print(f"Generated evaluation samples saved to {output_path}")

    # Display the generated images using matplotlib
    grid = vutils.make_grid(fake_images, nrow=8, normalize=True)
    plt.figure(figsize=(8,8))
    plt.axis("off")
    plt.title("Evaluation - Generated Images")
    plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
    plt.show()

if __name__ == "__main__":
    main()
