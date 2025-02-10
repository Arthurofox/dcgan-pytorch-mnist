import torch

class Config:
    # Data parameters
    image_size = 64      # We'll resize FashionMNIST images from 28x28 to 64x64
    channels = 1         # FashionMNIST is grayscale

    # Model hyperparameters
    nz = 100             # Dimension of latent noise vector
    ngf = 64             # Generator feature map size
    ndf = 64             # Discriminator feature map size

    # Training hyperparameters
    num_epochs = 5
    batch_size = 128
    lr = 0.0002
    beta1 = 0.5          # Beta1 hyperparam for Adam optimizer

    # Device configuration
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

