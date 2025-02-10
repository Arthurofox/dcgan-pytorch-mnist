import torch
import unittest
from src.config import Config
from src.models.generator import Generator
from src.models.discriminator import Discriminator

class TestModels(unittest.TestCase):
    def setUp(self):
        self.config = Config()
        self.device = self.config.device
        self.gen = Generator(self.config.nz, self.config.ngf, self.config.channels).to(self.device)
        self.disc = Discriminator(self.config.channels, self.config.ndf).to(self.device)
    
    def test_generator_output(self):
        noise = torch.randn(16, self.config.nz, 1, 1, device=self.device)
        output = self.gen(noise)
        # Expect output shape: (batch_size, channels, image_size, image_size)
        self.assertEqual(output.shape, (16, self.config.channels, self.config.image_size, self.config.image_size))
    
    def test_discriminator_output(self):
        # Create a dummy batch of images
        images = torch.randn(16, self.config.channels, self.config.image_size, self.config.image_size, device=self.device)
        output = self.disc(images)
        # Expect output shape: (batch_size,)
        self.assertEqual(output.shape, (16,))

if __name__ == '__main__':
    unittest.main()
