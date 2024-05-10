import torch

import torch.nn as nn
import torch.nn.functional as F


"""
NOTE: you can add as many functions as you need in this file, and for all the classes you can define extra methods if you need
"""

class VAE(nn.Module):
  # feel free to define your arguments
  def __init__(self, hidden_dim, latent_dim, class_emb_dim, num_classes=10):
    super().__init__()

    self.latent_dim = latent_dim
    self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),  # 28x28 -> 14x14
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # 14x14 -> 7x7
            nn.ReLU(),
            nn.Flatten()
    )

    # defining the network to estimate the mean
    self.mu_net = nn.Linear(64 * 7 * 7, latent_dim) # implement your mean estimation module here

    # defining the network to estimate the log-variance
    self.logvar_net = nn.Linear(64 * 7 * 7, latent_dim) # implement your log-variance estimation here

    # defining the class embedding module
    self.class_embedding = nn.Embedding(num_classes, class_emb_dim) # implement your class-embedding module here

    # defining the decoder here
    self.decoder = nn.Sequential(
        nn.Linear(latent_dim + class_emb_dim, 64 * 7 * 7),
        nn.ReLU(),
        nn.Unflatten(1, (64, 7, 7)),
        nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # 7x7 -> 14x14
        nn.ReLU(),
        nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),  # 14x14 -> 28x28
        nn.Sigmoid()
    ) # implement your decoder here

  def forward(self, x: torch.Tensor, y: torch.Tensor):
    """
    Args:
        x (torch.Tensor): image [B, 1, 28, 28]
        y (torch.Tensor): labels [B]

    Returns:
        reconstructed: image [B, 1, 28, 28]
        mu: [B, latent_dim]
        logvar: [B, latent_dim]
    """

    # implement your forward function here
    x = x.view(x.size(0), -1, 28, 28)

    # Encode
    encoded = self.encoder(x)

    # Estimate mean and logvar
    mu = self.mu_net(encoded)
    logvar = self.logvar_net(encoded)

    # Reparameterization trick to sample from the latent space
    z = self.reparameterize(mu, logvar)

    # Class embedding
    y_emb = self.class_embedding(y)

    # Concatenate latent variable with class embedding
    z = torch.cat([z, y_emb], dim=1)

    # Decode
    reconstructed = self.decoder(z)
    reconstructed = reconstructed.view(-1, 1, 28, 28)

    return reconstructed, mu, logvar

  def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor):
    """
    applies the reparameterization trick
    """
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device('cuda')

    new_sample = mu + torch.randn(logvar.shape).to(device) * torch.exp(logvar/2.0) # using the mu and logvar generate a sample

    return new_sample

  def kl_loss(self, mu, logvar):
    """
    calculates the KL divergence between a normal distribution with mean "mu" and
    log-variance "logvar" and the standard normal distribution (mean=0, var=1)
    """

    kl_div = -0.5 * torch.sum(1 + logvar - torch.exp(logvar)- mu**2) # calculate the kl-div using mu and logvar

    return kl_div.mean()

  def get_loss(self, x: torch.Tensor, y: torch.Tensor):
    """
    given the image x, and the label y calculates the prior loss and reconstruction loss
    """
    reconstructed, mu, logvar = self.forward(x, y)

    # reconstruction loss
    # compute the reconstruction loss here using the "reconstructed" variable above
    recons_loss = nn.BCELoss(reduction='sum')(reconstructed, x) / x.size(0)

    # prior matching loss
    prior_loss = self.kl_loss(mu, logvar)

    return recons_loss, prior_loss

  @torch.no_grad()
  def generate_sample(self, num_images: int, y, device):
    """
    generates num_images samples by passing noise to the model's decoder
    if y is not None (e.g., y = torch.tensor([1, 2, 3]).to(device)) the model
    generates samples according to the specified labels

    Returns:
        samples: [num_images, 1, 28, 28]
    """
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device('cuda')
    
    if y is None:
        # Generate random labels if not specified
        y = torch.randint(0, self.class_embedding.num_embeddings, (num_images,), device=device)
    
    z = torch.randn(num_images, self.latent_dim).to(device)
    y_emb = self.class_embedding(y).to(device)
    z = torch.cat([z, y_emb], dim=1)

    # sample from noise, find the class embedding and use both in the decoder to generate new samples
    samples = self.decoder(z)
    samples = samples.view(-1, 1, 28, 28)

    return samples


def load_vae_and_generate():
    device = torch.device('cuda')
    vae = VAE(300, 64, 64, 10) # define your VAE model according to your implementation above

    # loading the weights of VAE
    vae.load_state_dict(torch.load('vae.pt'))
    vae = vae.to(device)

    desired_labels = []
    for i in range(10):
        for _ in range(5):
            desired_labels.append(i)

    desired_labels = torch.tensor(desired_labels).to(device)
    generated_samples = vae.generate_sample(50, desired_labels, device)

    return generated_samples