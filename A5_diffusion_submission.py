import torch

import torch.nn as nn
import torch.nn.functional as F

from typing import Tuple

"""
NOTE: you can add as many functions as you need in this file, and for all the classes you can define extra methods if you need
"""


class VarianceScheduler:
    """
    This class is used to keep track of statistical variables used in the diffusion model
    and also adding noise to the data
    """
    def __init__(self, beta_start: float=0.0001, beta_end: float=0.02, num_steps :int=1000):
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device('cuda')

        self.betas = torch.linspace(beta_start, beta_end, num_steps).to(self.device) # defining the beta variables
        self.alphas = 1 - self.betas # defining the alpha variables
        self.alpha_bars = torch.tensor([torch.prod(self.alphas[:i + 1]) for i in range(len(self.alphas))]).to(self.device)# defining the alpha bar variables

        # NOTE:Feel free to add to this or modify it as you wish


    def add_noise(self, x: torch.Tensor, timestep: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        This method receives the input data and the timestep, generates a noise according to the
        timestep, perturbs the data with the noise, and returns the noisy version of the data and
        the noise itself

        Args:
            x (torch.Tensor): input image [B, 1, 28, 28]
            timestep (torch.Tensor): timesteps [B]

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: noisy_x [B, 1, 28, 28], noise [B, 1, 28, 28]
        """
        n, c, h, w = x.shape
        a_bar = self.alpha_bars[timestep].to(x.device)
        noise = torch.randn(n, c, h, w).to(x.device)

        noisy_x = (a_bar.sqrt().reshape(n, 1, 1, 1) * x + (1 - a_bar).sqrt().reshape(n, 1, 1, 1) * noise)
        # noisy_x = x + noise

        return noisy_x, noise


class NoiseEstimatingNet(nn.Module):
    """
    The implementation of the noise estimating network for the diffusion model
    """
    # feel free to add as many arguments as you need or change the arguments
    def __init__(self, time_emb_dim: int, class_emb_dim: int, num_classes: int=10):
        super().__init__()

        # add your codes here
        self.time_embed = nn.Embedding(1000, time_emb_dim)
        self.class_embedding = nn.Embedding(num_classes, class_emb_dim)

        # First half
        self.te1 = self._make_te(time_emb_dim + class_emb_dim, 1)
        self.b1 = nn.Sequential(
            MyBlock((1, 28, 28), 1, 10),
            MyBlock((10, 28, 28), 10, 10),
            MyBlock((10, 28, 28), 10, 10)
        )
        self.down1 = nn.Conv2d(10, 10, 4, 2, 1)

        self.te2 = self._make_te(time_emb_dim + class_emb_dim, 10)
        self.b2 = nn.Sequential(
            MyBlock((10, 14, 14), 10, 20),
            MyBlock((20, 14, 14), 20, 20),
            MyBlock((20, 14, 14), 20, 20)
        )
        self.down2 = nn.Conv2d(20, 20, 4, 2, 1)

        self.te3 = self._make_te(time_emb_dim + class_emb_dim, 20)
        self.b3 = nn.Sequential(
            MyBlock((20, 7, 7), 20, 40),
            MyBlock((40, 7, 7), 40, 40),
            MyBlock((40, 7, 7), 40, 40)
        )
        self.down3 = nn.Sequential(
            nn.Conv2d(40, 40, 2, 1),
            nn.SiLU(),
            nn.Conv2d(40, 40, 4, 2, 1)
        )

        # Bottleneck
        self.te_mid = self._make_te(time_emb_dim + class_emb_dim, 40)
        self.b_mid = nn.Sequential(
            MyBlock((40, 3, 3), 40, 20),
            MyBlock((20, 3, 3), 20, 20),
            MyBlock((20, 3, 3), 20, 40)
        )

        # Second half
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(40, 40, 4, 2, 1),
            nn.SiLU(),
            nn.ConvTranspose2d(40, 40, 2, 1)
        )

        self.te4 = self._make_te(time_emb_dim + class_emb_dim, 80)
        self.b4 = nn.Sequential(
            MyBlock((80, 7, 7), 80, 40),
            MyBlock((40, 7, 7), 40, 20),
            MyBlock((20, 7, 7), 20, 20)
        )

        self.up2 = nn.ConvTranspose2d(20, 20, 4, 2, 1)
        self.te5 = self._make_te(time_emb_dim + class_emb_dim, 40)
        self.b5 = nn.Sequential(
            MyBlock((40, 14, 14), 40, 20),
            MyBlock((20, 14, 14), 20, 10),
            MyBlock((10, 14, 14), 10, 10)
        )

        self.up3 = nn.ConvTranspose2d(10, 10, 4, 2, 1)
        self.te_out = self._make_te(time_emb_dim + class_emb_dim, 20)
        self.b_out = nn.Sequential(
            MyBlock((20, 28, 28), 20, 10),
            MyBlock((10, 28, 28), 10, 10),
            MyBlock((10, 28, 28), 10, 10, normalize=False)
        )

        self.conv_out = nn.Conv2d(10, 1, 3, 1, 1)


    def forward(self, x: torch.Tensor, timestep: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Estimate the noise given the input image, timestep, and the label

        Args:
            x (torch.Tensor): the input (noisy) image [B, 1, 28, 28]
            timestep (torch.Tensor): timestep [B]
            y (torch.Tensor): the corresponding labels for the images [B]

        Returns:
            torch.Tensor: out (the estimated noise) [B, 1, 28, 28]
        """

        t = self.time_embed(timestep)
        # Class embedding
        y_emb = self.class_embedding(y)

        # print(t.shape)
        # print(y_emb.shape)
        # Concatenate latent variable with class embedding
        z = torch.cat([t, y_emb], dim=1)

        n = len(x)

        out1 = self.b1(x + self.te1(z).reshape(n, -1, 1, 1))  # (N, 10, 28, 28)
        out2 = self.b2(self.down1(out1) + self.te2(z).reshape(n, -1, 1, 1))  # (N, 20, 14, 14)
        out3 = self.b3(self.down2(out2) + self.te3(z).reshape(n, -1, 1, 1))  # (N, 40, 7, 7)

        out_mid = self.b_mid(self.down3(out3) + self.te_mid(z).reshape(n, -1, 1, 1))  # (N, 40, 3, 3)

        out4 = torch.cat((out3, self.up1(out_mid)), dim=1)  # (N, 80, 7, 7)
        out4 = self.b4(out4 + self.te4(z).reshape(n, -1, 1, 1))  # (N, 20, 7, 7)

        out5 = torch.cat((out2, self.up2(out4)), dim=1)  # (N, 40, 14, 14)
        out5 = self.b5(out5 + self.te5(z).reshape(n, -1, 1, 1))  # (N, 10, 14, 14)

        out = torch.cat((out1, self.up3(out5)), dim=1)  # (N, 20, 28, 28)
        out = self.b_out(out + self.te_out(z).reshape(n, -1, 1, 1))  # (N, 1, 28, 28)

        out = self.conv_out(out)

        return out

    def _make_te(self, dim_in, dim_out):
        return nn.Sequential(
            nn.Linear(dim_in, dim_out),
            nn.SiLU(),
            nn.Linear(dim_out, dim_out),
        )

class MyBlock(nn.Module):
    def __init__(self, shape, in_c, out_c, kernel_size=3, stride=1, padding=1, activation=None, normalize=True):
        super(MyBlock, self).__init__()
        self.ln = nn.LayerNorm(shape)
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size, stride, padding)
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size, stride, padding)
        self.activation = nn.SiLU() if activation is None else activation
        self.normalize = normalize

    def forward(self, x):
        out = self.ln(x) if self.normalize else x
        out = self.conv1(out)
        out = self.activation(out)
        out = self.conv2(out)
        out = self.activation(out)
        return out

class DiffusionModel(nn.Module):
    """
    The whole diffusion model put together
    """
    def __init__(self, network: nn.Module, var_scheduler: VarianceScheduler):
        """

        Args:
            network (nn.Module): your noise estimating network
            var_scheduler (VarianceScheduler): variance scheduler for getting
                                the statistical variables and the noisy images
        """

        super().__init__()

        self.network = network
        self.var_scheduler = var_scheduler

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.float32:
        """
        The forward method for the diffusion model gets the input images and
        their corresponding labels

        Args:
            x (torch.Tensor): the input image [B, 1, 28, 28]
            y (torch.Tensor): labels [B]

        Returns:
            torch.float32: the loss between the actual noises and the estimated noise
        """

        # step1: sample timesteps
        # step2: compute the noisy versions of the input image according to your timesteps
        # step3: estimate the noises using your noise estimating network
        # step4: compute the loss between the estimated noises and the true noises

        timesteps = torch.randint(0, len(self.var_scheduler.betas), (x.size(0),)).to(x.device)
        noisy_x, true_noise = self.var_scheduler.add_noise(x, timesteps)
        estimated_noise = self.network(noisy_x, timesteps, y)
        loss = F.mse_loss(estimated_noise, true_noise)

        return loss

    @torch.no_grad()
    def generate_sample(self, num_images: int, y, device) -> torch.Tensor:

        # Create expanded labels for each image to be generated
        with torch.no_grad():
          if device is None:
            device = torch.device('cuda')

          x  = torch.randn(num_images,1,28,28).to(device)  

          for idx, t in enumerate(list(range(1000))[::-1]):
              # Estimating noise to be removed
              time_tensor = (torch.ones(num_images) * t).to(device).long()
              eta_theta = self.network(x,time_tensor,y)

              alpha_t = self.var_scheduler.alphas[t]
              alpha_t_bar = self.var_scheduler.alpha_bars[t]

              # Partially denoising the image
              x = (1 / alpha_t.sqrt()) * (x - (1 - alpha_t) / (1 - alpha_t_bar).sqrt() * eta_theta)

              if t > 0:
                  z = torch.randn(num_images, 1, 28, 28).to(device)

                  # Option 1: sigma_t squared = beta_t
                  beta_t = self.var_scheduler.betas[t]
                  sigma_t = beta_t.sqrt()

                  # Option 2: sigma_t squared = beta_tilda_t
                  # prev_alpha_t_bar = ddpm.alpha_bars[t-1] if t > 0 else ddpm.alphas[0]
                  # beta_tilda_t = ((1 - prev_alpha_t_bar)/(1 - alpha_t_bar)) * beta_t
                  # sigma_t = beta_tilda_t.sqrt()

                  # Adding some more noise like in Langevin Dynamics fashion
                  x = x + sigma_t * z     

        return x


def load_diffusion_and_generate():
    device = torch.device('cuda')
    var_scheduler = VarianceScheduler()  # define your variance scheduler
    network = NoiseEstimatingNet(64, 64, 10) # define your noise estimating network
    diffusion = DiffusionModel(network=network, var_scheduler=var_scheduler) # define your diffusion model

    # loading the weights of VAE
    diffusion.load_state_dict(torch.load('diffusion.pt'))
    diffusion = diffusion.to(device)

    desired_labels = []
    for i in range(10):
        for _ in range(5):
            desired_labels.append(i)

    desired_labels = torch.tensor(desired_labels).to(device)
    generated_samples = diffusion.generate_sample(50, desired_labels, device)

    return generated_samples