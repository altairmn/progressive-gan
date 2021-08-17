import torch.nn as nn
import torch.nn.functional as F
import torch
import time
import datetime
from torchsummary import summary
from data_loader import get_loader
import math
import os

from torch.utils.tensorboard import SummaryWriter

def getHeMultiplier(tensor):
    sz = tensor.size()
    fan_in = math.prod(sz[1:])
    return math.sqrt(2.0/fan_in)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.inputs = nn.ModuleList()
        self.layers = nn.ModuleList()
        self.output = nn.Sequential(
            nn.Conv2d(in_channels=513,out_channels=512,kernel_size=3,padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels=512,out_channels=512,kernel_size=4),
            nn.LeakyReLU(0.2))
        self.fc = nn.Linear(in_features=512,out_features=1)


        self.channel_dims = [512, 512, 512, 512, 256, 128, 64, 32, 16]
        self.progress = 0

        # setup for first step
        self.inputs.insert(0, nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=self.channel_dims[self.progress], kernel_size=1),
            nn.LeakyReLU(0.2)))
        
        self.layers.insert(0, nn.Identity())

        self.alpha = 0

    def stddev(self, x):
        """ Compute stddev
        """
        y = torch.sqrt(x.square().mean(dim=0) - x.mean(dim=0).square())
        return torch.mean(y)

    def transition(self):
        if self.alpha == 0.0:
            self.progress = self.progress + 1
            self.alpha = 0.5
            self.inputs.insert(0, nn.Conv2d(in_channels=3,out_channels=self.channel_dims[self.progress],kernel_size=1))
            self.layers.insert(0, nn.Sequential(
                nn.Conv2d(in_channels=self.channel_dims[self.progress],out_channels=self.channel_dims[self.progress],kernel_size=3,padding=1),
                nn.LeakyReLU(0.2),
                nn.Conv2d(in_channels=self.channel_dims[self.progress],out_channels=self.channel_dims[self.progress-1],kernel_size=3,padding=1),
                nn.LeakyReLU(0.2),
                nn.AvgPool2d(kernel_size=2,stride=2)))
            print("Adding transition layers")
        else:
            self.alpha = 0.0
            print("Completing transition")
        return self.progress

    def forward(self, img):
        x = self.inputs[0](img)
        x = F.leaky_relu(x, negative_slope=0.2, inplace=True)
        x = self.layers[0](x)

        if self.alpha > 0:
            y = F.avg_pool2d(img, kernel_size = 2, stride = 2)
            y = self.inputs[1](y)
            x = self.alpha * x + (1 - self.alpha) * y
 
        for layer in self.layers[1:]:
            x = layer(x)

        # concat stddev layer
        xdev = self.stddev(x.detach())
        xdev = xdev.repeat(x.size(0), 1, x.size(2), x.size(3))
        x = torch.cat((x, xdev), dim=1)

        x = self.output(x)
        x = x.view(-1, 512)
        x = self.fc(x)
        return x

    def apply_elr(self):
        for param in self.named_parameters():
            if "weight" in param[0]:
                param[1].data = param[1].data * getHeMultiplier(param[1].data)

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        # normalize after reshaping
        self.input = nn.ModuleList([
            nn.Linear(in_features=512, out_features=512*4*4),
            nn.Conv2d(512, 512, kernel_size=3,padding=1)])

        self.layers = nn.ModuleList()
        self.outputs = nn.ModuleList()

        self.channel_dims = [512, 512, 512, 512, 256, 128, 64, 32, 16]
        self.progress = 0

        # setup for first step
        self.outputs.append(nn.Conv2d(self.channel_dims[self.progress],3,kernel_size=1))

        self.layers.append(nn.Identity())

        self.progress = 0
        self.alpha = 0.0

    def transition(self):
        if self.alpha == 0:
            self.progress = self.progress + 1
            self.alpha = 0.5
            self.outputs.append(nn.Conv2d(self.channel_dims[self.progress],3,kernel_size=1))
            self.layers.append(nn.Sequential(
                nn.Upsample(scale_factor = 2, mode = 'nearest'),
                nn.Conv2d(in_channels=self.channel_dims[self.progress-1],out_channels=self.channel_dims[self.progress],kernel_size=3,padding=1),
                nn.LocalResponseNorm(size = 2 * self.channel_dims[self.progress], alpha = 2, beta = 0.5, k = 1e-8),
                nn.LeakyReLU(negative_slope = 0.2),
                nn.Conv2d(in_channels=self.channel_dims[self.progress],out_channels=self.channel_dims[self.progress],kernel_size=3,padding=1),
                nn.LocalResponseNorm(size = 2 * self.channel_dims[self.progress], alpha = 2, beta = 0.5, k = 1e-8),
                nn.LeakyReLU(negative_slope = 0.2)
            ))
        else:
            self.alpha = 0

    def forward(self, x):
        # input section
        # x = torch.randn(batch_size, 512)
        x = x.divide(x.norm(dim=1, keepdim=True))
        x = F.leaky_relu(self.input[0](x), 0.2, True)
        x = x.view(-1, 512, 4, 4)
        x = F.leaky_relu(self.input[1](x), 0.2, True)
        x = F.local_response_norm(x, size=2*512, alpha=2, beta=0.5, k=1e-8)

        # pass through hidden layers
        for layer in self.layers[:-1]:
            x = layer(x)

        y = x
        x = self.layers[-1](x)
        x = self.outputs[-1](x)
        if self.alpha > 0:
            y = F.interpolate(y,scale_factor=2, mode='nearest')
            y = self.outputs[-2](y)
            x = self.alpha * x + (1 - self.alpha) * y
        x = x.tanh()
        return x

    def apply_elr(self):
        for param in self.named_parameters():
            if "weight" in param[0]:
                param[1].data = param[1].data * getHeMultiplier(param[1].data)

class Solver(object):
    """Solver for training and testing Progressive GAN."""

    def __init__(self):
        """Initialize configurations."""

        # Model configurations.
        self.lambda_gp = 10

        # Training configurations.
        self.image_sizes = [4, 8, 16, 32, 64, 128, 256]
        self.batch_sizes = [16, 16, 16, 16, 16, 16, 14]
        self.num_iters = 800000
        self.g_lr = 0.001
        self.d_lr = 0.001
        self.n_critic = 1
        self.beta1 = 0
        self.beta2 = 0.99
        self.eps = 1e-8
        self.epochs = 10

        # Miscellaneous.
        self.use_tensorboard = True
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Directories.
        self.data_dir = "./data"
        self.tensorboard_dir = "./runs"
        self.log_dir = "./logs"
        self.sample_dir = "./samples"
        self.result_dir = "./results"
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        if not os.path.exists(self.tensorboard_dir):
            os.makedirs(self.tensorboard_dir)
        if not os.path.exists(self.sample_dir):
            os.makedirs(self.sample_dir)
        if not os.path.exists(self.result_dir):
            os.makedirs(self.result_dir)

        # Step size.
        self.log_step = 10
        self.sample_step = 1000

        # Build the model and tensorboard.
        self.build_model()
        if self.use_tensorboard:
            self.build_tensorboard()

    @torch.no_grad()
    def init_weights(self, module):
        if type(module) == nn.Conv2d or type(module) == nn.Linear:
            nn.init.normal_(module.weight, mean=0., std=1.)
            nn.init.zeros_(module.bias)

    def build_model(self):
        """Create a generator and a discriminator."""
        self.G = Generator()
        self.D = Discriminator() 

        # apply weight
        self.G.apply(self.init_weights)
        self.D.apply(self.init_weights)

        self.g_optimizer = torch.optim.Adam(self.G.parameters(), self.g_lr, [self.beta1, self.beta2], self.eps)
        self.d_optimizer = torch.optim.Adam(self.D.parameters(), self.d_lr, [self.beta1, self.beta2], self.eps)

        self.G.to(self.device)
        self.D.to(self.device)

    def print_network(self, model, name):
        """Print out the network information."""
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(model)
        print(name)
        print("The number of parameters: {}".format(num_params))

    def build_tensorboard(self):
        """Build a tensorboard logger."""
        self.writer = SummaryWriter(f'runs/celebahq/quickstart_tb')

    def reset_grad(self):
        """Reset the gradient buffers."""
        self.g_optimizer.zero_grad()
        self.d_optimizer.zero_grad()

    def gradient_penalty(self, y, x):
        """Compute gradient penalty: (L2_norm(dy/dx) - 1)**2."""
        weight = torch.ones(y.size()).to(self.device)
        dydx = torch.autograd.grad(outputs=y,
                                   inputs=x,
                                   grad_outputs=weight,
                                   retain_graph=True,
                                   create_graph=True,
                                   only_inputs=True)[0]

        dydx = dydx.view(dydx.size(0), -1)
        dydx_l2norm = torch.sqrt(torch.sum(dydx**2, dim=1))
        return torch.mean((dydx_l2norm-1)**2)

    def train(self):
        """Train StarGAN within a single dataset."""
        # Progress
        progress = 0

        # Learning rate cache for decaying.
        g_lr = self.g_lr
        d_lr = self.d_lr

        # epochs; training will run for this many epochs; each epoch indicates transition
        epochs = self.epochs

        # Start training from scratch or resume training.
        start_iters = 0

        for epoch in range(epochs):

            # reinitialize writer
            self.writer = SummaryWriter(f'runs/celebahq/{epoch} w/ progress {progress}')

            # Start training.
            print(f'Start training for epoch {epoch}...')
            start_time = time.time()
            batch_size = self.batch_sizes[progress]
            image_size = self.image_sizes[progress]
            data_loader = get_loader(self.data_dir, image_size, batch_size)
            epoch_iters = self.num_iters//batch_size + 1
            for i in range(start_iters, epoch_iters):

                # =================================================================================== #
                #                             1. Preprocess input data                                #
                # =================================================================================== #

                # Fetch real images and labels.
                try:
                    x_real, _ = next(data_iter)
                except:
                    data_iter = iter(data_loader)
                    x_real, _ = next(data_iter)

                x_real = x_real.to(self.device)           # Input images.

                # =================================================================================== #
                #                             2. Train the discriminator                              #
                # =================================================================================== #

                # Compute loss with real images.
                out_src = self.D(x_real)
                d_loss_real = - torch.mean(out_src)

                # Compute loss with fake images.
                x_latent = torch.randn(batch_size, 512)
                x_fake = self.G(x_latent)
                out_src = self.D(x_fake.detach()) # don't want to train G here
                d_loss_fake = torch.mean(out_src)

                # Compute loss for gradient penalty.
                alpha = torch.rand(x_real.size(0), 1, 1, 1).to(self.device)
                x_hat = (alpha * x_real.data + (1 - alpha) * x_fake.data).requires_grad_(True)
                out_src = self.D(x_hat)
                d_loss_gp = self.gradient_penalty(out_src, x_hat)

                # Backward and optimize.
                d_loss = d_loss_real + d_loss_fake + self.lambda_gp * d_loss_gp
                self.reset_grad()
                d_loss.backward()
                self.d_optimizer.step()
                #self.D.apply_elr()

                # Logging.
                loss = {}
                loss['D/loss_real'] = d_loss_real.item()
                loss['D/loss_fake'] = d_loss_fake.item()
                loss['D/loss_gp'] = d_loss_gp.item()
                
                # =================================================================================== #
                #                               3. Train the generator                                #
                # =================================================================================== #
                
                if (i+1) % self.n_critic == 0:
                    # Original-to-target domain.
                    x_fake = self.G(x_latent)
                    out_src = self.D(x_fake)
                    g_loss_fake = -torch.mean(out_src)

                    # Backward and optimize.
                    g_loss = g_loss_fake
                    self.reset_grad()
                    g_loss.backward()
                    self.g_optimizer.step()
                    #self.G.apply_elr()

                    # Logging.
                    loss['G/loss_fake'] = g_loss_fake.item()

                # =================================================================================== #
                #                                 4. Miscellaneous                                    #
                # =================================================================================== #

                # Print out training information.
                if (i+1) % self.log_step == 0:
                    et = time.time() - start_time
                    et = str(datetime.timedelta(seconds=et))[:-7]
                    log = "Elapsed [{}], Iteration [{}/{}]".format(et, i+1, epoch_iters)
                    for tag, value in loss.items():
                        log += ", {}: {:.4f}".format(tag, value)
                    print(log)

                    if self.use_tensorboard:
                        for tag, value in loss.items():
                            self.writer.add_scalar(tag, value, i+1)

        print(f"Finished epoch {epoch}.")
        g_progress = self.G.transition()
        print(f"Transitioned Generator to {g_progress}")
        d_progress = self.D.transition()
        print(f"Transitioned Discriminator to {d_progress}")
        if g_progress != d_progress:
            print(f"ERROR: The discriminator and the generator are not progressing equally")
            exit(0)
        progress = g_progress

if __name__=="__main__":
    solver = Solver()
    solver.train()