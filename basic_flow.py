import os
import argparse

import numpy as np
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.autograd import Variable
from matplotlib import pyplot as plt
from mag.experiment import Experiment


parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)

parser.add_argument(
    "--log_interval", type=int, default=300,
    help="How frequenlty to print the training stats."
)
parser.add_argument(
    "--plot_interval", type=int, default=300,
    help="How frequenlty to plot samples from current distribution."
)
parser.add_argument(
    "--plot_points", type=int, default=1000,
    help="How many to points to generate for one plot."
)

args = parser.parse_args()

torch.manual_seed(42)

X_LIMS = (-7, 7)
Y_LIMS = (-7, 7)


def safe_log(z):
    return torch.log(z + 1e-7)


def p_z(z):

    z1, z2 = torch.chunk(z, chunks=2, dim=1)
    norm = torch.sqrt(z1 ** 2 + z2 ** 2)

    exp1 = torch.exp(-0.5 * ((z1 - 2) / 0.8) ** 2)
    exp2 = torch.exp(-0.5 * ((z1 + 2) / 0.8) ** 2)
    u = 0.5 * ((norm - 4) / 0.4) ** 2 - torch.log(exp1 + exp2)

    return torch.exp(-u)


def random_normal_samples(n):
    return torch.zeros(n, 2).normal_(mean=0, std=1)


def scatter_points(points, directory, iteration, flow_length):

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111)
    ax.scatter(points[:, 0], points[:, 1], alpha=0.7, s=25)
    ax.set_xlim(*X_LIMS)
    ax.set_ylim(*Y_LIMS)
    ax.set_title(
        "Flow length: {}\n Samples on iteration #{}"
        .format(flow_length, iteration)
    )

    fig.savefig(os.path.join(directory, "flow_result_{}.png".format(iteration)))
    plt.close()


def plot_density(density, directory):

    x1 = np.linspace(*X_LIMS, 300)
    x2 = np.linspace(*Y_LIMS, 300)
    x1, x2 = np.meshgrid(x1, x2)
    shape = x1.shape
    x1 = x1.ravel()
    x2 = x2.ravel()

    z = np.c_[x1, x2]
    z = torch.FloatTensor(z)
    z = Variable(z)

    density = p_z(z).data.numpy().reshape(shape)

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111)
    ax.imshow(density, extent=(*X_LIMS, *Y_LIMS), cmap="summer")
    ax.set_title("True density")

    fig.savefig(os.path.join(directory, "density.png"))
    plt.close()


class NormalizingFlow(nn.Module):

    def __init__(self, dim, flow_length):
        super().__init__()

        self.transforms = nn.Sequential(*(
            PlanarFlow(dim) for _ in range(flow_length)
        ))

        self.log_jacobians = nn.Sequential(*(
            PlanarFlowLogDetJacobian(t) for t in self.transforms
        ))

    def forward(self, z):

        log_jacobians = []

        for transform, log_jacobian in zip(self.transforms, self.log_jacobians):
            log_jacobians.append(log_jacobian(z))
            z = transform(z)

        zk = z

        return zk, log_jacobians


class PlanarFlow(nn.Module):

    def __init__(self, dim):
        super().__init__()

        self.weight = nn.Parameter(torch.Tensor(1, dim))
        self.bias = nn.Parameter(torch.Tensor(1))
        self.scale = nn.Parameter(torch.Tensor(1, dim))
        self.tanh = nn.Tanh()

        self.reset_parameters()

    def reset_parameters(self):

        self.weight.data.uniform_(-0.01, 0.01)
        self.scale.data.uniform_(-0.01, 0.01)
        self.bias.data.uniform_(-0.01, 0.01)

    def forward(self, z):

        activation = F.linear(z, self.weight, self.bias)
        return z + self.scale * self.tanh(activation)


class PlanarFlowLogDetJacobian(nn.Module):
    """A helper class to compute the determinant of the gradient of
    the planar flow transformation."""

    def __init__(self, affine):
        super().__init__()

        self.weight = affine.weight
        self.bias = affine.bias
        self.scale = affine.scale
        self.tanh = affine.tanh

    def forward(self, z):

        activation = F.linear(z, self.weight, self.bias)
        psi = (1 - self.tanh(activation) ** 2) * self.weight
        det_grad = 1 + torch.mm(psi, self.scale.t())
        return safe_log(det_grad.abs())


class FreeEnergyBound(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, zk, log_jacobians):

        sum_of_log_jacobians = sum(log_jacobians)
        return (-sum_of_log_jacobians - safe_log(p_z(zk))).mean()


with Experiment({
    "batch_size": 40,
    "iterations": 10000,
    "initial_lr": 0.01,
    "lr_decay": 0.999,
    "flow_length": 4,
    "name": "planar"
}) as experiment:

    config = experiment.config
    experiment.register_directory("samples")
    experiment.register_directory("distributions")

    flow = NormalizingFlow(dim=2, flow_length=config.flow_length)
    bound = FreeEnergyBound()
    optimizer = optim.RMSprop(flow.parameters(), lr=config.initial_lr)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, config.lr_decay)

    plot_density(p_z, directory=experiment.distributions)

    def should_log(iteration):
        return iteration % args.log_interval == 0

    def should_plot(iteration):
        return iteration % args.plot_interval == 0

    for iteration in range(1, config.iterations + 1):

        scheduler.step()

        samples = Variable(random_normal_samples(config.batch_size))
        zk, log_jacobians = flow(samples)

        optimizer.zero_grad()
        loss = bound(zk, log_jacobians)
        loss.backward()
        optimizer.step()

        if should_log(iteration):
            print("Loss on iteration {}: {}".format(iteration , loss.data[0]))

        if should_plot(iteration):
            samples = Variable(random_normal_samples(args.plot_points))
            zk, det_grads = flow(samples)
            scatter_points(
                zk.data.numpy(),
                directory=experiment.samples,
                iteration=iteration,
                flow_length=config.flow_length
            )
