# IMPORTS
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt


def compute_loss(x, x_recon, z, params, alpha=1.0, beta=1.0):
    """
    Function to compute the total loss, reconstruction loss, and distance loss of the model
    """

    recon_loss = nn.MSELoss()(x_recon, x)
    
    pairwise_dist_params = torch.cdist(params, params, p=2)
    pairwise_dist_latent = torch.cdist(z, z, p=2)
    dist_loss = torch.mean((pairwise_dist_params - pairwise_dist_latent) ** 2)

    total_loss = alpha * recon_loss + beta * dist_loss
    return total_loss, recon_loss, dist_loss

def train(autoencoder, train_loader, params_train, epochs=20, alpha=1.0, beta=1.0, lr=0.001, verbose=True):
    """
    Function to train the model
    """

    optimizer = optim.Adam(autoencoder.parameters(), lr=lr)
    
    train_losses = []
    recon_losses = []
    dist_losses = []

    if verbose:
            print("=" * 50)
            print(" " * 15 + "Training Parameters")
            print("=" * 50)
            print(f"{'Parameter':<20} | {'Value':<20}")
            print("-" * 50)
            print(f"{'Epochs':<20} | {epochs:<20}")
            print(f"{'Learning rate':<20} | {lr:<20}")
            print(f"{'Alpha':<20} | {alpha:<20}")
            print(f"{'Beta':<20} | {beta:<20}")
            print("-" * 50)

    for epoch in tqdm(range(epochs), desc='Training'):
        autoencoder.train()
        epoch_loss = 0.0
        epoch_recon = 0.0
        epoch_dist = 0.0

        for signals, indices in train_loader:
            optimizer.zero_grad()
            
                
            recon, z = autoencoder(signals)
                
            loss, recon_loss, dist_loss = compute_loss(signals, recon, z, params_train[indices], alpha, beta)
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            epoch_recon += recon_loss.item()
            epoch_dist += dist_loss.item()
            
        train_losses.append(epoch_loss / len(train_loader))
        recon_losses.append(epoch_recon / len(train_loader))
        dist_losses.append(epoch_dist / len(train_loader))

        if verbose and (epoch % 2 == 0 or epoch == epochs-1):
            print(f'Epoch {epoch+1}/{epochs} | '
                  f'Loss: {train_losses[-1]:.4f} | '
                  f'Recon: {recon_losses[-1]:.4f} | '
                  f'Dist: {dist_losses[-1]:.4f}')

    return train_losses, recon_losses, dist_losses

def plot_loss(train_losses, recon_losses, dist_losses, figsize=(10, 6)):
    """
    Function to plot the evolution of the loss
    """
    plt.figure(figsize=figsize)

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, color=colors[0], label='Loss Totale', linewidth=2)
    plt.plot(epochs, recon_losses, color=colors[1], label='Loss Reconstruction', linewidth=2)
    plt.plot(epochs, dist_losses, color=colors[2], label='Loss Distance', linewidth=2)

    plt.title('Loss evolution', fontsize=14)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Loss values', fontsize=12)
    
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    
    plt.xlim(1, len(epochs))
    plt.ylim(0, max(train_losses)*1.1)
    
    plt.tight_layout()
    plt.show()