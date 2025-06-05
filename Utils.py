# IMPORTS
import random
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

import torch
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

import seaborn as sns
from scipy.stats import normaltest, shapiro, kurtosis, skew

# Module responsible for the implementation of the functions used in the project notably the functions used to analyze the latent space of the model and used to
# plot the results of the analysis. Note that this module is not the core of the project, it is only used to visualize the results and make some data processing.

def cosine_distance(z1, z2):
    """
    Computes the cosine distance that measures the level of perpendicularity between two vectors.
    """
    return 1 - abs(torch.dot(z1, z2) / (torch.norm(z1) * torch.norm(z2)))

def compute_latent_distance(autoencoder, signal_ref, signal_var):
    """
    Computes the Euclidean distance L2 between 2 vectors here latent.
    """
 
    if isinstance(signal_ref, np.ndarray):
        signal_ref = torch.tensor(signal_ref, dtype=torch.float32)
    if isinstance(signal_var, np.ndarray):
        signal_var = torch.tensor(signal_var, dtype=torch.float32)
    
    device = next(autoencoder.parameters()).device
    signal_ref = signal_ref.to(device)
    signal_var = signal_var.to(device)

    _, latent_ref = autoencoder(signal_ref)
    _, latent_var = autoencoder(signal_var)

    latent_distance = torch.norm(latent_ref - latent_var, p=2).item()
    return latent_distance

def compute_vector_distances(latent_vectors):
    """
    Computes the Euclidean distances between all vectors in a batch.
    """
    distances = torch.cdist(latent_vectors, latent_vectors, p=2) 
    return distances


def generate_varied_signal(generator, a_fixed, b_fixed, w_fixed, phi_fixed, param_name, value, nb_points, Fe, noise_std, normalize):
    """
    Génère un signal en faisant varier un paramètre spécifique.
    """
    b = b_fixed.copy()
    w = w_fixed.copy()
    phi = phi_fixed.copy()
    a = a_fixed.copy()

    if param_name.startswith("b"):
        b[int(param_name[1])] = value
    elif param_name.startswith("a"):
        a[int(param_name[1])] = value
    elif param_name.startswith("w"):
        w[int(param_name[1])] = value
    elif param_name.startswith("phi"):
        phi[int(param_name[3])] = value

    w.sort()
    signal_var = generator.generate_signal(
        a=a, b=b, w=w, phi=phi, nb_points=nb_points, Fe=Fe, noise_std=noise_std, normalize=normalize
    )
    return signal_var, (a, b, w, phi)

def compute_distances(generator, autoencoder, a_fixed, b_fixed, w_fixed, phi_fixed, param_ranges, nb_points, Fe, noise_std, normalize):
    """
    Computes the distances in latent space and parameter space for each variation.
    """
    signal_ref = generator.generate_signal(
        a=a_fixed, b=b_fixed, w=w_fixed, phi=phi_fixed, nb_points=nb_points, Fe=Fe, noise_std=noise_std, normalize=normalize
    )

    latent_distances = {}
    param_distances = {}

    for param_name, param_range in param_ranges.items():
        latent_dist = []
        param_dist = []
        for value in param_range:
            signal_var, (a, b, w, phi) = generate_varied_signal(generator, a_fixed, b_fixed, w_fixed, phi_fixed, param_name, value,  nb_points=nb_points, Fe=Fe, noise_std=noise_std, normalize=normalize)

            latent_distance = compute_latent_distance(autoencoder, signal_ref, signal_var)
            latent_dist.append(latent_distance)

            ref_params = np.concatenate([a_fixed, b_fixed, w_fixed, phi_fixed])
            var_params = np.concatenate([a, b, w, phi])
            param_distance = np.linalg.norm(ref_params - var_params)
            param_dist.append(param_distance)

        latent_distances[param_name] = latent_dist
        param_distances[param_name] = param_dist

    return latent_distances, param_distances

def plot_variations(latent_distances, param_distances, param_ranges, selected_params, param_groups):
    """
    Displays graphs of distance variations in latent space and parameters input space.
    """
    filtered_latent_distances = {key: latent_distances[key] for key in selected_params if key in latent_distances}
    filtered_param_distances = {key: param_distances[key] for key in selected_params if key in param_distances}
    filtered_param_ranges = {key: param_ranges[key] for key in selected_params if key in param_ranges}

    num_rows = len(param_groups)
    num_cols = 4  

    plt.figure(figsize=(16, 4 * num_rows))

    for row_idx, group in enumerate(param_groups):
        for col_idx, param_name in enumerate(group):
            plt.subplot(num_rows, num_cols, row_idx * num_cols + 2 * col_idx + 1)
            plt.plot(
                filtered_param_ranges[param_name], 
                filtered_latent_distances[param_name], 
                label=f"Latent Distance ({param_name})", 
                color="blue"
            )
            plt.title(f"Variation of {param_name} (Latent Space)")
            plt.xlabel(f"{param_name} Value")
            plt.ylabel("Latent Distance")
            plt.legend()
            plt.grid(True)

            plt.subplot(num_rows, num_cols, row_idx * num_cols + 2 * col_idx + 2)
            plt.plot(
                filtered_param_ranges[param_name], 
                filtered_param_distances[param_name], 
                label=f"Parameter Distance ({param_name})", 
                color="orange"
            )
            plt.title(f"Variation of {param_name} (Parameter Space)")
            plt.xlabel(f"{param_name} Value")
            plt.ylabel("Parameter Distance")
            plt.legend()
            plt.grid(True)

    plt.tight_layout()
    plt.show()

def compute_latent_vectors(dataset, autoencoder):
    """
    Function that returns a batch of latent vectors for a dataset of generated synthetic signals and a pre-trained model.
    """
   
    latent_vectors = []
    for signal, _ in tqdm(dataset, desc='Latent vectors encoder'):

        signal_tensor = torch.tensor(signal, dtype=torch.float32)
        with torch.no_grad():
            _, latent = autoencoder(signal_tensor)  
        latent_vectors.append(latent.squeeze().numpy()) 

    latent_vectors = np.array(latent_vectors)

    return latent_vectors

def compute_reconstructed_signals(dataset, autoencoder):
    """
    Generate a reconstructed signal batch for a synthetic dataset and a pre-trained model
    """

    signals = [item[0] for item in dataset]
    signals_tensor = torch.tensor(signals, dtype=torch.float32)
    with torch.no_grad():
        reconstructed_signals, _ = autoencoder(signals_tensor)  

    return reconstructed_signals.numpy()


def get_latent_vectors_variations(generator, autoencoder, a_fixed, b_fixed, w_fixed, phi_fixed, param_ranges, nb_points, Fe, noise_std, normalize):
    """
    Function that returns a dictionary of latent vectors associated with signals whose parameters are varied one by one independently of each other, in order to 
    calculate the trajectories of the latent vectors according to the parameters of interest.
    """

    latent_vectors = {}
    device = next(autoencoder.parameters()).device
    autoencoder.eval()

    for param_name, param_range in param_ranges.items():
        batch_signals = []
        
        for value in param_range:
            signal, _ = generate_varied_signal(
                generator=generator,
                a_fixed=a_fixed,
                b_fixed=b_fixed,
                w_fixed=w_fixed,
                phi_fixed=phi_fixed,
                param_name=param_name,
                value=value,
                nb_points=nb_points,
                Fe=Fe,
                noise_std=noise_std,
                normalize=normalize
            )
            batch_signals.append(signal)

        signals_tensor = torch.tensor(np.array(batch_signals), dtype=torch.float32).to(device)
        
        with torch.no_grad():
            _, latents = autoencoder(signals_tensor)
        
        latent_vectors[param_name] = latents.cpu().numpy()

    return latent_vectors

def analyze_latent_distribution(autoencoder, signals_tensor, n_bins=30):
    """
    Analyzes and visualizes the distribution of latent values for the first 8 dimensions.
    """

    with torch.no_grad():
        _, latent_vectors = autoencoder(signals_tensor)
    latent_vectors = latent_vectors.cpu().numpy()

    n_latent_dims = latent_vectors.shape[1]

    stats = {}

    plt.figure(figsize=(20, 10)) 
    for dim in range(0, min(8, n_latent_dims)):  
        values = latent_vectors[:, dim]
        k2, p_normaltest = normaltest(values)
        stat_shapiro, p_shapiro = shapiro(values)  
        kurt = kurtosis(values) 
        skw = skew(values) 

        stats[dim] = {
            "mean": np.mean(values),
            "std": np.std(values),
            "kurtosis": kurt,
            "skewness": skw,
            "normaltest_p": p_normaltest,
            "shapiro_p": p_shapiro,
        }

        plt.subplot(4, 2, dim + 1)  
        sns.histplot(values, bins=n_bins, kde=True, color="skyblue", alpha=0.7)
        plt.title(f"Latent Dim {dim}", fontsize=14)
        plt.xlabel("Value", fontsize=12)
        plt.ylabel("Frequency", fontsize=12)
        plt.grid(True, alpha=0.6)
        plt.annotate(
            f"Mean: {stats[dim]['mean']:.2f}\n"
            f"Std: {stats[dim]['std']:.2f}\n"
            f"Kurt: {stats[dim]['kurtosis']:.2f}\n"
            f"Skew: {stats[dim]['skewness']:.2f}\n"
            f"p (Normal): {stats[dim]['normaltest_p']:.2e}\n"
            f"p (Shapiro): {stats[dim]['shapiro_p']:.2e}",
            xy=(0.65, 0.6),
            xycoords="axes fraction",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", edgecolor="gray", facecolor="white", alpha=0.7),
        )

    plt.tight_layout()
    plt.show()

    return stats

def plot_latent_variance_mean(autoencoder, signals_tensor):
    """
    Plot histograms of variances and means for all latent dimensions.
    """

    with torch.no_grad():
        _, latent_vectors = autoencoder(signals_tensor)
    latent_vectors = latent_vectors.cpu().numpy()

    means = np.mean(latent_vectors, axis=0)
    variances = np.var(latent_vectors, axis=0)

    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    sns.histplot(variances, bins=20, kde=True, color="skyblue", alpha=0.7)
    plt.title("Variance Histogram", fontsize=14)
    plt.xlabel("Variance", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.grid(True, alpha=0.6)

    plt.subplot(1, 2, 2)
    sns.histplot(means, bins=20, kde=True, color="orange", alpha=0.7)
    plt.title("Mean Histogram", fontsize=14)
    plt.xlabel("Mean", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.grid(True, alpha=0.6)

    plt.tight_layout()
    plt.show()

def reconstruct_signal_from_params(params, segment_length=1024):
    """
    Reconstructs an ecg signal from argument parameters, using the Fourier inversion formula.
    """
    a = np.array(params["a"]) 
    w = np.array(params["w"]) 
    phi = np.array(params["phi"]) 

    t = np.arange(segment_length) / segment_length
    signal = np.zeros(segment_length)
    for i in range(len(a)):
        signal += a[i] * np.cos(2 * np.pi * w[i] * t + phi[i])
    return signal

def plot_ecg_latent_space_variations(autoencoder, full_dataset, n_points=1000):
    """
    Displays the displacement of latent vectors when the first 3 prameters a1, w1, phi1 of an ecg signal are varied. 
    """

    example_signal, example_params = full_dataset[0]
    a = np.array(example_params["a"]) 
    w = np.array(example_params["w"])  
    phi = np.array(example_params["phi"]) 

    fig, axs = plt.subplots(1, 3, figsize=(18, 6))

    component_types = ["frequency", "amplitude", "phase"]
    titles = ["Frequency Variation w1", "Amplitude Variation a1", "Phase Variation phi1"]
    cmaps = ["viridis", "viridis", "viridis"]

    for idx, (component_type, title, cmap) in enumerate(zip(component_types, titles, cmaps)):

        if component_type == "frequency":
            original_value = w[3]
            variation_values = np.linspace(original_value, original_value + 1, n_points)
        elif component_type == "amplitude":
            original_value = a[0]
            variation_values = np.linspace(original_value - 0.5, original_value + 10, n_points)
        elif component_type == "phase":
            original_value = phi[0]
            variation_values = np.linspace(original_value - np.pi, original_value + np.pi, n_points)

        latent_vectors = []

        for value in variation_values:

            if component_type == "frequency":
                w[0] = value
            elif component_type == "amplitude":
                a[0] = value
            elif component_type == "phase":
                phi[0] = value

            modified_params = {"a": a.tolist(), "w": w.tolist(), "phi": phi.tolist()}
            modified_signal = reconstruct_signal_from_params(modified_params)

            modified_signal_tensor = torch.tensor(modified_signal, dtype=torch.float32).unsqueeze(0)

            with torch.no_grad():
                _, latent = autoencoder(modified_signal_tensor)
            latent_vectors.append(latent.numpy())

        latent_vectors = np.concatenate(latent_vectors, axis=0)

        pca = PCA(n_components=2)
        latent_pca = pca.fit_transform(latent_vectors)

        scatter = axs[idx].scatter(latent_pca[:, 0], latent_pca[:, 1], c=variation_values, cmap=cmap, alpha=0.7)
        fig.colorbar(scatter, ax=axs[idx], label=f"{component_type.capitalize()} Value")
        axs[idx].set_title(title, fontsize=14)
        axs[idx].set_xlabel("PCA Component 1", fontsize=12)
        axs[idx].set_ylabel("PCA Component 2", fontsize=12)
        axs[idx].grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.show()

def plot_pca(latent_vectors):
    """
    Displays the PCA visualization of the latent space and the cumulative explained variance.
    """
    pca = PCA()
    pca.fit(latent_vectors)

    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    components = np.arange(1, len(cumulative_variance) + 1)

    plt.figure(figsize=(16, 8))

    plt.subplot(1, 2, 1) 
    latent_2d = PCA(n_components=2).fit_transform(latent_vectors)
    plt.scatter(latent_2d[:, 0], latent_2d[:, 1], alpha=0.7, label="Signals")
    plt.title("PCA Visualization of Latent Space")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2) 
    plt.plot(components, cumulative_variance, marker='o', linestyle='-', label='Cumulative Variance')
    plt.axhline(y=0.95, color='r', linestyle='--', label='95% Variance Threshold')
    plt.axhline(y=0.90, color='g', linestyle='--', label='90% Variance Threshold')
    plt.title("Cumulative Explained Variance by PCA Components")
    plt.xlabel("Number of PCA Components")
    plt.ylabel("Cumulative Explained Variance")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


def find_closest_and_farthest_pairs(distances):
    """
    Finds the closest and farthest pairs of signals based on the computed distances.
    """

    distances.fill_diagonal_(float('inf'))

    valid_distances = distances[distances != float('inf')]  

    closest_distance = valid_distances.min()
    closest_indices = (distances == closest_distance).nonzero(as_tuple=True)
    closest_pair_indices = (closest_indices[0][0].item(), closest_indices[1][0].item())
    
    farthest_distance = valid_distances.max()
    farthest_indices = (distances == farthest_distance).nonzero(as_tuple=True)
    farthest_pair_indices = (farthest_indices[0][0].item(), farthest_indices[1][0].item())
    
    return closest_pair_indices, farthest_pair_indices, closest_distance.item(), farthest_distance.item()

def plot_signals(signals, titles, colors, distances, time_axis):
    """
    Plot signals with the distance displayed.
    """
    plt.figure(figsize=(18, 6))  
    
    for i, (signal_pair, title, color_pair, distance) in enumerate(zip(signals, titles, colors, distances)):
        plt.subplot(1, 3, i + 1) 
        for j, (signal, color) in enumerate(zip(signal_pair, color_pair)):
            plt.plot(time_axis, signal, color=color, linewidth=1.5, label=f"Signal {j + 1}")
        plt.title(f"{title}\nDistance: {distance:.4f}", fontsize=12)
        plt.xlabel("Time (s)", fontsize=10)
        plt.ylabel("Amplitude", fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.xticks(fontsize=8)
        plt.yticks(fontsize=8)
        plt.legend(fontsize=8)
    
    plt.tight_layout()
    plt.show()