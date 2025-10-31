"""
Visualization functions for latent space analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.manifold import TSNE


def plot_tsne(latent_vectors, labels, instrument_families, save_path, 
              perplexity=5, learning_rate=200, n_iter=1000):
    """
    Create t-SNE visualization colored by instrument family.
    
    Args:
        latent_vectors: [n_samples, latent_dim] array of latent representations
        labels: [n_samples] array of instrument family labels
        instrument_families: List of unique instrument types
        save_path: Path to save the plot
        perplexity: t-SNE perplexity parameter
        learning_rate: t-SNE learning rate
        n_iter: Number of t-SNE iterations
    """
    print(f"\n=== Running t-SNE ===")
    print(f"Samples: {len(latent_vectors)}")
    print(f"Latent dimensions: {latent_vectors.shape[1]}")
    print(f"Perplexity: {perplexity}")
    
    # Run t-SNE
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        learning_rate=learning_rate,
        max_iter=n_iter,
        random_state=42,
        verbose=1
    )
    
    latent_2d = tsne.fit_transform(latent_vectors)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Get unique labels and colors
    unique_labels = np.unique(labels)
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))
    
    # Plot each instrument family
    for i, family in enumerate(unique_labels):
        mask = labels == family
        ax.scatter(
            latent_2d[mask, 0],
            latent_2d[mask, 1],
            c=[colors[i]],
            label=family,
            alpha=0.6,
            s=50,
            edgecolors='black',
            linewidths=0.5
        )
    
    ax.set_xlabel('t-SNE Dimension 1', fontsize=12)
    ax.set_ylabel('t-SNE Dimension 2', fontsize=12)
    ax.set_title('Latent Space Visualization (t-SNE) by Instrument Family', fontsize=14, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved t-SNE plot to {save_path}")


def plot_descriptor_correlation(descriptor_values, latent_values, 
                                descriptor_name, dim_idx, save_path):
    """
    Scatter plot showing correlation between descriptor and latent dimension.
    
    Args:
        descriptor_values: [n_samples] array of descriptor values
        latent_values: [n_samples] array of values from one latent dimension
        descriptor_name: Name of the descriptor (e.g., 'Spectral Centroid')
        dim_idx: Index of the latent dimension
        save_path: Path to save the plot
    """
    from scipy.stats import pearsonr
    
    # Compute correlation
    r, p_value = pearsonr(descriptor_values, latent_values)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Scatter plot
    ax.scatter(descriptor_values, latent_values, alpha=0.5, s=20, edgecolors='black', linewidths=0.5)
    
    # Add regression line
    z = np.polyfit(descriptor_values, latent_values, 1)
    p = np.poly1d(z)
    x_line = np.linspace(descriptor_values.min(), descriptor_values.max(), 100)
    ax.plot(x_line, p(x_line), 'r--', linewidth=2, label='Linear fit')
    
    # Labels and title
    ax.set_xlabel(descriptor_name, fontsize=12)
    ax.set_ylabel(f'Latent Dimension {dim_idx}', fontsize=12)
    ax.set_title(f'{descriptor_name} vs Latent Dimension {dim_idx}\nPearson r = {r:.4f} (p = {p_value:.4e})',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved correlation plot to {save_path}")
    print(f"  Correlation: r = {r:.4f}, p = {p_value:.4e}")


def plot_latent_traversal(vae, base_sample, dim_idx, save_path, n_steps=10, bins_per_octave=24, num_octaves=7):
    """
    Visualize what happens when you interpolate along one dimension.
    Shows spectrograms of reconstructions.
    
    Args:
        vae: Trained VAE model
        base_sample: [1, n_bins] or [n_bins] base CQT sample
        dim_idx: Which latent dimension to traverse
        save_path: Path to save the plot
        n_steps: Number of interpolation steps
    """
    if base_sample.ndim == 1:
        base_sample = base_sample[np.newaxis, :]
    
    # Encode base sample
    encoder_output = vae.encoder.predict(base_sample, verbose=0)
    if isinstance(encoder_output, (list, tuple)):
        z_mean = encoder_output[0]
    else:
        z_mean = encoder_output
    
    # Create traversal range
    traversal_range = np.linspace(-3, 3, n_steps)
    
    # Generate reconstructions
    reconstructions = []
    for value in traversal_range:
        z_modified = z_mean.copy()
        z_modified[0, dim_idx] = value
        reconstruction = vae.decoder.predict(z_modified, verbose=0)
        reconstructions.append(reconstruction[0])

    # 1D vector: reshape to 2D for visualization
    # Assume square-ish shape for visualization
    n_bins = reconstructions[0].shape[0]
    n_rows = num_octaves
    n_cols = bins_per_octave
    
    print(f"Reshaping {n_bins} bins to ({n_rows}, {n_cols}) for visualization")
    
    # Create plot with bar charts instead of spectrograms
    fig, axes = plt.subplots(2, n_steps // 2, figsize=(20, 8))
    axes = axes.flatten()
    
    for i, (recon, value) in enumerate(zip(reconstructions, traversal_range)):
        ax = axes[i]
        # Plot as 1D signal
        ax.plot(recon, linewidth=0.5)
        ax.set_title(f'z[{dim_idx}] = {value:.2f}', fontsize=10)
        ax.set_xlabel('Frequency Bin')
        ax.set_ylabel('Magnitude')
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, max(np.max(r) for r in reconstructions) * 1.1])
    
    
    plt.suptitle(f'Latent Dimension {dim_idx} Traversal', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved latent traversal plot to {save_path}")


def plot_latent_distribution(latent_vectors, save_path):
    """
    Plot distribution of latent dimensions.
    
    Args:
        latent_vectors: [n_samples, latent_dim] array
        save_path: Path to save the plot
    """
    latent_dim = latent_vectors.shape[1]
    
    # Select subset of dimensions to plot
    n_dims_to_plot = min(16, latent_dim)
    dims_to_plot = np.linspace(0, latent_dim - 1, n_dims_to_plot, dtype=int)
    
    fig, axes = plt.subplots(4, 4, figsize=(16, 12))
    axes = axes.flatten()
    
    for i, dim_idx in enumerate(dims_to_plot):
        ax = axes[i]
        values = latent_vectors[:, dim_idx]
        
        ax.hist(values, bins=50, alpha=0.7, edgecolor='black')
        ax.axvline(values.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {values.mean():.2f}')
        ax.set_xlabel(f'Dimension {dim_idx}')
        ax.set_ylabel('Count')
        ax.set_title(f'Dim {dim_idx} (σ={values.std():.2f})')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Latent Space Dimension Distributions', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved latent distribution plot to {save_path}")