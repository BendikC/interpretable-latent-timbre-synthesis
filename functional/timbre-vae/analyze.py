"""
Analyze trained VAE models for interpretability.
"""

import argparse
import numpy as np
from pathlib import Path

from config import TrainingConfig
from models import create_vae_model, create_simple_vae_model
from analysis.dataset_utils import load_nsynth_with_labels, stratified_sample
from analysis.interpretability import (
    compute_descriptor_correlation,
    compute_disentanglement_score,
    independence_test
)
from analysis.visualization import (
    plot_tsne,
    plot_descriptor_correlation,
    plot_latent_traversal
)
from audio_features import compute_spectral_centroid_tf


def parse_arguments():
    parser = argparse.ArgumentParser(description='Analyze trained VAE')
    parser.add_argument('--config', type=str, default='./default.ini',
                       help='Path to analysis config file')
    parser.add_argument('--use-custom-loss', action='store_true',
                       help='Model was trained with custom loss')
    return parser.parse_args()


def load_trained_model(config, use_custom_loss):
    """Load trained VAE from weights."""
    import tensorflow as tf
    
    if use_custom_loss:
        vae = create_vae_model(config)
    else:
        vae = create_simple_vae_model(config)
    
    # Build model
    dummy = tf.zeros((1, config.n_bins))
    _ = vae(dummy)
    
    # Load weights
    vae.load_weights(config.model_weights)
    print(f"✓ Loaded model from {config.model_weights}")
    
    return vae


def main():
    args = parse_arguments()
    config = TrainingConfig(args.config)
    
    # Create output directory
    output_dir = Path(config.output_dir_analysis)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=== Loading Trained Model ===")
    vae = load_trained_model(config, args.use_custom_loss)
    
    print("\n=== Loading Dataset with Labels ===")
    cqt_data, labels, metadata = load_nsynth_with_labels(
        config.my_cqt
    )
    print(f"Loaded {len(cqt_data)} samples")
    print(f"Instrument families: {np.unique(labels)}")
    
    # === 1. Correlation Analysis ===
    print("\n=== Computing Descriptor Correlations ===")
    
    results = compute_descriptor_correlation(
        vae, cqt_data, compute_spectral_centroid_tf, config.centroid_dim
    )
    
    print(f"Spectral Centroid ↔ Dim {config.centroid_dim}:")
    print(f"  Pearson r = {results['pearson_r']:.4f} (p = {results['pearson_p']:.4e})")
    
    plot_descriptor_correlation(
        *results['scatter_data'],
        "Spectral Centroid",
        config.centroid_dim,
        output_dir / "centroid_correlation.png"
    )
    
    descriptors = {
        'spectral_centroid': compute_spectral_centroid_tf
    }
    
    # === 3. Disentanglement Metrics ===
    print("\n=== Computing Disentanglement Scores ===")
    
    scores = compute_disentanglement_score(vae, cqt_data, descriptors)
    print(f"Disentanglement scores: {scores}")
    
    # === 4. t-SNE Visualization ===
    print("\n=== Generating t-SNE Visualization ===")
    
    # Sample balanced data for t-SNE
    tsne_data, tsne_labels = stratified_sample(
        cqt_data, labels,
        n_per_class=1
    )
    
    # Encode to latent space
    encoder_output = vae.encoder.predict(tsne_data, batch_size=64)
    z_mean = encoder_output[0]  # [n_samples, latent_dim]
    
    plot_tsne(
        z_mean,
        tsne_labels,
        np.unique(tsne_labels),
        output_dir / "tsne_by_instrument.png"
    )
    
    # === 5. Latent Traversal ===
    print("\n=== Generating Latent Traversals ===")
    
    base_sample = cqt_data[0:1]  # Use first sample
    
    for dim in [config.centroid_dim, config.centroid_dim + 1]:
        plot_latent_traversal(
            vae, base_sample, dim,
            save_path=output_dir / f"traversal_dim{dim}.png"
        )
    
    print(f"\n=== Analysis Complete ===")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()