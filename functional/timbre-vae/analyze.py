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
from data_utils import load_model_weights


def parse_arguments():
    parser = argparse.ArgumentParser(description='Analyze trained VAE')
    parser.add_argument('--config', type=str, default='./default.ini',
                       help='Path to analysis config file')
    parser.add_argument('--use-custom-loss', action='store_true',
                       help='Model was trained with custom loss')
    parser.add_argument('--n-samples', type=int, default=1000,
                       help='Number of samples for analysis')
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
    
    print(f"\n=== Model Architecture ===")
    print(f"Input shape: (None, {config.n_bins})")
    print(f"Latent dim: {config.latent_dim}")
    
    # Use the smart loader
    success = load_model_weights(vae, config.model_weights, prefer_separate=True)
    
    if not success:
        raise ValueError(f"Failed to load model weights from {config.model_weights}")
    
    # Verify
    test_output = vae(dummy, training=False)
    print(f"✓ Model verified, output shape: {test_output.shape}")
    
    return vae


def check_latent_space_health(vae, data_sample):
    """Check if latent space is healthy (not collapsed)."""
    print("\n=== Checking Latent Space Health ===")
    
    # Encode sample
    encoder_output = vae.encoder.predict(data_sample[:100], batch_size=64, verbose=0)
    z_mean = encoder_output[0]  # [n_samples, latent_dim]
    z_log_var = encoder_output[1]
    
    # Check statistics
    print(f"z_mean stats:")
    print(f"  Shape: {z_mean.shape}")
    print(f"  Min: {z_mean.min():.6f}")
    print(f"  Max: {z_mean.max():.6f}")
    print(f"  Mean: {z_mean.mean():.6f}")
    print(f"  Std: {z_mean.std():.6f}")
    
    print(f"\nz_log_var stats:")
    print(f"  Min: {z_log_var.min():.6f}")
    print(f"  Max: {z_log_var.max():.6f}")
    print(f"  Mean: {z_log_var.mean():.6f}")
    
    # Check for collapse
    if z_mean.std() < 0.01:
        print("⚠️  WARNING: Latent space may have collapsed (very low variance)")
        print("   This happens when KL weight is too high or training failed")
        return False
    
    # Check per-dimension variance
    dim_stds = z_mean.std(axis=0)
    inactive_dims = np.sum(dim_stds < 0.01)
    
    print(f"\nPer-dimension analysis:")
    print(f"  Active dimensions: {z_mean.shape[1] - inactive_dims}/{z_mean.shape[1]}")
    print(f"  Inactive dimensions: {inactive_dims}")
    print(f"  Mean std per dim: {dim_stds.mean():.6f}")
    print(f"  Min/Max std: [{dim_stds.min():.6f}, {dim_stds.max():.6f}]")
    
    if inactive_dims > z_mean.shape[1] * 0.5:
        print(f"⚠️  WARNING: {inactive_dims} dimensions are inactive (>50%)")
        print("   Consider reducing latent_dim or adjusting KL beta")
        return False
    
    print("✓ Latent space appears healthy")
    return True


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
    print(f"CQT shape: {cqt_data.shape}")
    print(f"Instrument families: {np.unique(labels)}")
    
    # Sample subset for analysis
    if len(cqt_data) > args.n_samples:
        print(f"\nSubsampling {args.n_samples} samples for analysis...")
        indices = np.random.choice(len(cqt_data), args.n_samples, replace=False)
        cqt_data = cqt_data[indices]
        labels = labels[indices]
    
    # Check latent space health
    is_healthy = check_latent_space_health(vae, cqt_data)
    
    if not is_healthy:
        print("\n⚠️  Latent space issues detected. Analysis may produce poor results.")
        print("Consider retraining with:")
        print("  - Lower kl_beta (e.g., 0.001 instead of 0.01)")
        print("  - KL annealing (gradually increase beta)")
        print("  - Check reconstruction loss is decreasing")
        response = input("\nContinue anyway? (y/n): ")
        if response.lower() != 'y':
            return
    
    # === 1. Correlation Analysis ===
    if args.use_custom_loss and config.spectral_centroid_weight > 0:
        print("\n=== Computing Descriptor Correlations ===")
        
        try:
            results = compute_descriptor_correlation(
                vae, cqt_data, compute_spectral_centroid_tf, config.centroid_dim
            )
            
            if np.isnan(results['pearson_r']):
                print("⚠️  Correlation is NaN - latent dimension may be constant")
            else:
                print(f"Spectral Centroid ↔ Dim {config.centroid_dim}:")
                print(f"  Pearson r = {results['pearson_r']:.4f} (p = {results['pearson_p']:.4e})")
            
            plot_descriptor_correlation(
                *results['scatter_data'],
                "Spectral Centroid",
                config.centroid_dim,
                output_dir / "centroid_correlation.png"
            )
        except Exception as e:
            print(f"⚠️  Correlation analysis failed: {e}")
    
    descriptors = {
        'spectral_centroid': compute_spectral_centroid_tf
    }
    
    # === 2. Disentanglement Metrics ===
    print("\n=== Computing Disentanglement Scores ===")
    
    try:
        scores = compute_disentanglement_score(vae, cqt_data, descriptors)
        
        # Save scores
        np.save(output_dir / 'disentanglement_scores.npy', scores)
        print(f"✓ Saved disentanglement scores to: {output_dir / 'disentanglement_scores.npy'}")
        
    except Exception as e:
        print(f"⚠️  Disentanglement analysis failed: {e}")
    
    # === 3. t-SNE Visualization ===
    print("\n=== Generating t-SNE Visualization ===")
    
    try:
        # Sample balanced data for t-SNE (at least 10 per class)
        n_per_class = max(10, min(100, len(cqt_data) // (len(np.unique(labels)) * 2)))
        
        tsne_data, tsne_labels = stratified_sample(
            cqt_data, labels,
            n_per_class=n_per_class
        )
        
        print(f"t-SNE sample size: {len(tsne_data)} ({n_per_class} per class)")
        
        if len(tsne_data) < 30:
            print("⚠️  Warning: t-SNE sample size is small, results may be unreliable")
        
        # Encode to latent space
        encoder_output = vae.encoder.predict(tsne_data, batch_size=64, verbose=0)
        z_mean = encoder_output[0]  # [n_samples, latent_dim]
        
        plot_tsne(
            z_mean,
            tsne_labels,
            np.unique(tsne_labels),
            output_dir / "tsne_by_instrument.png"
        )
        
    except Exception as e:
        print(f"⚠️  t-SNE visualization failed: {e}")
    
    # === 4. Latent Traversal ===
    print("\n=== Generating Latent Traversals ===")
    
    try:
        base_sample = cqt_data[0:1]  # Use first sample
        
        # Try centroid dimension if disentanglement is enabled
        dims_to_traverse = [0, 1, 2]  # First few dimensions
        if args.use_custom_loss and hasattr(config, 'centroid_dim'):
            dims_to_traverse = [config.centroid_dim, config.centroid_dim + 1, config.centroid_dim + 2]
        
        for dim in dims_to_traverse:
            if dim < config.latent_dim:
                plot_latent_traversal(
                    vae, base_sample, dim,
                    save_path=output_dir / f"traversal_dim{dim}.png"
                )
        
    except Exception as e:
        print(f"⚠️  Latent traversal failed: {e}")
    
    print(f"\n=== Analysis Complete ===")
    print(f"Results saved to: {output_dir}")
    
    # Print summary
    if is_healthy:
        print("\n✓ Model appears to be working correctly")
    else:
        print("\n⚠️  Model has issues - consider retraining")


if __name__ == "__main__":
    main()