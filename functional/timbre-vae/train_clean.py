# -*- coding: utf-8 -*-
"""
Clean VAE training script with modular architecture.

This script orchestrates the training of a Variational Autoencoder (VAE) for 
audio timbre synthesis with audio feature preservation losses.
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
tf.keras.backend.clear_session()

import argparse
import time
import os

# Local imports
from config import TrainingConfig
from data_utils import load_cqt_dataset, setup_workspace, create_training_callbacks, create_optimizer
from models import create_vae_model, save_model_plots, Sampling, create_simple_vae_model
from evaluation import generate_audio_examples, generate_loss_plot, save_training_results


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train a VAE for audio timbre synthesis')
    parser.add_argument('--config', type=str, default='./default.ini', 
                       help='Path to the config file')
    parser.add_argument('--use-custom-loss', action='store_true',
                       help='Use custom training step with additional audio feature losses')
    return parser.parse_args()


def load_existing_model(config, workdir, use_custom_loss):
    """Load existing model for continued training."""
    my_model_path = workdir / 'model' / 'mymodel_last.h5'
    
    with tf.keras.utils.CustomObjectScope({'Sampling': Sampling}):
        vae = tf.keras.models.load_model(my_model_path)
    
    print("Loaded existing model for continued training")
    vae.summary()
    
    return vae


def main():
    """Main training pipeline."""
    # Parse arguments and load configuration
    args = parse_arguments()
    config = TrainingConfig(args.config)
    
    # Record start time
    start_time = time.time()
    config.config['extra']['start'] = time.asctime(time.localtime(start_time))
    
    # Setup workspace
    workdir = setup_workspace(config)
    
    # Load training data
    training_array = load_cqt_dataset(config.my_cqt)
    config.training_data = training_array  # Store for later reference
    
    # Save initial configuration
    print("Saving initial configs...")
    config.save_config(workdir / 'config.ini')
    
    # Adjust buffer size if needed
    if config.buffer_size_dataset:
        config.train_buf = len(training_array)
    
    # Create model directories
    model_dir = workdir / "model"
    log_dir = workdir / 'logs'
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # Create or load model
    if not config.continue_training:
        print("Creating new VAE model...")
        
        if args.use_custom_loss:
            print("Using custom VAE with additional audio feature losses")
            vae = create_vae_model(config)  # Your custom VAE class
        else:
            print("Using simple VAE (matching original train.py)")
            vae = create_simple_vae_model(config)  # Simple Keras model like original
        
        # Print model summaries
        if hasattr(vae, 'encoder'):
            print("\n=== Encoder Summary ===")
            vae.encoder.summary()
            print("\n=== Decoder Summary ===")
            vae.decoder.summary()
        
        print("\n=== Full VAE Summary ===")
        vae.summary()
        
        # Create model plots if requested
        if config.plot_model:
            print("Saving model architecture plots...")
            save_model_plots(vae, workdir)
        
        # Create optimizer and compile model
        optimizer = create_optimizer(config)
        
        if args.use_custom_loss:
            vae.compile(optimizer=optimizer)
        else:
            # Compile like the original train.py
            vae.compile(optimizer=optimizer, loss=tf.keras.losses.MeanSquaredError())
        
    else:
        print("Loading existing model...")
        vae = load_existing_model(config, workdir, args.use_custom_loss)
    
    # Create training callbacks
    callbacks = create_training_callbacks(config, model_dir, log_dir)
    
    # Train the model
    print(f"\n=== Starting Training ===")
    print(f"Training samples: {len(training_array)}")
    print(f"Epochs: {config.epochs}")
    print(f"Batch size: {config.batch_size}")
    print(f"Latent dimension: {config.latent_dim}")
    print(f"KL beta: {config.kl_beta}")
    
    if args.use_custom_loss:
        print(f"Attack time weight: {config.attack_time_weight}")
        print(f"Spectral centroid weight: {config.spectral_centroid_weight}")
        print("Using custom training step with audio feature losses")
        
        # Create dataset for custom training
        dataset = tf.data.Dataset.from_tensor_slices(training_array)
        dataset = dataset.batch(config.batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        history = vae.fit(
            dataset,
            epochs=config.epochs, 
            callbacks=callbacks,
            verbose=1
        )
    else:
        print("Using standard VAE training (like original train.py)")
        
        # Train exactly like the original train.py
        history = vae.fit(
            training_array, 
            training_array,  # VAE reconstructs its input
            epochs=config.epochs, 
            batch_size=config.batch_size, 
            callbacks=callbacks,
            verbose=1
        )
    
    # Record end time
    end_time = time.time()
    
    # Save training results
    save_training_results(history, config, workdir, start_time, end_time)
    
    # Generate training plots
    generate_loss_plot(history, workdir)
    
    # Generate audio examples
    generate_audio_examples(config, workdir, vae)
    
    print("\n=== Training Complete ===")
    print(f"Results saved to: {workdir}")
    print("Bye!")


if __name__ == "__main__":
    main()