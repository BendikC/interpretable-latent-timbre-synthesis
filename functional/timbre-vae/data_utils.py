# -*- coding: utf-8 -*-
"""Data loading and preprocessing utilities."""

import numpy as np
import os
from pathlib import Path
from models import AudioFeatureVAE, KLAnnealingCallback

def check_gpu_availability():
    """Check if GPU is available and print device information.
    
    Returns:
        bool: True if GPU is available, False otherwise
    """
    import tensorflow as tf
    
    print("\n=== GPU Availability Check ===")
    
    # List physical devices
    gpus = tf.config.list_physical_devices('GPU')
    cpus = tf.config.list_physical_devices('CPU')
    
    print(f"CPUs Available: {len(cpus)}")
    print(f"GPUs Available: {len(gpus)}")

    if gpus:
        print("\nGPU Details:")
        for i, gpu in enumerate(gpus):
            print(f"  GPU {i}: {gpu.name}")
            # Try to get memory info if available
            try:
                gpu_details = tf.config.experimental.get_device_details(gpu)
                if gpu_details:
                    print(f"    Details: {gpu_details}")
            except:
                pass
        
        # Check if TensorFlow is actually built with CUDA
        print(f"\nTensorFlow built with CUDA: {tf.test.is_built_with_cuda()}")
        print(f"GPU available for TensorFlow: {tf.test.is_gpu_available()}")
        
        print("\n✓ GPU will be used for training")
        return True
    else:
        print("\n⚠ No GPU detected. Training will use CPU (slower)")
        return False


def load_cqt_dataset(cqt_path):
    """Load all CQT files from directory and concatenate.
    
    Args:
        cqt_path: Path to directory containing .npy CQT files
        
    Returns:
        np.ndarray: Concatenated CQT training data
    """
    print('Creating the dataset...')
    training_array = []
    new_loop = True

    for f in os.listdir(cqt_path): 
        if f.endswith('.npy'):
            print(f'Adding -> {f}')
            file_path = cqt_path / f
            new_array = np.load(file_path)
            if new_loop:
                training_array = new_array
                new_loop = False
            else:
                training_array = np.concatenate((training_array, new_array), axis=0)

    total_cqt = len(training_array)
    print(f'Total number of CQT frames: {total_cqt}')
    
    return training_array


def setup_workspace(config):
    """Set up workspace directory for training run.
    
    Args:
        config: TrainingConfig object
        
    Returns:
        Path: Working directory path
    """
    if not config.continue_training:
        run_id = config.run_number
        while True:
            try:
                # Use output_dir + description instead of dataset_path
                runs_dir = config.output_dir / config.description
                run_name = f'run-{run_id:03d}'
                workdir = runs_dir / run_name 
                os.makedirs(workdir)
                break
            except OSError:
                if workdir.is_dir():
                    run_id = run_id + 1
                    continue
                raise
        
        config.update_workspace(workdir)
    else:
        # For continued training, workspace must be explicitly set
        if config.workspace is None:
            raise ValueError("workspace must be set for continue_training=True")
        workdir = Path(config.workspace)

    print(f"Workspace: {workdir}")
    return workdir


def save_model_weights(vae, model_dir, save_separate=True):
    """Save VAE model weights.
    
    Args:
        vae: Trained VAE model
        model_dir: Directory to save weights
        save_separate: If True, save encoder/decoder separately for easier loading
    """
    import tensorflow as tf
    
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Save full model weights (for backward compatibility)
    full_model_path = model_dir / 'mymodel_last.weights.h5'
    print(f"Saving full model weights to: {full_model_path}")
    vae.save_weights(str(full_model_path))
    
    if save_separate:
        # Save encoder and decoder separately (cleaner loading)
        encoder_path = model_dir / 'encoder.weights.h5'
        decoder_path = model_dir / 'decoder.weights.h5'
        
        print(f"Saving encoder weights to: {encoder_path}")
        vae.encoder.save_weights(str(encoder_path))
        
        print(f"Saving decoder weights to: {decoder_path}")
        vae.decoder.save_weights(str(decoder_path))
        
        print("✓ Saved separate encoder/decoder weights for easier loading")


def load_model_weights(vae, weights_path, prefer_separate=True):
    """Load VAE model weights with automatic fallback.
    
    Args:
        vae: VAE model instance to load weights into
        weights_path: Path to weights file or model directory
        prefer_separate: If True, try to load separate encoder/decoder files first
        
    Returns:
        bool: True if loading succeeded
    """
    import tensorflow as tf
    
    weights_path = Path(weights_path)
    
    # If it's a directory, look for weights files
    if weights_path.is_dir():
        model_dir = weights_path
        full_weights = model_dir / 'mymodel_last.weights.h5'
        encoder_weights = model_dir / 'encoder.weights.h5'
        decoder_weights = model_dir / 'decoder.weights.h5'
    else:
        # It's a file path
        model_dir = weights_path.parent
        full_weights = weights_path
        encoder_weights = model_dir / 'encoder.weights.h5'
        decoder_weights = model_dir / 'decoder.weights.h5'
    
    # Strategy 1: Try separate encoder/decoder files (cleanest)
    if prefer_separate and encoder_weights.exists() and decoder_weights.exists():
        try:
            print(f"Loading encoder from: {encoder_weights}")
            vae.encoder.load_weights(str(encoder_weights))
            
            print(f"Loading decoder from: {decoder_weights}")
            vae.decoder.load_weights(str(decoder_weights))
            
            print("✓ Loaded separate encoder/decoder weights")
            return True
        except Exception as e:
            print(f"⚠️  Failed to load separate weights: {e}")
    
    # Strategy 2: Try full model weights (standard)
    if full_weights.exists():
        try:
            print(f"Loading full model weights from: {full_weights}")
            vae.load_weights(str(full_weights))
            print("✓ Loaded full model weights")
            return True
        except Exception as e:
            print(f"⚠️  Failed to load full model weights: {e}")
    
    print(f"❌ Could not load weights from: {weights_path}")
    return False


def create_training_callbacks(config, model_dir, log_dir, vae=None):
    """Create training callbacks.
    
    Args:
        config: TrainingConfig object
        model_dir: Model save directory
        log_dir: Tensorboard log directory
        vae: VAE model (optional, for custom save callback)
        
    Returns:
        list: List of Keras callbacks
    """
    import tensorflow as tf
    
    model_dir = Path(model_dir)
    modelpath = model_dir / 'mymodel_last.weights.h5'
    
    # Custom callback to save encoder/decoder separately
    class SaveSeparateWeights(tf.keras.callbacks.Callback):
        def __init__(self, vae, model_dir):
            super().__init__()
            self.vae = vae
            self.model_dir = Path(model_dir)
        
        def on_epoch_end(self, epoch, logs=None):
            """Save separate weights at end of each epoch."""
            save_model_weights(self.vae, self.model_dir, save_separate=True)
    
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(modelpath),
            save_best_only=config.save_best_only,
            save_weights_only=True,
            monitor='loss',
            verbose=1
        ),         
        tf.keras.callbacks.EarlyStopping(
            monitor='loss',
            min_delta=config.early_delta,
            patience=config.early_patience_epoch,
            verbose=1
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir=str(log_dir), 
            histogram_freq=1
        )
    ]
    
    # Add custom callback for separate weight saving
    if vae is not None:
        callbacks.append(SaveSeparateWeights(vae, model_dir))

    if vae is not None and isinstance(vae, AudioFeatureVAE):
        kl_callback = KLAnnealingCallback(vae)
        callbacks.append(kl_callback)
        print("✓ Added KL annealing callback")
    
    return callbacks


def create_optimizer(config):
    """Create optimizer with optional learning rate schedule.
    
    Args:
        config: TrainingConfig object
        
    Returns:
        tf.keras.optimizers.Optimizer: Configured optimizer
    """
    import tensorflow as tf
    
    learning_rate = config.learning_rate
    
    if config.learning_schedule:
        learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(
            learning_rate * 100,
            decay_steps=int(config.epochs * 0.8),
            decay_rate=0.96,
            staircase=True
        )

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=learning_rate, 
        beta_1=config.adam_beta_1, 
        beta_2=config.adam_beta_2,
        clipnorm=1.0
    )
    
    return optimizer