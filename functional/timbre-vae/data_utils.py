# -*- coding: utf-8 -*-
"""Data loading and preprocessing utilities."""

import numpy as np
import os
from pathlib import Path


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
                my_runs = config.dataset_path / config.description
                run_name = f'run-{run_id:03d}'
                workdir = my_runs / run_name 
                os.makedirs(workdir)
                break
            except OSError:
                if workdir.is_dir():
                    run_id = run_id + 1
                    continue
                raise
        
        config.update_workspace(workdir)
    else:
        workdir = Path(config.workspace)

    print(f"Workspace: {workdir}")
    return workdir


def create_training_callbacks(config, model_dir, log_dir):
    """Create training callbacks.
    
    Args:
        config: TrainingConfig object
        model_dir: Model save directory
        log_dir: Tensorboard log directory
        
    Returns:
        list: List of Keras callbacks
    """
    import tensorflow as tf
    
    modelpath = model_dir / 'mymodel_last.h5'
    
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(modelpath),
            save_best_only=config.save_best_only,
            monitor='loss',
            verbose=1,
            save_weights_only=True ## TODO: look into if I should change this back to False
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
        beta_2=config.adam_beta_2
    )
    
    return optimizer