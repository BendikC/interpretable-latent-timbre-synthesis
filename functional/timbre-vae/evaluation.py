# -*- coding: utf-8 -*-
"""Example generation and evaluation utilities."""

import numpy as np
import random
import os
import librosa
import tensorflow as tf
import soundfile as sf


def generate_audio_examples(config, workdir, vae, no_of_examples=30):
    """Generate audio examples from trained model.
    
    Args:
        config: TrainingConfig object
        workdir: Working directory path
        vae: Trained VAE model
    """
    print("Generating examples...")
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    my_examples_folder = workdir / 'audio_examples'
    my_examples_folder.mkdir(exist_ok=True)  # Use mkdir instead of os.makedirs
    
    # Check if audio directory exists
    if not hasattr(config, 'my_audio') or not config.my_audio.exists():
        print("Warning: Audio directory not found, skipping example generation")
        return
    
    for f in os.listdir(config.my_audio):
        if not f.endswith('.wav'):  # Skip non-audio files
            continue

        if no_of_examples <= 0:
            break
        no_of_examples -= 1
            
        print(f"Examples for {os.path.splitext(f)[0]}") 
        file_path = config.my_audio / f 
        
        try:
            my_file_duration = librosa.get_duration(filename=file_path)
            
            # Fix: Handle case where file is shorter than example_length
            if my_file_duration <= config.example_length:
                print(f"Warning: {f} duration ({my_file_duration:.2f}s) is shorter than example_length ({config.example_length}s)")
                my_offset = 0
                actual_duration = my_file_duration
            else:
                max_offset = int(my_file_duration) - config.example_length
                my_offset = random.randint(0, max_offset)
                actual_duration = config.example_length
            
            s, fs = librosa.load(file_path, duration=actual_duration, offset=my_offset, sr=None)
            
            # Get the CQT magnitude
            print("Calculating CQT")
            C_complex = librosa.cqt(y=s, sr=fs, hop_length=config.hop_length, 
                                   bins_per_octave=config.bins_per_octave, n_bins=config.n_bins)
            C = np.abs(C_complex)
            
            C_32 = C.astype('float32')
            y_inv_32 = librosa.griffinlim_cqt(C, sr=fs, n_iter=config.n_iter, 
                                             hop_length=config.hop_length, 
                                             bins_per_octave=config.bins_per_octave, 
                                             dtype=np.float32)
            
            # Generate the same CQT using the model
            my_array = np.transpose(C_32)
            test_dataset = tf.data.Dataset.from_tensor_slices(my_array).batch(config.batch_size).prefetch(AUTOTUNE)
            output = tf.constant(0., dtype='float32', shape=(1, config.n_bins))
            
            print("Working on regenerating cqt magnitudes with the DL model")
            for step, x_batch_train in enumerate(test_dataset):
                reconstructed = vae(x_batch_train)
                output = tf.concat([output, reconstructed], 0)

            output_np = np.transpose(output.numpy())
            output_inv_32 = librosa.griffinlim_cqt(output_np[1:], 
                                                  sr=fs, n_iter=config.n_iter, 
                                                  hop_length=config.hop_length, 
                                                  bins_per_octave=config.bins_per_octave, 
                                                  dtype=np.float32)
            
            if config.normalize_examples:
                output_inv_32 = librosa.util.normalize(output_inv_32)
            
            # Save audio files
            print("Saving audio files...")
            my_audio_out_fold = my_examples_folder / os.path.splitext(f)[0]
            my_audio_out_fold.mkdir(exist_ok=True)  # Use pathlib mkdir
            
            # Fix: Use soundfile instead of deprecated librosa.output.write_wav
            sf.write(my_audio_out_fold / 'original.wav', s, config.sample_rate)
            sf.write(my_audio_out_fold / 'original-icqt+gL.wav', y_inv_32, config.sample_rate)
            sf.write(my_audio_out_fold / 'VAE-output+gL.wav', output_inv_32, config.sample_rate)
            
        except Exception as e:
            print(f"Error processing {f}: {e}")
            continue


def generate_loss_plot(history, workdir):
    """Generate and save training loss plot.
    
    Args:
        history: Training history object
        workdir: Working directory path
    """
    import matplotlib.pyplot as plt
    
    print("Generating loss plot...")
    history_dict = history.history
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Training Metrics')
    
    # Plot all available metrics
    metrics = ['loss', 'reconstruction_loss', 'kl_loss', 'attack_loss', 'centroid_loss']
    
    for i, metric in enumerate(metrics):
        row = i // 3
        col = i % 3
        
        if metric in history_dict:
            axes[row, col].plot(history_dict[metric])
            axes[row, col].set_title(metric.replace('_', ' ').title())
            axes[row, col].set_xlabel('Epochs')
            axes[row, col].set_ylabel('Loss')
        else:
            axes[row, col].text(0.5, 0.5, f'{metric}\nNot Available', 
                              ha='center', va='center', transform=axes[row, col].transAxes)
    
    # Remove empty subplot
    if len(metrics) < 6:
        fig.delaxes(axes[1, 2])
    
    plt.tight_layout()
    fig.savefig(workdir / 'training_metrics_plot.pdf', dpi=300)
    plt.close()
    
    # Also create simple loss plot for compatibility
    fig = plt.figure()
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    if 'loss' in history_dict:
        plt.plot(history_dict['loss'])
    plt.savefig(workdir / 'my_history_plot.pdf', dpi=300)
    plt.close()


def save_training_results(history, config, workdir, start_time, end_time):
    """Save training results and updated config.
    
    Args:
        history: Training history object
        config: TrainingConfig object
        workdir: Working directory path
        start_time: Training start time
        end_time: Training end time
    """
    import json
    import time
    
    # Save history
    print('\nhistory dict:', history.history)
    with open(workdir / 'my_history.json', 'w') as json_file:
        json.dump(history.history, json_file)

    # Update config with training results
    config.config['extra']['end'] = time.asctime(time.localtime(end_time))
    time_elapsed = end_time - start_time
    config.config['extra']['time_elapsed'] = str(time_elapsed)
    config.config['extra']['total_epochs'] = str(len(history.history['loss']))
    config.config['dataset']['total_frames'] = str(len(config.training_data))
    
    # Save updated config
    config.save_config(workdir / 'config.ini')
    
    print("Finished training...")
    print(f"Training time: {time_elapsed:.2f} seconds")
    print(f"Total epochs: {len(history.history['loss'])}")