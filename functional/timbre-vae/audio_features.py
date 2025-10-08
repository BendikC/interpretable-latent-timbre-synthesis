# -*- coding: utf-8 -*-
"""Audio feature extraction utilities for VAE training."""

import numpy as np
import tensorflow as tf

def compute_spectral_centroid_tf(cqt_magnitude):
    """Compute spectral centroid using pure TensorFlow operations."""
    # cqt_magnitude shape: [batch_size, n_bins]
    
    # Create frequency bins (approximate, you may need to adjust based on your CQT setup)
    n_bins = tf.shape(cqt_magnitude)[1]
    freq_bins = tf.cast(tf.range(n_bins), tf.float32)
    freq_bins = tf.expand_dims(freq_bins, 0)  # [1, n_bins]
    
    # Weighted sum of frequencies
    weighted_freq = tf.reduce_sum(cqt_magnitude * freq_bins, axis=1)  # [batch_size]
    total_magnitude = tf.reduce_sum(cqt_magnitude, axis=1)  # [batch_size]
    
    # Avoid division by zero
    centroid = weighted_freq / (total_magnitude + 1e-8)
    return centroid

def compute_attack_time_tf(cqt_magnitude):
    """Compute attack time using pure TensorFlow operations."""
    # Simple approximation: find the frame with maximum energy
    # This is a simplified version - you might want a more sophisticated approach
    
    # Sum across frequency bins to get energy per frame
    energy_per_frame = tf.reduce_sum(cqt_magnitude, axis=1)  # [batch_size]
    
    # For simplicity, use the position of maximum energy as attack time proxy
    max_positions = tf.cast(tf.argmax(energy_per_frame, axis=0), tf.float32)
    
    # If you have multiple samples in batch, you might want to process differently
    # This is a simplified version
    batch_size = tf.shape(cqt_magnitude)[0]
    attack_times = tf.fill([batch_size], max_positions)
    
    return attack_times


def compute_attack_time(cqt_magnitude):
    """Compute attack time from CQT magnitude spectrum.
    
    Args:
        cqt_magnitude: CQT magnitude array with shape (n_bins, n_frames)
        
    Returns:
        float: Attack time normalized by total frames
    """
    # Sum across frequency bins to get energy over time
    energy = np.sum(cqt_magnitude, axis=0)
    # Find peak energy frame
    peak_frame = np.argmax(energy)
    # Attack time is the frame of peak energy (normalized by total frames)
    return peak_frame / len(energy)


def compute_spectral_centroid(cqt_magnitude, bins_per_octave=48, num_octaves=8):
    """Compute spectral centroid from CQT magnitude spectrum.
    
    Args:
        cqt_magnitude: CQT magnitude array with shape (n_bins, n_frames)
        bins_per_octave: Number of frequency bins per octave
        num_octaves: Number of octaves covered
        
    Returns:
        float: Mean spectral centroid normalized by total bins
    """
    # Create frequency array (in bins)
    freq_bins = np.arange(cqt_magnitude.shape[0])
    
    # Compute centroid for each time frame
    centroids = []
    for t in range(cqt_magnitude.shape[1]):
        magnitude_frame = cqt_magnitude[:, t]
        if np.sum(magnitude_frame) > 0:
            centroid = np.sum(freq_bins * magnitude_frame) / np.sum(magnitude_frame)
        else:
            centroid = len(freq_bins) / 2  # Default to middle
        centroids.append(centroid)
    
    # Return mean centroid normalized by total bins
    return np.mean(centroids) / len(freq_bins)


def compute_features_from_cqt_batch(cqt_batch):
    """Compute audio features for a batch of CQT frames.
    
    Args:
        cqt_batch: Tensor of shape (batch_size, n_bins)
        
    Returns:
        tuple: (attack_times, centroids) as TensorFlow constants
    """
    batch_size = cqt_batch.shape[0]
    n_bins = cqt_batch.shape[1]
    
    attack_times = []
    centroids = []
    
    for i in range(batch_size):
        # Reshape single frame to simulate time dimension
        cqt_frame = cqt_batch[i].numpy().reshape(-1, 1)
        
        # For single frames, attack time doesn't make much sense,
        # so we'll use the energy distribution as a proxy
        energy_weighted_bin = np.sum(np.arange(n_bins) * cqt_frame.flatten()) / (np.sum(cqt_frame) + 1e-8)
        attack_time_proxy = energy_weighted_bin / n_bins
        
        # Spectral centroid
        centroid = compute_spectral_centroid(cqt_frame)
        
        attack_times.append(attack_time_proxy)
        centroids.append(centroid)
    
    return tf.constant(attack_times, dtype=tf.float32), tf.constant(centroids, dtype=tf.float32)


def compute_audio_features_full_sequence(audio, sample_rate, hop_length, bins_per_octave, n_bins):
    """Compute audio features from full audio sequence.
    
    Args:
        audio: Audio time series
        sample_rate: Sample rate
        hop_length: Hop length for CQT
        bins_per_octave: Bins per octave
        n_bins: Total number of bins
        
    Returns:
        dict: Dictionary containing computed features
    """
    import librosa
    
    # Compute CQT
    C_complex = librosa.cqt(y=audio, sr=sample_rate, hop_length=hop_length, 
                           bins_per_octave=bins_per_octave, n_bins=n_bins)
    C = np.abs(C_complex)
    
    # Compute features
    attack_time = compute_attack_time(C)
    spectral_centroid = compute_spectral_centroid(C, bins_per_octave, n_bins // bins_per_octave)
    
    # Additional features can be added here
    features = {
        'attack_time': attack_time,
        'spectral_centroid': spectral_centroid,
        'rms_energy': np.mean(librosa.feature.rms(y=audio)),
        'zero_crossing_rate': np.mean(librosa.feature.zero_crossing_rate(audio))
    }
    
    return features