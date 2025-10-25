# -*- coding: utf-8 -*-
"""Audio feature extraction utilities for VAE training."""

import numpy as np
import tensorflow as tf

def compute_spectral_centroid_tf(cqt_magnitude, fmin=32.7, bins_per_octave=48):
    """Compute spectral centroid using proper frequency mapping."""
    # cqt_magnitude shape: [batch_size, n_bins]
    
    n_bins = tf.shape(cqt_magnitude)[1]
    bin_indices = tf.cast(tf.range(n_bins), tf.float32)
    
    # Convert bin indices to actual frequencies (Hz)
    frequencies = fmin * tf.pow(2.0, bin_indices / bins_per_octave)
    frequencies = tf.expand_dims(frequencies, 0)  # [1, n_bins]
    
    # Compute centroid using real frequencies
    weighted_freq = tf.reduce_sum(cqt_magnitude * frequencies, axis=1)
    total_magnitude = tf.reduce_sum(cqt_magnitude, axis=1)
    centroid_hz = weighted_freq / (total_magnitude + 1e-8)
    
    # Normalize to 0-1 range for training stability
    fmax = fmin * tf.pow(2.0, tf.cast(n_bins, tf.float32) / bins_per_octave)
    centroid_normalized = tf.math.log(centroid_hz / fmin) / tf.math.log(fmax / fmin)
    
    return centroid_normalized

def compute_attack_time_tf(cqt_magnitude):
    """Compute attack sharpness proxy from single CQT frames."""
    # For single frames, we can't measure true attack time
    # Instead, measure "spectral sharpness" - concentrated energy = sharper attack
    
    # Compute spectral spread (inverse of sharpness)
    centroid = compute_spectral_centroid_tf(cqt_magnitude)
    
    n_bins = tf.shape(cqt_magnitude)[1]
    bin_indices = tf.cast(tf.range(n_bins), tf.float32) / tf.cast(n_bins, tf.float32)
    bin_indices = tf.expand_dims(bin_indices, 0)  # [1, n_bins]
    
    # Compute spectral spread around centroid
    centroid_expanded = tf.expand_dims(centroid, 1)  # [batch_size, 1]
    spread = tf.reduce_sum(cqt_magnitude * tf.square(bin_indices - centroid_expanded), axis=1)
    total_magnitude = tf.reduce_sum(cqt_magnitude, axis=1)
    spectral_spread = spread / (total_magnitude + 1e-8)
    
    # Invert spread to get sharpness (lower spread = sharper attack)
    attack_sharpness = 1.0 / (1.0 + spectral_spread * 10.0)  # Scale factor for range
    
    # DEBUG PRINTS
    tf.print("Attack sharpness - min:", tf.reduce_min(attack_sharpness), 
             "max:", tf.reduce_max(attack_sharpness), "mean:", tf.reduce_mean(attack_sharpness))
    
    return attack_sharpness


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