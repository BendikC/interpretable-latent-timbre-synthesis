# -*- coding: utf-8 -*-
"""VAE model architecture definitions."""

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import backend as K
from audio_features import compute_features_from_cqt_batch, compute_attack_time_tf, compute_spectral_centroid_tf


class Sampling(layers.Layer):
    """Sampling layer for VAE reparameterization trick."""
    
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


def build_encoder(input_dim, latent_dim, n_units):
    """Build VAE encoder.
    
    Args:
        input_dim: Input dimension (n_bins)
        latent_dim: Latent space dimension
        n_units: Number of hidden units
        
    Returns:
        tf.keras.Model: Encoder model
    """
    inputs = tf.keras.Input(shape=(input_dim,), name='encoder_input')
    x = layers.Dense(n_units, activation='relu')(inputs)
    z_mean = layers.Dense(latent_dim, name='z_mean')(x)
    z_log_var = layers.Dense(latent_dim, name='z_log_var')(x)
    
    # For encoder output, we return both mean and log_var
    encoder = tf.keras.Model(inputs=inputs, outputs=[z_mean, z_log_var], name='encoder')
    return encoder


def build_decoder(latent_dim, output_dim, n_units, output_activation='sigmoid'):
    """Build VAE decoder.
    
    Args:
        latent_dim: Latent space dimension
        output_dim: Output dimension (n_bins)
        n_units: Number of hidden units
        output_activation: Output layer activation function
        
    Returns:
        tf.keras.Model: Decoder model
    """
    latent_inputs = tf.keras.Input(shape=(latent_dim,), name='z_sampling')
    x = layers.Dense(n_units, activation='relu')(latent_inputs)
    outputs = layers.Dense(output_dim, activation=output_activation)(x)
    
    decoder = tf.keras.Model(inputs=latent_inputs, outputs=outputs, name='decoder')
    return decoder


class AudioFeatureVAE(tf.keras.Model):
    """VAE with audio feature preservation losses."""
    
    def __init__(self, encoder, decoder, kl_beta=1.0, 
                 attack_time_weight=0.1, spectral_centroid_weight=0.1,
                 centroid_dim=0, disentangle_weight=1.0, **kwargs):
        super(AudioFeatureVAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.kl_beta = kl_beta
        self.attack_time_weight = attack_time_weight
        self.spectral_centroid_weight = spectral_centroid_weight
        self.sampling_layer = Sampling()

        self.centroid_dim = centroid_dim  # Which latent dimension controls centroid
        self.disentangle_weight = disentangle_weight
        
        # Metrics trackers
        self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = tf.keras.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = tf.keras.metrics.Mean(name="kl_loss")
        self.attack_loss_tracker = tf.keras.metrics.Mean(name="attack_loss")
        self.centroid_loss_tracker = tf.keras.metrics.Mean(name="centroid_loss")
        self.disentangle_loss_tracker = tf.keras.metrics.Mean(name="disentangle_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
            self.attack_loss_tracker,
            self.centroid_loss_tracker,
            self.disentangle_loss_tracker,
        ]

    def call(self, inputs, training=None):
        """Forward pass through the VAE."""
        z_mean, z_log_var = self.encoder(inputs)
        z = self.sampling_layer([z_mean, z_log_var])
        reconstruction = self.decoder(z)
        return reconstruction

    def encode(self, inputs):
        """Encode inputs to latent space."""
        return self.encoder(inputs)

    def decode(self, z):
        """Decode latent vectors to reconstructions."""
        return self.decoder(z)

    def train_step(self, data):
        """Custom training step with audio feature losses."""
        with tf.GradientTape() as tape:
            # Forward pass
            z_mean, z_log_var = self.encoder(data)
            z = self.sampling_layer([z_mean, z_log_var])
            reconstruction = self.decoder(z)
            
            # Reconstruction loss
            reconstruction_loss = tf.reduce_mean(
                tf.keras.losses.mse(data, reconstruction)
            )
            
            # KL divergence loss - CORRECT FORMULA
            # KL = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
            kl_loss = -0.5 * tf.reduce_mean(
                tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1)
            )
            
            # Clip to prevent explosion
            kl_loss = tf.clip_by_value(kl_loss, 0.0, 1000.0)

            # Audio feature losses using pure TensorFlow (DIFFERENTIABLE!)
            input_centroid = compute_spectral_centroid_tf(data)
            output_centroid = compute_spectral_centroid_tf(reconstruction)
            centroid_loss = tf.reduce_mean(tf.square(input_centroid - output_centroid))
            
            # IMPROVED: Disentanglement loss with better scaling
            z_centroid_dim = z[:, self.centroid_dim:self.centroid_dim+1]
            
            # Use sigmoid instead of tanh for better stability
            # Sigmoid naturally maps to [0, 1]
            predicted_centroid = tf.nn.sigmoid(z_centroid_dim)
            predicted_centroid = tf.squeeze(predicted_centroid, axis=1)
            
            # Loss with stability epsilon
            disentangle_loss = tf.reduce_mean(
                tf.square(predicted_centroid - input_centroid) + 1e-8
            )

            tf.debugging.check_numerics(reconstruction_loss, "reconstruction_loss")
            tf.debugging.check_numerics(kl_loss, "kl_loss")
            tf.debugging.check_numerics(centroid_loss, "centroid_loss")
            tf.debugging.check_numerics(disentangle_loss, "disentangle_loss")
            
            # Total loss
            total_loss = (reconstruction_loss + 
                         self.kl_beta * kl_loss + 
                         self.spectral_centroid_weight * centroid_loss + 
                         self.disentangle_weight * disentangle_loss)
        
        # Compute gradients and update weights (now this will work!)
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        
        # Update metrics
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        self.centroid_loss_tracker.update_state(centroid_loss)
        self.disentangle_loss_tracker.update_state(disentangle_loss)
        
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
            "centroid_loss": self.centroid_loss_tracker.result(),
            "disentangle_loss": self.disentangle_loss_tracker.result(),
        }

def create_vae_model(config):
    """Factory function to create VAE model from config.
    
    Args:
        config: TrainingConfig object
        
    Returns:
        AudioFeatureVAE: Configured VAE model
    """
    # Build encoder and decoder
    encoder = build_encoder(
        input_dim=config.n_bins,
        latent_dim=config.latent_dim,
        n_units=config.n_units
    )
    
    decoder = build_decoder(
        latent_dim=config.latent_dim,
        output_dim=config.n_bins,
        n_units=config.n_units,
        output_activation=config.VAE_output_activation
    )
    
    # Create VAE
    vae = AudioFeatureVAE(
        encoder=encoder,
        decoder=decoder,
        kl_beta=config.kl_beta,
        attack_time_weight=config.attack_time_weight,
        spectral_centroid_weight=config.spectral_centroid_weight,
        disentangle_weight=config.disentangle_weight,
        centroid_dim=config.centroid_dim
    )

    # Build the model by calling it with dummy data
    dummy_input = tf.zeros((1, config.n_bins))
    _ = vae(dummy_input)
    
    return vae

def create_simple_vae_model(config):
    """
    Create a simple VAE model that matches the original train.py implementation.
    Uses a custom Model subclass for simplicity with modern Keras.
    """
    
    class SimpleVAE(tf.keras.Model):
        def __init__(self, encoder, decoder, kl_beta):
            super(SimpleVAE, self).__init__()
            self.encoder_model = encoder
            self.decoder_model = decoder
            self.kl_beta = kl_beta
            
        def call(self, inputs):
            z_mean, z_log_var, z = self.encoder_model(inputs)
            reconstruction = self.decoder_model(z)
            
            # Add KL loss - CORRECT FORMULA
            kl_loss = -0.5 * tf.reduce_mean(
                tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1)
            )
            self.add_loss(self.kl_beta * kl_loss)
            
            return reconstruction
    
    # Build encoder that returns mean, log_var, and sample
    original_dim = config.n_bins
    original_inputs = tf.keras.Input(shape=(original_dim,), name='encoder_input')
    x = layers.Dense(config.n_units, activation='relu')(original_inputs)
    z_mean = layers.Dense(config.latent_dim, name='z_mean')(x)
    z_log_var = layers.Dense(config.latent_dim, name='z_log_var')(x)
    z = Sampling()((z_mean, z_log_var))
    encoder = tf.keras.Model(inputs=original_inputs, outputs=[z_mean, z_log_var, z], name='encoder')
    
    # Build decoder
    latent_inputs = tf.keras.Input(shape=(config.latent_dim,), name='z_sampling')
    x = layers.Dense(config.n_units, activation='relu')(latent_inputs)
    outputs = layers.Dense(original_dim, activation=config.VAE_output_activation)(x)
    decoder = tf.keras.Model(inputs=latent_inputs, outputs=outputs, name='decoder')
    
    # Create VAE
    vae = SimpleVAE(encoder, decoder, config.kl_beta)
    
    # Store for compatibility
    vae.encoder = encoder
    vae.decoder = decoder
    
    return vae


def save_model_plots(vae, workdir):
    """Save model architecture plots.
    
    Args:
        vae: VAE model
        workdir: Working directory path
    """
    tf.keras.utils.plot_model(
        vae,
        to_file=workdir.joinpath('model_vae.jpg'),
        show_shapes=True,
        show_layer_names=True,
        rankdir='TB',
        expand_nested=True,
        dpi=300
    )

    tf.keras.utils.plot_model(
        vae.encoder,
        to_file=workdir.joinpath('model_encoder.jpg'),
        show_shapes=True,
        show_layer_names=True,
        rankdir='TB',
        expand_nested=True,
        dpi=300
    )

    tf.keras.utils.plot_model(
        vae.decoder,
        to_file=workdir.joinpath('model_decoder.jpg'),
        show_shapes=True,
        show_layer_names=True,
        rankdir='TB',
        expand_nested=True,
        dpi=300
    )