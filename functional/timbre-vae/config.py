# -*- coding: utf-8 -*-
"""Configuration management for VAE training."""

import configparser
import sys
from pathlib import Path


class TrainingConfig:
    """Centralized configuration management."""
    
    def __init__(self, config_path):
        self.config = configparser.ConfigParser(allow_no_value=True)
        try:
            self.config.read(config_path)
        except FileNotFoundError:
            print(f'Config File Not Found at {config_path}')
            sys.exit()
        
        self._load_config()
    
    def _load_config(self):
        """Load all configuration sections."""
        self._load_audio_config()
        self._load_dataset_config()
        self._load_training_config()
        self._load_model_config()
        self._load_extra_config()
    
    def _load_audio_config(self):
        """Load audio processing parameters."""
        self.sample_rate = self.config['audio'].getint('sample_rate')
        self.hop_length = self.config['audio'].getint('hop_length')
        self.bins_per_octave = self.config['audio'].getint('bins_per_octave')
        self.num_octaves = self.config['audio'].getint('num_octaves')
        self.n_bins = int(self.num_octaves * self.bins_per_octave)
        self.n_iter = self.config['audio'].getint('n_iter')
    
    def _load_dataset_config(self):
        """Load dataset parameters."""
        # Audio dataset path (for reading raw audio files)
        audio_dataset_str = self.config['dataset'].get('audio_dataset')
        self.audio_dataset = Path(audio_dataset_str) if audio_dataset_str else None
        if self.audio_dataset and not self.audio_dataset.exists():
            print(f"Warning: Audio dataset path does not exist: {self.audio_dataset}")
        
        # CQT dataset path (for reading preprocessed CQT .npy files)
        cqt_dataset_str = self.config['dataset'].get('cqt_dataset')
        if not cqt_dataset_str:
            raise ValueError("cqt_dataset must be specified in config")
        self.cqt_dataset = Path(cqt_dataset_str)
        if not self.cqt_dataset.exists():
            raise FileNotFoundError(f"CQT dataset not found: {self.cqt_dataset.resolve()}")
        
        # Output directory (for saving models, logs, results)
        output_dir_str = self.config['dataset'].get('output_dir')
        if not output_dir_str:
            raise ValueError("output_dir must be specified in config")
        self.output_dir = Path(output_dir_str)
        
        # Legacy compatibility: keep these aliases
        self.my_cqt = self.cqt_dataset
        self.my_audio = self.audio_dataset
        
        # Run configuration
        self.run_number = self.config['dataset'].getint('run_number')
        
        # Workspace handling (legacy - now derived from output_dir + description)
        self.workspace = None
    
    def _load_training_config(self):
        """Load training parameters."""
        self.epochs = self.config['training'].getint('epochs')
        self.learning_rate = self.config['training'].getfloat('learning_rate')
        self.batch_size = self.config['training'].getint('batch_size')
        self.train_buf = self.config['training'].getint('buffer_size')
        self.buffer_size_dataset = self.config['training'].getboolean('buffer_size_dataset')
        self.max_to_keep = self.config['training'].getint('max_ckpts_to_keep')
        self.ckpt_epochs = self.config['training'].getint('checkpoint_epochs')
        self.continue_training = self.config['training'].getboolean('continue_training')
        self.learning_schedule = self.config['training'].getboolean('learning_schedule')
        self.save_best_only = self.config['training'].getboolean('save_best_only')
        self.early_patience_epoch = self.config['training'].getint('early_patience_epoch')
        self.early_delta = self.config['training'].getfloat('early_delta')
        self.adam_beta_1 = self.config['training'].getfloat('adam_beta_1')
        self.adam_beta_2 = self.config['training'].getfloat('adam_beta_2')
    
    def _load_model_config(self):
        """Load model architecture parameters."""
        self.latent_dim = self.config['VAE'].getint('latent_dim')
        self.n_units = self.config['VAE'].getint('n_units')
        self.kl_beta = self.config['VAE'].getfloat('kl_beta')
        self.batch_normalization = self.config['VAE'].getboolean('batch_norm')
        self.VAE_output_activation = self.config['VAE'].get('output_activation')
        
        # Disentanglement parameters
        self.centroid_dim = self.config['VAE'].getint('centroid_dim', fallback=0)
        self.disentangle_weight = self.config['VAE'].getfloat('disentangle_weight', fallback=1.0)
        
        # Audio feature loss weights
        self.attack_time_weight = self.config['VAE'].getfloat('attack_time_weight', fallback=0.1)
        self.spectral_centroid_weight = self.config['VAE'].getfloat('spectral_centroid_weight', fallback=0.1)
    
    def _load_extra_config(self):
        """Load miscellaneous parameters."""
        self.example_length = self.config['extra'].getint('example_length')
        self.normalize_examples = self.config['extra'].getboolean('normalize_examples')
        self.plot_model = self.config['extra'].getboolean('plot_model')
        self.description = self.config['extra'].get('description')
    
    def update_workspace(self, workspace_path):
        """Update workspace path in config."""
        self.workspace = workspace_path
        # Don't update the config object - workspace is now derived, not configured
    
    def save_config(self, config_path):
        """Save current configuration to file."""
        with open(config_path, 'w') as configfile:
            self.config.write(configfile)