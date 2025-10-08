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
        self.dataset_path = Path(self.config['dataset'].get('datapath'))
        if not self.dataset_path.exists():
            raise FileNotFoundError(self.dataset_path.resolve())
        
        self.cqt_dataset = self.config['dataset'].get('cqt_dataset')
        self.run_number = self.config['dataset'].getint('run_number')
        self.my_cqt = self.dataset_path / self.cqt_dataset
        self.my_audio = self.dataset_path / 'audio'
        
        if not self.my_cqt.exists():
            raise FileNotFoundError(self.my_cqt.resolve())
        
        # Workspace handling
        if self.config['dataset'].get('workspace') is not None:
            self.workspace = Path(self.config['dataset'].get('workspace'))
        else:
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
        
        # Audio feature loss weights
        self.attack_time_weight = self.config['VAE'].getfloat('attack_time_weight') if self.config.has_option('VAE', 'attack_time_weight') else 0.1
        self.spectral_centroid_weight = self.config['VAE'].getfloat('spectral_centroid_weight') if self.config.has_option('VAE', 'spectral_centroid_weight') else 0.1
    
    def _load_extra_config(self):
        """Load miscellaneous parameters."""
        self.example_length = self.config['extra'].getint('example_length')
        self.normalize_examples = self.config['extra'].getboolean('normalize_examples')
        self.plot_model = self.config['extra'].getboolean('plot_model')
        self.description = self.config['extra'].get('description')
    
    def update_workspace(self, workspace_path):
        """Update workspace path in config."""
        self.workspace = workspace_path
        self.config['dataset']['workspace'] = str(workspace_path.resolve())
    
    def save_config(self, config_path):
        """Save current configuration to file."""
        with open(config_path, 'w') as configfile:
            self.config.write(configfile)