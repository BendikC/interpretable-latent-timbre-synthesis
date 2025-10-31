from __future__ import absolute_import, division, print_function, unicode_literals

import random
import numpy as np
import os, sys, argparse, time
import librosa
from tqdm import tqdm
import configparser
from pathlib import Path
import json
from collections import defaultdict

#Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='./default.ini', help='path to the config file')
parser.add_argument('--stratified', action='store_true', help='Enable stratified sampling')
parser.add_argument('--samples_per_instrument', type=int, default=100, help='Number of samples per instrument family')
parser.add_argument('--instruments', type=str, default=None, help='Comma-separated list of instrument families (e.g., "bass,brass,guitar")')
parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
args = parser.parse_args()

# Set random seed
random.seed(args.seed)
np.random.seed(args.seed)

#Get configs
config_path = args.config
config = configparser.ConfigParser(allow_no_value=True)
config.read(config_path)

#audio configs
sample_rate = config['audio'].getint('sample_rate')
hop_length = config['audio'].getint('hop_length')
bins_per_octave = config['audio'].getint('bins_per_octave')
num_octaves = config['audio'].getint('num_octaves')
n_bins = num_octaves * bins_per_octave
n_iter = config['audio'].getint('n_iter')

#dataset - read from audio_dataset, write to cqt_dataset
audio_dataset_path = Path(config['dataset'].get('audio_dataset'))
cqt_dataset_path = Path(config['dataset'].get('cqt_dataset'))

# Create output directory for CQT files
os.makedirs(cqt_dataset_path, exist_ok=True)

# Audio files are directly in audio_dataset_path
my_audio_folder = audio_dataset_path


def parse_nsynth_filename(filename):
    """
    Parse NSynth filename to extract instrument family.
    NSynth format: instrument_family-source-pitch-velocity.wav
    Example: bass_synthetic_001-src_002-pitch_036-velocity_025.wav
    """
    parts = filename.replace('.wav', '').split('-')
    if len(parts) >= 1:
        # First part contains instrument family
        instrument_family = parts[0].split('_')[0]  # e.g., 'bass' from 'bass_synthetic_001'
        return instrument_family
    return 'unknown'


def stratified_sampling(audio_folder, samples_per_instrument, target_instruments=None):
    """
    Perform stratified sampling on NSynth dataset.
    
    Args:
        audio_folder: Path to audio files
        samples_per_instrument: Number of samples to take per instrument family
        target_instruments: List of instrument families to include (None = all)
    
    Returns:
        List of selected filenames
    """
    print("\n=== Stratified Sampling ===")
    
    # Group files by instrument family
    instrument_files = defaultdict(list)
    all_files = [f for f in os.listdir(audio_folder) if f.endswith('.wav')]
    
    print(f"Total audio files found: {len(all_files)}")
    
    for f in all_files:
        instrument = parse_nsynth_filename(f)
        instrument_files[instrument].append(f)
    
    # Show distribution
    print("\nInstrument distribution:")
    for instrument, files in sorted(instrument_files.items()):
        print(f"  {instrument:20s}: {len(files):5d} files")
    
    # Filter by target instruments if specified
    if target_instruments:
        target_instruments = set(target_instruments)
        instrument_files = {k: v for k, v in instrument_files.items() 
                          if k in target_instruments}
        print(f"\nFiltered to target instruments: {target_instruments}")
    
    # Sample from each instrument family
    selected_files = []
    print("\nSampling strategy:")
    
    for instrument, files in sorted(instrument_files.items()):
        available = len(files)
        n_samples = min(samples_per_instrument, available)
        
        # Random sampling without replacement
        sampled = random.sample(files, n_samples)
        selected_files.extend(sampled)
        
        print(f"  {instrument:20s}: sampling {n_samples:4d} / {available:5d} files")
    
    print(f"\nTotal selected: {len(selected_files)} files")
    
    # Save sampling manifest
    manifest = {
        'total_files': len(selected_files),
        'samples_per_instrument': samples_per_instrument,
        'target_instruments': list(target_instruments) if target_instruments else 'all',
        'seed': args.seed,
        'instrument_counts': {k: len([f for f in selected_files if parse_nsynth_filename(f) == k]) 
                             for k in instrument_files.keys()}
    }
    
    manifest_path = cqt_dataset_path / 'sampling_manifest.json'
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    print(f"\n✓ Saved sampling manifest to {manifest_path}")
    
    return selected_files


# Determine which files to process
if args.stratified:
    # Parse target instruments if specified
    target_instruments = None
    if args.instruments:
        target_instruments = [inst.strip() for inst in args.instruments.split(',')]
        print(f"Target instruments: {target_instruments}")
    
    # Perform stratified sampling
    files_to_process = stratified_sampling(
        my_audio_folder, 
        args.samples_per_instrument,
        target_instruments
    )
else:
    # Process all files
    files_to_process = [f for f in os.listdir(my_audio_folder) if f.endswith('.wav')]
    print(f'TOTAL FILES: {len(files_to_process)}')

# Save config to CQT directory for reference
with open(cqt_dataset_path / 'config.ini', 'w') as configfile:
    config.write(configfile)

current_num = 0

if __name__ == '__main__': 
    for f in files_to_process:
        print(f'DONE: {current_num}/{len(files_to_process)}')
        my_audio = os.path.join(my_audio_folder, f)
        outfile = os.path.join(cqt_dataset_path, os.path.splitext(f)[0])
        
        if os.path.exists(outfile + '.npy'):
            print(f'{outfile} exists. Next file...')
            current_num += 1
            continue
        
        try:
            print(f'loading...: {my_audio}')    
            s, fs = librosa.load(my_audio, sr=None)
            if fs != sample_rate:
                print(f"resampling {my_audio}")
                s, fs = librosa.load(my_audio, sr=sample_rate)
            
            print('get CQTs...')
            # Get the CQT magnitude
            C_complex = librosa.cqt(
                y=s, 
                sr=sample_rate, 
                hop_length=hop_length, 
                bins_per_octave=bins_per_octave, 
                n_bins=n_bins
            )
            C = np.abs(C_complex)
            
            print('data processing...')
            # TensorFlow expects the transpose of librosa's output
            C = np.transpose(C)
            # TensorFlow works in float32; CQT arrays are float64
            C = C.astype('float32')
            
            print(f'writing: {outfile}')
            np.save(outfile, C)
            current_num += 1
        except Exception as e:
            print(f'There was an issue with {f}: {e}')    
            current_num += 1
            continue

print(f"\n✓ Dataset creation complete!")
print(f"  Processed: {current_num}/{len(files_to_process)} files")
print(f"  Output directory: {cqt_dataset_path}")