from __future__ import absolute_import, division, print_function, unicode_literals

import random
import numpy as np
import os, sys, argparse, time
import librosa
from tqdm import tqdm
import configparser
from pathlib import Path

#Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='./default.ini', help='path to the config file')
args = parser.parse_args()

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
pbar = len([f for f in os.listdir(my_audio_folder) if f.endswith('.wav')])
print(f'TOTAL FILES: {pbar}')

# Save config to CQT directory for reference
with open(cqt_dataset_path / 'config.ini', 'w') as configfile:
    config.write(configfile)

current_num = 0

if __name__ == '__main__': 
    for f in os.listdir(my_audio_folder): 
        if os.path.isfile(os.path.join(my_audio_folder, f)) and f.endswith('.wav'):
            print(f'DONE: {current_num}')
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