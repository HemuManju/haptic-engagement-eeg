import yaml
import pandas as pd
import numpy as np
from utils import *
from data.create_eeg_dataset import eeg_dataset
from data.create_haptic_dataset import haptic_dataset
from data.clean_eeg_dataset import clean_dataset
from features.band_power import band_power_dataset
from features.engagement import engagement_index
from features.emg_features import create_emg_features
from features.utils import save_to_r_dataset, read_with_pickle, read_with_deepdish
from models.index_validation import validate_engagement_index

config = yaml.load(open('config.yml'), Loader=yaml.FullLoader)

with skip_run('skip', 'create_eeg_dataset') as check, check():
    eeg_dataset = eeg_dataset(config)
    save_path = Path(__file__).parents[1] / config['raw_eeg_dataset']
    save_with_deepdish(str(save_path), eeg_dataset, save=True)

with skip_run('skip', 'clean_eeg_dataset') as check, check():
    clean_dataset = clean_dataset(config)
    save_path = Path(__file__).parents[1] / config['clean_eeg_dataset']
    save_with_deepdish(str(save_path), clean_dataset, save=True)

with skip_run('run', 'index_validation') as check, check():
    index_validation_dataset = validate_engagement_index(config)
    print(index_validation_dataset)
    save_path = Path(__file__).parents[1] / config['index_validation_dataset']
    save_with_deepdish(str(save_path), index_validation_dataset, save=True)

with skip_run('run', 'convert_enagement_to_r_dataset') as check, check():
    # Read the pandas dataframe
    read_path = Path(__file__).parents[1] / config['index_validation_dataset']
    df = read_with_deepdish(read_path)
    save_path = Path(__file__).parents[1] / config['validate_r_dataset']
    save_to_r_dataset(df, str(save_path))

with skip_run('skip', 'band_power') as check, check():
    band_power_dataset = band_power_dataset(config)
    save_path = Path(__file__).parents[1] / config['band_power_dataset']
    save_with_pickle(str(save_path), band_power_dataset, save=True)

with skip_run('skip', 'engagement_index') as check, check():
    engagement_index_dataset = engagement_index(config['subjects'],
                                                config['hand_type'],
                                                config['control_type'], config)
    save_path = Path(__file__).parents[1] / config['engagement_index_dataset']
    save_with_pickle(str(save_path), engagement_index_dataset, save=True)

with skip_run('skip', 'convert_enagement_to_r_dataset') as check, check():
    # Read the pandas dataframe
    read_path = Path(__file__).parents[1] / config['engagement_index_dataset']
    df = read_with_pickle(read_path)
    save_path = Path(__file__).parents[1] / config['eeg_r_dataset']
    save_to_r_dataset(df, str(save_path))

with skip_run('skip', 'create_haptic_dataset') as check, check():
    haptic_dataset = haptic_dataset(config)
    save_path = Path(__file__).parents[1] / config['raw_haptic_dataset']
    save_with_deepdish(str(save_path), haptic_dataset, save=True)

with skip_run('skip', 'emg_features') as check, check():
    emg_dataset = create_emg_features(config)
    save_path = Path(__file__).parents[1] / config['emg_dataset']
    save_with_pickle(str(save_path), emg_dataset, save=True)

with skip_run('skip', 'convert_emg_to_r_dataset') as check, check():
    # Read the pandas dataframe
    read_path = Path(__file__).parents[1] / config['emg_dataset']
    df = read_with_pickle(read_path)
    save_path = Path(__file__).parents[1] / config['emg_r_dataset']
    save_to_r_dataset(df, str(save_path))
