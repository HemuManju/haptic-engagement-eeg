import yaml
import pandas as pd
import numpy as np
from utils import *
from data.create_eeg_dataset import eeg_dataset
from data.haptics_utils import get_haptic_path, get_haptic_time
from data.clean_eeg_dataset import clean_dataset
from features.band_power import band_power_dataset
from features.engagement import engagement_index
from features.utils import save_to_r_dataset, read_dataframe_dict


config = yaml.load(open('config.yml'))


with skip_run_code('skip', 'create_eeg_dataset') as check, check():
    eeg_dataset = eeg_dataset(config['subjects'], config['hand_type'], config['control_type'],
    config)
    save_path = Path(__file__).parents[1] / config['raw_eeg_dataset']
    save_dataset(str(save_path), eeg_dataset, save=True)


with skip_run_code('skip', 'clean_eeg_dataset') as check, check():
    clean_dataset = clean_dataset(config['subjects'], config['hand_type'], config['control_type'],
    config)
    save_path = Path(__file__).parents[1] / config['clean_eeg_dataset']
    save_dataset(str(save_path), clean_dataset, save=True)


with skip_run_code('skip', 'create_haptic_dataset') as check, check():
    clean_dataset = haptic_dataset(config['subjects'], config['hand_type'],config['control_type'],
    config)
    save_path = Path(__file__).parents[1] / config['raw_emg_dataset']
    save_dataset(str(save_path), clean_dataset, save=True)


with skip_run_code('skip', 'band_power_dataset') as check, check():
    band_power_dataset = band_power_dataset(config['subjects'], config['hand_type'], config['control_type'], config)
    save_path = Path(__file__).parents[1] / config['band_power_dataset']
    save_dataframe_dict(str(save_path), band_power_dataset, save=True)


with skip_run_code('skip', 'engagement_index') as check, check():
    engagement_index_dataset = engagement_index(config['subjects'], config['hand_type'], config['control_type'], config)
    save_path = Path(__file__).parents[1] / config['engagement_index_dataset']
    save_dataframe_dict(str(save_path), engagement_index_dataset, save=True)


with skip_run_code('run', 'convert_to_r_dataset') as check, check():
    # Read the pandas dataframe
    read_path = Path(__file__).parents[1] / config['engagement_index_dataset']
    df = read_dataframe_dict(read_path)
    save_to_r_dataset(df, config)
