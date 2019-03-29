from data.create_eeg_dataset import eeg_dataset
from utils import *
import yaml

config = yaml.load(open('config.yml'))

with skip_run_code('run', 'create_eeg_dataset') as check, check():
    eeg_dataset = eeg_dataset(config['subjects'], config['hand_type'])
    save_path = Path(__file__).parents[1] / config['raw_eeg_dataset']
    save_dataset(str(save_path), eeg_dataset, save=True)
