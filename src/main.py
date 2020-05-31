import yaml
from pathlib import Path
import matplotlib.pyplot as plt

from data.create_eeg_dataset import eeg_dataset
from data.create_haptic_dataset import (haptic_dataset, haptic_force_dataset)
from data.clean_eeg_dataset import clean_dataset

from features.band_power import band_power_dataset
from features.engagement import (engagement_index,
                                 avg_engagement_index_with_force)
from features.emg_features import create_emg_features
from features.haptic_features import create_haptic_features
from features.utils import (save_to_r_dataset, read_with_pickle,
                            read_with_deepdish)

from models.index_validation import validate_engagement_index

from visualization.visualize import (topo_map, force_error,
                                     plot_mixed_effect_model)

from utils import (skip_run, save_with_deepdish, save_with_pickle)

# The configuration file
config_path = Path(__file__).parents[1] / 'src/config.yml'
config = yaml.load(open(str(config_path)), Loader=yaml.SafeLoader)

with skip_run('skip', 'create_eeg_dataset') as check, check():
    eeg_dataset = eeg_dataset(config)
    save_path = Path(__file__).parents[1] / config['raw_eeg_dataset']
    save_with_deepdish(str(save_path), eeg_dataset, save=True)

with skip_run('skip', 'clean_eeg_dataset') as check, check():
    clean_dataset = clean_dataset(config)
    save_path = Path(__file__).parents[1] / config['clean_eeg_dataset']
    save_with_deepdish(str(save_path), clean_dataset, save=True)

with skip_run('skip', 'index_validation') as check, check():
    index_validation_dataset = validate_engagement_index(config)
    print(index_validation_dataset)
    save_path = Path(__file__).parents[1] / config['index_validation_dataset']
    save_with_deepdish(str(save_path), index_validation_dataset, save=True)

with skip_run('skip', 'convert_enagement_to_r_dataset') as check, check():
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

with skip_run('skip', 'engagement_index_with_laterality') as check, check():
    engagement_index_dataset = engagement_index(config['subjects'],
                                                config['hand_type'],
                                                config['control_type'], config)
    save_path = Path(__file__).parents[1] / config[
        'engagement_index_dataset_with_laterality']
    save_with_pickle(str(save_path), engagement_index_dataset, save=True)

with skip_run('skip', 'convert_enagement_to_r_dataset') as check, check():
    # Read the pandas dataframe
    read_path = Path(__file__).parents[1] / config['engagement_index_dataset']
    df = read_with_pickle(read_path)
    save_path = Path(__file__).parents[1] / config['eeg_r_dataset']
    save_to_r_dataset(df, str(save_path))

with skip_run('skip', 'convert_enag_later_to_r_dataset') as check, check():
    # Read the pandas dataframe
    read_path = Path(__file__).parents[1] / config[
        'engagement_index_dataset_with_laterality']
    df = read_with_pickle(read_path)
    save_path = Path(__file__).parents[1] / config['eeg_laterality_r_dataset']
    save_to_r_dataset(df, str(save_path))

with skip_run('skip', 'create_haptic_dataset') as check, check():
    haptic_dataset = haptic_dataset(config)
    save_path = Path(__file__).parents[1] / config['raw_haptic_dataset']
    save_with_deepdish(str(save_path), haptic_dataset, save=True)

with skip_run('skip', 'haptic_features') as check, check():
    emg_dataset = create_haptic_features(config)
    print(emg_dataset)
    save_path = Path(__file__).parents[1] / config['haptic_dataset']
    save_with_pickle(str(save_path), emg_dataset, save=False)

with skip_run('skip', 'convert_haptic_to_r_dataset') as check, check():
    # Read the pandas dataframe
    read_path = Path(__file__).parents[1] / config['haptic_dataset']
    df = read_with_pickle(read_path)
    save_path = Path(__file__).parents[1] / config['haptic_r_dataset']
    save_to_r_dataset(df, str(save_path))

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

with skip_run('skip', 'plot_topomap') as check, check():
    subjects = config['subjects'][1]
    hand_type = config['hand_type'][1]
    control_type = config['control_type']
    topo_map(subjects, hand_type, control_type, config)

with skip_run('skip', 'plot_force_error') as check, check():
    subjects = config['subjects'][2]
    hand_type = config['hand_type'][1]
    control_type = config['control_type'][1]
    print(subjects, hand_type, control_type)
    force_error(subjects, hand_type, control_type, config)

with skip_run('skip', 'sync_force_data') as check, check():
    force_dataset = haptic_force_dataset(config)
    save_path = Path(__file__).parents[1] / config['haptic_force_dataset']
    save_with_deepdish(str(save_path), force_dataset, save=True)

with skip_run('skip', 'sync_force_eeg_save_to_r') as check, check():
    df = avg_engagement_index_with_force(config)
    save_path = Path(
        __file__).parents[1] / config['engagement_index_force_dataset']
    save_to_r_dataset(df, str(save_path))

with skip_run('skip', 'plot_mixed_effect_predictions') as check, check():
    plot_mixed_effect_model(config, 'non_dominant')
    plot_mixed_effect_model(config, 'dominant')
    plt.show()