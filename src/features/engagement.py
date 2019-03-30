import numpy as np
from .utils import *
import pandas as pd


def get_engagement_index(data, config):
    """Calculate engagement_index using different features.

    Parameters
    ----------
    data : dataframe
        Dataframe containing band powers.
    config : yaml
        Configuration file.

    Returns
    -------
    dataframe
        Dataframe of calculated engagement indices.

    """

    # Feature Beta/(Alpha + Theta)
    num_bands = ['total_Beta']
    num_electrodes = ['POz','Fz','Cz','C3','C4','F3','F4','P3','P4']
    den_bands = ['Theta', 'total_Alpha']
    den_electrodes = ['POz','Fz','Cz','C3','C4','F3','F4','P3','P4']
    numerator_features = [electrode + '_' + band for electrode in num_electrodes for band in num_bands]
    denominator_features = [electrode + '_' + band for electrode in den_electrodes for band in den_bands]
    alpha = [col for col in data[denominator_features].columns if 'Alpha' in col]
    theta = [col for col in data[denominator_features].columns if 'Theta' in col]
    beta_alpha_theta = data[numerator_features].mean(axis = 1)/(data[alpha].mean(axis = 1) + data[theta].mean(axis = 1))

    # Feature Theta/Alpha
    num_bands = ['Theta']
    num_electrodes = ['Fz'] # frontal
    den_bands = ['total_Alpha']
    den_electrodes = ['P3', 'P4'] # perietal
    numerator_features = [electrode + '_' + band for electrode in num_electrodes for band in num_bands]
    denominator_features = [electrode + '_' + band for electrode in den_electrodes for band in den_bands]
    theta_alpha = data[numerator_features].mean(axis = 1)/ data[denominator_features].mean(axis = 1)

    # Feature Theta
    num_bands = ['Theta']
    num_electrodes = ['F3','Fz','F4']
    numerator_features = [electrode + '_' + band for electrode in num_electrodes for band in num_bands]
    theta = data[numerator_features].mean(axis = 1)

    # Feature 1/Alpha
    den_bands = ['lower_Alpha']
    den_electrodes = ['P3','POz','P4']
    denominator_features = [electrode + '_' + band for electrode in den_electrodes for band in den_bands]
    alpha_1 = 1/data[denominator_features].mean(axis = 1)

    # Feature Beta/Theta
    num_bands = ['lower_Beta']
    num_electrodes = ['F3','Fz','F4']
    den_bands = ['Theta']
    den_electrodes = ['F3','Fz','F4']
    numerator_features = [electrode + '_' + band for electrode in num_electrodes for band in num_bands]
    denominator_features = [electrode + '_' + band for electrode in den_electrodes for band in den_bands]
    beta_theta = data[numerator_features].mean(axis = 1)/data[denominator_features].mean(axis = 1)

    # Feature selection Beta/Alpha
    num_bands = [ 'total_Beta']
    num_electrodes = ['P3','POz','P4']
    den_bands = ['lower_Alpha']
    den_electrodes = ['P3','POz','P4']
    numerator_features = [electrode + '_' + band for electrode in num_electrodes for band in num_bands]
    denominator_features = [electrode + '_' + band for electrode in den_electrodes for band in den_bands]
    beta_alpha = data[numerator_features].mean(axis = 1)/data[denominator_features].mean(axis = 1)

    # Form a dataframe from the calculated features
    temp_data = [beta_alpha_theta.values, theta_alpha.values, theta.values, alpha_1.values, beta_theta.values, beta_alpha.values]
    df = pd.DataFrame(temp_data, config['features']).T

    return df


def engagement_index(subjects, hand_type, config):
    """Enagement index of subjects and hand_type.

    Parameters
    ----------
    subjects : list
        List of all the subjects.
    hand_type : list
        List of hand types dominant or non-dominant.
    config : yaml
        The configuration file.

    Returns
    -------
    dataframe of all enagement index
        Description of returned object.

    """

    read_path = Path(__file__).parents[2] / config['band_power_dataset']
    all_data = read_dataframe_dict(read_path)
    engagement_index = pd.DataFrame(np.empty((0,len(config['features']))), columns=config['features'])
    for subject in subjects:
        temp_data = all_data[subject]
        for hand in hand_type:
            data = temp_data[temp_data['hand_type']==hand]
            df = get_engagement_index(data, config)
            df['hand_type'] = hand
        df['subject'] = subject
        engagement_index =  pd.concat([engagement_index, df], ignore_index=True, sort=False)

    return engagement_index
