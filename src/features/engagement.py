import numpy as np
from pathlib import Path
from .utils import read_with_pickle
import pandas as pd


def calculate_engagement_index(data, config):
    """Calculate engagement_index using different features.

    Parameters
    ----------
    data : dataframe
        Dataframe containing band powers.
    config : yaml
        Configuration file.

    Returns
    -------
    df  : pandas dataframe
        Dataframe of calculated engagement indices.

    """

    # Feature Beta/(Alpha + Theta)
    num_bands = ['lower_Beta']
    num_electrodes = ['POz', 'Fz', 'Cz', 'C3', 'C4', 'F3', 'F4', 'P3', 'P4']
    den_bands = ['Theta', 'total_Alpha']
    den_electrodes = ['POz', 'Fz', 'Cz', 'C3', 'C4', 'F3', 'F4', 'P3', 'P4']
    numerator_features = [
        electrode + '_' + band for electrode in num_electrodes
        for band in num_bands
    ]
    denominator_features = [
        electrode + '_' + band for electrode in den_electrodes
        for band in den_bands
    ]
    alpha = [
        col for col in data[denominator_features].columns if 'Alpha' in col
    ]
    theta = [
        col for col in data[denominator_features].columns if 'Theta' in col
    ]
    beta_alpha_theta = data[numerator_features].mean(
        axis=1) / (data[alpha].mean(axis=1) + data[theta].mean(axis=1))

    # Feature Theta/Alpha
    num_bands = ['Theta']
    num_electrodes = ['Fz']  # frontal
    den_bands = ['total_Alpha']
    den_electrodes = ['P3', 'P4']  # perietal
    numerator_features = [
        electrode + '_' + band for electrode in num_electrodes
        for band in num_bands
    ]
    denominator_features = [
        electrode + '_' + band for electrode in den_electrodes
        for band in den_bands
    ]
    theta_alpha = data[numerator_features].mean(
        axis=1) / data[denominator_features].mean(axis=1)

    # Feature Theta
    num_bands = ['Theta']
    num_electrodes = ['F3', 'Fz', 'F4']
    numerator_features = [
        electrode + '_' + band for electrode in num_electrodes
        for band in num_bands
    ]
    theta = data[numerator_features].mean(axis=1)

    # Feature 1/Alpha
    den_bands = ['total_Alpha']
    den_electrodes = ['P3', 'POz', 'P4']
    denominator_features = [
        electrode + '_' + band for electrode in den_electrodes
        for band in den_bands
    ]
    alpha_1 = 1 / data[denominator_features].mean(axis=1)

    # Feature Beta/Theta
    num_bands = ['lower_Beta']
    num_electrodes = ['F3', 'Fz', 'F4']
    den_bands = ['Theta']
    den_electrodes = ['F3', 'Fz', 'F4']
    numerator_features = [
        electrode + '_' + band for electrode in num_electrodes
        for band in num_bands
    ]
    denominator_features = [
        electrode + '_' + band for electrode in den_electrodes
        for band in den_bands
    ]
    beta_theta = data[numerator_features].mean(
        axis=1) / data[denominator_features].mean(axis=1)

    # Feature selection Beta/Alpha
    num_bands = ['lower_Beta']
    num_electrodes = ['POz', 'Cz', 'P3', 'P4']
    den_bands = ['total_Alpha']
    den_electrodes = ['POz', 'Cz', 'P3', 'P4']
    numerator_features = [
        electrode + '_' + band for electrode in num_electrodes
        for band in num_bands
    ]
    denominator_features = [
        electrode + '_' + band for electrode in den_electrodes
        for band in den_bands
    ]
    beta_alpha = data[numerator_features].mean(
        axis=1) / data[denominator_features].mean(axis=1)

    # Form a dataframe from the calculated features
    temp_data = [
        beta_alpha_theta.values, theta_alpha.values, theta.values,
        alpha_1.values, beta_theta.values, beta_alpha.values
    ]
    df = pd.DataFrame(temp_data, config['features']).T

    return df


def engagement_index(subjects, hand_type, control_type, config):
    """Enagement index of subjects and hand_type.

    Parameters
    ----------
    subject : str
        String of subject ID e.g. 8801.
    hand_type : str
        hand_type of the subject dominant or non-dominant.
    control_type : str
        Control type (error augmentation or error reduction)
    config : yaml
        The configuration file.

    Returns
    -------
    engagement_index: pandas dataframe
        A dataframe of all enagement index

    """

    read_path = Path(__file__).parents[2] / config['band_power_dataset']
    data = read_with_pickle(read_path)
    engagement_index = pd.DataFrame(np.empty((0, len(config['features']))),
                                    columns=config['features'])
    for subject in subjects:
        subject_data = data[subject]
        for hand in hand_type:
            hand_data = subject_data[subject_data['hand_type'] == hand]
            for control in control_type:
                control_data = hand_data[hand_data['control_type'] == control]
                df = calculate_engagement_index(control_data, config)
                df['subject'] = subject
                df['hand_type'] = hand
                df['control_type'] = control

                engagement_index = pd.concat([engagement_index, df],
                                             ignore_index=True,
                                             sort=False)

    return engagement_index
