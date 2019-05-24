import numpy as np
from .utils import *
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
    dataframe
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


def validate_engagement_index(config):
    """Enagement index of subjects and hand_type.

    Parameters
    ----------
    config : yaml
        Configuration file.

    Returns
    -------
    dataframe of all enagement index
        Description of returned object.

    """
    engagement_levels = ['high', 'low']
    read_path = Path(__file__).parents[2] / config['index_validation_path']
    data = pd.read_excel(read_path)
    # Replace engagement level in dataframe
    data['engagement_level'].replace(1, 'high', inplace=True)
    data['engagement_level'].replace(0, 'low', inplace=True)

    index_validation = pd.DataFrame(np.empty((0, len(config['features']))),
                                    columns=config['features'])
    for i in range(25):
        subject_data = data[data['subject'] == i + 1]
        for level in engagement_levels:
            temp_data = subject_data[subject_data['engagement_level'] == level]
            # Cleaning
            temp_data.dropna()

            df = calculate_engagement_index(temp_data, config)
            df['subject'] = int(i + 1)
            df['engagement_level'] = level
            index_validation = pd.concat([index_validation, df],
                                         ignore_index=True,
                                         sort=False)
    return index_validation
