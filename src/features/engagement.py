import numpy as np
from pathlib import Path
from .utils import read_with_pickle, read_with_deepdish
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

    # Alpha laterality index
    num_bands = ['total_Alpha']
    num_electrodes = ['P3']
    den_bands = ['total_Alpha']
    den_electrodes = ['P4']
    numerator_features = [
        electrode + '_' + band for electrode in num_electrodes
        for band in num_bands
    ]
    denominator_features = [
        electrode + '_' + band for electrode in den_electrodes
        for band in den_bands
    ]
    alpha_lat = (data[denominator_features].mean(axis=1) -
                 data[numerator_features].mean(axis=1)) / (
                     data[denominator_features].mean(axis=1) +
                     data[numerator_features].mean(axis=1))

    # Theta laterality index
    num_bands = ['Theta']
    num_electrodes = ['F3']
    den_bands = ['Theta']
    den_electrodes = ['F4']
    numerator_features = [
        electrode + '_' + band for electrode in num_electrodes
        for band in num_bands
    ]
    denominator_features = [
        electrode + '_' + band for electrode in den_electrodes
        for band in den_bands
    ]
    theta_lat = (data[denominator_features].mean(axis=1) -
                 data[numerator_features].mean(axis=1)) / (
                     data[denominator_features].mean(axis=1) +
                     data[numerator_features].mean(axis=1))

    # Mu laterality index
    num_bands = ['higher_Alpha']
    num_electrodes = ['C3']
    den_bands = ['higher_Alpha']
    den_electrodes = ['C4']
    numerator_features = [
        electrode + '_' + band for electrode in num_electrodes
        for band in num_bands
    ]
    denominator_features = [
        electrode + '_' + band for electrode in den_electrodes
        for band in den_bands
    ]
    mu_lat = (data[denominator_features].mean(axis=1) -
              data[numerator_features].mean(axis=1)) / (
                  data[denominator_features].mean(axis=1) +
                  data[numerator_features].mean(axis=1))

    # Alpha frontal asymmetry
    num_bands = ['total_Alpha']
    num_electrodes = ['F3']
    den_bands = ['total_Alpha']
    den_electrodes = ['F4']
    numerator_features = [
        electrode + '_' + band for electrode in num_electrodes
        for band in num_bands
    ]
    denominator_features = [
        electrode + '_' + band for electrode in den_electrodes
        for band in den_bands
    ]
    alpha_front_lat = (data[denominator_features].mean(axis=1) -
                       data[numerator_features].mean(axis=1)) / (
                           data[denominator_features].mean(axis=1) +
                           data[numerator_features].mean(axis=1))

    # # SMR
    # num_bands = ['mu_Rythm']
    # num_electrodes = ['C3']
    # den_bands = ['mu_Rythm']
    # den_electrodes = ['C4']
    # numerator_features = [
    #     electrode + '_' + band for electrode in num_electrodes
    #     for band in num_bands
    # ]
    # denominator_features = [
    #     electrode + '_' + band for electrode in den_electrodes
    #     for band in den_bands
    # ]
    # smr = (data[denominator_features].mean(axis=1) +
    #        data[numerator_features].mean(axis=1)) / 2

    # Form a dataframe from the calculated features
    temp_data = [
        beta_alpha_theta.values, theta_alpha.values, theta.values,
        alpha_1.values, beta_theta.values, beta_alpha.values
        # , alpha_lat.values,theta_lat.values, mu_lat.values,
        # alpha_front_lat.values, smr.values
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


def engagement_index_with_force(config):
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

    # Force data
    read_path = Path(__file__).parents[2] / config['haptic_force_dataset']
    force_data = read_with_deepdish(read_path)

    for subject in config['subjects']:
        subject_data = data[subject]
        for hand in config['hand_type']:
            hand_data = subject_data[subject_data['hand_type'] == hand]
            for control in config['control_type']:

                control_data = hand_data[hand_data['control_type'] == control]
                f_data = force_data[subject]['haptic'][hand][control]
                f_data = f_data.get_data(['total_force']).mean(axis=-1)
                df = calculate_engagement_index(control_data, config)

                df['subject'] = subject
                df['hand_type'] = hand
                df['control_type'] = control
                # Assert the length are same
                assert f_data.shape[0] == df.shape[
                    0], 'Data are of different length'

                df['total_force'] = f_data

                engagement_index = pd.concat([engagement_index, df],
                                             ignore_index=True,
                                             sort=False)

    return engagement_index


def avg_engagement_index_with_force(config):
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

    # Force data
    read_path = Path(__file__).parents[2] / config['haptic_force_dataset']
    force_data = read_with_deepdish(read_path)

    # Read mu rythm
    read_path = Path(__file__).parents[2] / config['laterality_dataset']
    laterality_data = read_with_pickle(read_path)

    for subject in config['subjects']:
        subject_data = data[subject]
        for hand in config['hand_type']:
            hand_data = subject_data[subject_data['hand_type'] == hand]
            for control in config['control_type']:

                # Engagement data
                control_data = hand_data[hand_data['control_type'] == control]
                df = calculate_engagement_index(control_data, config)

                # Force data
                f_data = force_data[subject]['haptic'][hand][control]
                f_data = f_data.get_data(['total_force']).mean(axis=-1)

                # mu_rythm data
                subject_id = laterality_data['subject'] == subject
                hand_type = laterality_data['hand_type'] == hand
                control_type = laterality_data['control_type'] == control
                lat_data = laterality_data[subject_id & hand_type
                                           & control_type]

                # Assert the length are same
                assert f_data.shape[0] == lat_data.shape[
                    0], 'Data are of different length'

                # Average them
                df['smr'] = lat_data['smr'].values
                df['total_force'] = f_data

                moving_window = config['moving_window']
                df = df.rolling(moving_window).mean().dropna()[::moving_window]
                df['subject'] = subject
                df['hand_type'] = hand
                df['control_type'] = control
                df = df.reset_index(drop=True)

                engagement_index = pd.concat([engagement_index, df],
                                             ignore_index=True,
                                             sort=False)

    return engagement_index
