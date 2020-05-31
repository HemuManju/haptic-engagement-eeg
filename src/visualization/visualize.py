from pathlib import Path

import numpy as np
import pandas as pd

import scipy

import mne
from mne.time_frequency import psd_multitaper

import matplotlib.pyplot as plt

from data.haptics_utils import create_haptic_emg_epoch

from .utils import read_eeg_epochs


def topo_map(subjects, hand_type, control_type, config):

    plt.rcParams.update({'font.size': 14})
    # Convert the input if it is not a list
    if type(subjects) != list:
        subjects = [subjects]

    fig, ax = plt.subplots(len(subjects), len(control_type), figsize=[13, 5])
    freq_band = [13, 15]

    for j, subject in enumerate(subjects):
        for i, control in enumerate(control_type):
            if len(subjects) == 1:
                plot_axes = ax[i]
            else:
                plot_axes = ax[j, i]

            epochs = read_eeg_epochs(subject, hand_type, control, config)
            picks = mne.pick_types(epochs.info, eeg=True)
            info = mne.pick_info(epochs.info,
                                 mne.pick_types(epochs.info, eeg=True))
            psds, freqs = psd_multitaper(epochs,
                                         fmin=1.0,
                                         fmax=64.0,
                                         picks=picks,
                                         n_jobs=6,
                                         verbose=False,
                                         normalization='full')
            power = psds[:, :, (freqs >= freq_band[0]) &
                         (freqs < freq_band[1])].mean(axis=0)
            img, cn = mne.viz.plot_topomap(power.mean(axis=1) * 10e12,
                                           pos=info,
                                           axes=plot_axes,
                                           show=False,
                                           cmap='viridis',
                                           vmax=7,
                                           extrapolate='local')
            clb = plot_axes.figure.colorbar(img,
                                            ax=plot_axes,
                                            fraction=0.046,
                                            pad=0.04)
            clb.ax.set_title('$\mu V^2$')
            title = control_type[i].split('_')
            plot_axes.set_title(' '.join(title) + '\n ($\mu$ rhythm)')

    # plt.show()
    if hand_type == 'dominant':
        plt.suptitle('Dominant hand')
    else:
        plt.suptitle('Non dominant hand')
    plt.tight_layout()
    save_path = Path(__file__).parents[2]
    if hand_type == 'dominant':
        plt.savefig(
            str(save_path) + '/reports/figures/topomap_dominant_hand.pdf')
    else:
        plt.savefig(
            str(save_path) + '/reports/figures/topomap_non_dominant_hand.pdf')
    return None


def force_error(subject, hand_type, control_type, config):
    epochs = create_haptic_emg_epoch(subject, hand_type, control_type, config)
    total_force = epochs.get_data(picks='total_force')
    total_force = np.mean(total_force, axis=2)

    total_error = epochs.get_data(picks='total_error')
    total_error = np.mean(total_error, axis=2)

    plt.plot(total_force)
    # plt.plot(total_error)
    plt.show()


def scatterfit(x, y, a=None, b=None):
    """
    Compute the mean deviation of the data about the linear model given if A,B
    (y=ax+b) provided as arguments. Otherwise, compute the mean deviation about
    the best-fit line.

    x,y assumed to be np arrays. a,b scalars.
    Returns the float sd with the mean deviation.

    Author: Rodrigo Nemmen
    """

    if a is None:
        # Performs linear regression
        a, b, r, p, err = scipy.stats.linregress(x, y)

    # Std. deviation of an individual measurement (Bevington, eq. 6.15)
    N = np.size(x)
    sd = 1. / (N - 1.) * np.sum((y - a * x - b)**2)
    sd = np.sqrt(sd)

    return sd


def confband(xd, yd, a, b, conf=0.95, x=None):
    """
    Calculates the confidence band of the linear regression model at the desired confidence
    level, using analytical methods. The 2sigma confidence interval is 95% sure to contain
    the best-fit regression line. This is not the same as saying it will contain 95% of
    the data points.
    Arguments:
    - conf: desired confidence level, by default 0.95 (2 sigma)
    - xd,yd: data arrays
    - a,b: linear fit parameters as in y=ax+b
    - x: (optional) array with x values to calculate the confidence band. If none is provided, will
    by default generate 100 points in the original x-range of the data.

    Returns:
    Sequence (lcb,ucb,x) with the arrays holding the lower and upper confidence bands
    corresponding to the [input] x array.
    Usage:
    >>> lcb,ucb,x=nemmen.confband(all.kp,all.lg,a,b,conf=0.95)
    calculates the confidence bands for the given input arrays
    >>> pylab.fill_between(x, lcb, ucb, alpha=0.3, facecolor='gray')
    plots a shaded area containing the confidence band
    References:
    1. http://en.wikipedia.org/wiki/Simple_linear_regression, see Section Confidence intervals
    2. http://www.weibull.com/DOEWeb/confidence_intervals_in_simple_linear_regression.htm
    Author: Rodrigo Nemmen
    v1 Dec. 2011
    v2 Jun. 2012: corrected bug in computing dy
    """
    alpha = 1. - conf  # significance
    n = xd.size  # data sample size

    if x is None:
        x = np.linspace(xd.min(), xd.max(), 100)

    # Predicted values (best-fit model)
    y = a * x + b

    # Auxiliary definitions
    sd = scatterfit(xd, yd, a, b)  # Scatter of data about the model
    sxd = np.sum((xd - xd.mean())**2)
    sx = (x - xd.mean())**2  # array

    # Quantile of Student's t distribution for p=1-alpha/2
    q = scipy.stats.t.ppf(1. - alpha / 2., n - 2)

    # Confidence band
    dy = q * sd * np.sqrt(1. / n + sx / sxd)
    ucb = y + dy  # Upper confidence band
    lcb = y - dy  # Lower confidence band

    return lcb, ucb, x


def plot_mixed_effect_model(config, hand_type):
    read_path = Path(__file__).parents[
        1] / 'visualization/df_both_hand_with_predictions.csv'
    data = pd.read_csv(read_path)
    data['subject'] = data['subject'].astype(str)
    data['hand_type'] = data['hand_type'].astype(str)
    data['control_type'] = data['control_type'].astype(str)

    plt.style.use('clean')
    fig, ax = plt.subplots(figsize=(8, 6),
                           nrows=3,
                           ncols=2,
                           sharex=True,
                           sharey=True)
    row, col = np.indices((3, 2))
    row, col = row.flatten(), col.flatten()
    subjects = [
        x for x in config['subjects']
        if x not in ['8823', '8830', '8819', '8820']
    ]
    for i, subject in enumerate(subjects):
        for hand in config['hand_type']:
            temp = data[(data['subject'] == subject)
                        & (data['hand_type'] == hand)]
            x = temp['total_force']
            y_pred = temp['predicted_values']
            y_true = np.log(temp['beta_alpha_theta'])

            if hand == hand_type:
                r = row[i]
                c = col[i]
                ax[r, c].scatter(x, y_pred, c='b')
                ax[r, c].scatter(x, y_true, c='r', marker='s', label='True')
                # Fit a line
                m, b = np.polyfit(x, y_pred, 1)
                ax[r, c].plot(x, m * x + b, 'k', label='Regression')
                ax[r, c].grid(True)

                lcb, ucb, x_test = confband(x, y_true, m, b)
                ax[r, c].fill_between(x_test, lcb, ucb, alpha=0.50)
                # ax[r, c].set_ylim([-1.0, -0.5])
                if i == 3:
                    ax[r, c].legend(loc='upper right',
                                    ncol=1,
                                    borderpad=0,
                                    framealpha=0.0)

    # handles, labels = ax[0, 0].get_legend_handles_labels()
    # fig.legend(handles, labels, loc='right', ncol=1)
    plt.setp(ax[2, 1], xlabel='Force (N)')
    plt.setp(ax[2, 0], xlabel='Force (N)')
    plt.setp(ax[1, 0], ylabel=r'$ln(\beta/(\alpha + \theta))$')
    plt.tight_layout()
    title = ' '.join(hand_type.split('_')) + ' hand'
    plt.suptitle(title.capitalize(), y=1.0)
    plt.savefig(hand_type + '.pdf', dpi=300)
