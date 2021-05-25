import datetime

import wfdb
import os

from collections import Counter
from .beat import *
import ecgclassification.config as config
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter


# Counts the total occurrence of each class in the defined dataset.
def count_total_occurrences(occurrences):
    total = {}
    for k, v in occurrences.items():
        for key, value in v.items():
            if key not in total:
                total[key] = value
            else:
                total[key] += value
    return total


# Count the occurrence of classes for each record
def count_occurrences(filepath, patients=[]):
    total = {}
    for file in os.listdir(filepath):
        if file.endswith('atr'):
            record = wfdb.rdrecord(filepath + file.split('.')[0])

            if patients and not int(record.record_name) in patients:
                continue
            annotation = wfdb.rdann(filepath + record.record_name, 'atr')

            occurrence = Counter(annotation.symbol)
            total[record.record_name] = {key: val for key, val in occurrence.items() if key in config.relsym}
    return total


def count_aami(total):
    aami_sorted = {}
    for k, v in total.items():
        aami_class = ba_aami(k)
        if aami_class not in aami_sorted:
            aami_sorted[aami_class] = v
        else:
            aami_sorted[aami_class] += v
    return aami_sorted


def print_table(d):
    print("{:<8} {}".format('Patient', 'Counters'))
    for k, v in d.items():
        print("{:<8} {}".format(k, v))


def pretty_print(d):
    print("{:<8} {:<15}".format('Key', 'Value'))
    for k, v in d.items():
        print("{:<8} {:<15}".format(k, v))


def get_timestamp(i):
    return datetime.timedelta(seconds=round(i / config.signal_frequency))


def visualize_beat(beat, title=None, color='tab:green', visualise_channels=False):
    x = [i / config.signal_frequency for i in range(config.window)]
    if visualise_channels:
        x = [i / config.signal_frequency for i in range(config.window)]
        fig, ax = plt.subplots(nrows=2, ncols=2)
        signal = beat.signal
        ax[0, 0].title.set_text('Extended signal ({})'.format(title))
        ax[0, 0].plot(x, signal, color='tab:red')
        sg = savgol_filter(beat.signal, 25, 2)
        ax[0, 1].title.set_text('savgol(red)')
        ax[0, 1].plot(x, sg, color='tab:orange')
        blue = [abs(i) for i in (signal - sg)]
        blue = savgol_filter(blue, 101, 2)
        ax[1, 0].set_ylim(0.01, 0.05)
        ax[1, 0].title.set_text('abs(red - orange)')
        ax[1, 0].plot(x, blue, color='tab:green')
        orange = savgol_filter(blue, 101, 2)
        ax[1, 1].title.set_text('savgol(green)')
        ax[1, 1].plot(x, orange, color='tab:blue')
        plt.tight_layout()
        plt.show()
    else:
        fig, ax = plt.subplots()
        if title is None:
            title = "{} -- {} -- {} ".format(beat.ba, beat.patient, get_timestamp(beat.start))
        plt.title(title, fontsize=20)
        plt.locator_params(axis='y')
        ax.plot(x, beat.signal, '-D', markevery=[int(config.window / 2)], mfc='b', color=color)
        ax.set_xlabel('Time in s', fontsize=18)
        ax.set_ylabel('Voltage in mV', fontsize=18)
        plt.show()



#

