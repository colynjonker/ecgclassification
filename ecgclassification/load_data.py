import numpy as np
import os
import wfdb

from datetime import datetime

from scipy.signal import resample

from .beat import *


def format_signal(beat_signal, mid, extend_signal=True, half_window=config.hw):
    signal = list(beat_signal)
    if not signal:
        return None
    if extend_signal is None:
        sel, ser = None, None
    elif extend_signal:
        sel, ser = signal[0], signal[-1]
    else:
        sel, ser = min(signal), min(signal)
    start = mid - half_window
    if start >= 0:
        left = signal[start:mid]
    else:
        left = [sel] * (start * -1)
        left.extend(signal[:mid])
    right = signal[mid:mid + half_window]
    if len(right) != half_window:
        n = [ser] * (half_window - len(right))
        right.extend(n)
    left.extend(right)
    if len(left) == half_window * 2:
        return left
    return None


def load_beats(filepath, patients=[]):
    beats, measure_time, end = [], datetime.now(), 0
    for file in os.listdir(filepath):
        if file.endswith('.atr'):
            record = wfdb.rdrecord(filepath + file.split('.')[0])

            # Skip if patient is not in specified patterns
            if patients and not int(record.record_name in patients):
                continue

            annotation = wfdb.rdann(filepath + record.record_name, 'atr')
            signal = np.array([i[0] for i in record.p_signal])
            scale = config.signal_frequency / 360
            signal = resample(signal, int(scale*len(signal)))

            anno_sample = [int(i * scale) for i in annotation.sample]

            # - 3
            for i in range(annotation.ann_len - 3):
                ba = annotation.symbol[i]
                if ba in config.relsym:
                    beat = Beat(ba, record.record_name)
                    beat.start = end
                    end = int((anno_sample[i] + anno_sample[i + 1]) / 2)
                    beat.end = end
                    beat_signal = signal[beat.start:beat.end]
                    beat.mid = anno_sample[i] - beat.start
                    beat.signal = format_signal(beat_signal, beat.mid)
                    #beat.signal = beat_signal
                    if beat.signal:
                        beats.append(beat)
    print('Loading beats took {}s'.format(round((datetime.now() - measure_time).total_seconds(), 1)))
    return beats


def load_data(filepath, classes='aami', patients=[]):
    beats = load_beats(filepath, patients=patients)
    measure_time = datetime.now()
    data = {'train': [[], []], 'valid': [[], []], 'tests': [[], []]}
    for b in beats:
        if b.patient in config.train:
            s = 'train'
        elif b.patient in config.valid:
            s = 'valid'
        elif b.patient in config.tests:
            s = 'tests'
        else:
            continue

        if classes == 'aami':
            label = b.aami_num
        else:
            label = ba_num(b.ba)
        data[s][0].append(b.signal)
        data[s][1].append(label)
    for k in data:
        data[k][0] = np.asarray(data[k][0])
        data[k][1] = np.array(data[k][1])
    print('Loading the signals took: {}s'.format(round((datetime.now() - measure_time).total_seconds(), 1)))
    return data


def load_data_v2(filepath, patients=[], classes='aami'):
    beats = load_beats(filepath, patients=patients)
    measure_time = datetime.now()
    data = {'data': [[], []]}
    for b in beats:
        s = 'data'

        if classes == 'aami':
            label = b.aami_num
        else:
            label = ba_num(b.ba)
        data[s][0].append(b.signal)
        data[s][1].append(label)
    for k in data:
        data[k][0] = np.asarray(data[k][0])
        data[k][1] = np.array(data[k][1])
    print('Loading the signals took: {}s'.format(round((datetime.now() - measure_time).total_seconds(), 1)))
    return data

