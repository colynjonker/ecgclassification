import numpy as np
import os
import wfdb

from datetime import datetime
from .beat import *


def format_signal(beat_signal, mid, extend_signal =True, half_window=config.hw):
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


def load_beats(filepath, patients=[], signal_format='ES'):
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

            anno_sample = [int(i*scale) for i in annotation.sample]

            # - 3
            for i in range(annotation.ann_len - 3 ):
                ba = annotation.symbol[i]
                print(ba)
                if ba in config.relsym:
                    beat = Beat(ba, record.record_name)
                    beat.start = end
                    end = int((anno_sample[i] + anno_sample[i+1]) / 2)
                    beat.end = end
                    if signal_format == 'OS':
                        beat.signal = signal[anno_sample[i] - config.hw:anno_sample[i] + config.hw]
                    else:
                        beat_signal = signal[beat.start:beat.end]
                        beat.mid = anno_sample[i] - beat.start
                        if signal_format == 'ES':
                            extend_signal = True
                        elif signal_format == 'IS':
                            extend_signal = None
                        elif signal_format == 'MS':
                            extend_signal = False
                        beat.signal = format_signal(beat_signal, beat.mid, extend_signal=extend_signal)
                    if beat.signal:
                        beats.append(beat)
    print('Loading beats took {}s'.format(round((datetime.now() - measure_time).total_seconds(), 1)))
    return beats


def load_data(filepath, classes='aami', patients=[], signal_format='ES'):
    if signal_format == 'OS':
        pass
    else:
        pass
        beats = load_beats(filepath, patients=patients, signal_format=signal_format)
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


