import wfdb
import os
import operator
from collections import Counter
import ecgclassification.config as config


# Counts the occurrence of annotations in each record in the folder
def count_total_occurrences(occurrences):
    total = {}
    for k, v in occurrences.items():
        for key, value in v.items():
            if key not in total:
                total[key] = value
            else:
                total[key] += value
    return total


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


def print_table(d):
    print("{:<8} {}".format('Patient', 'Counters'))
    for k, v in d.items():
        print("{:<8} {}".format(k, v))


def pretty_print(d):
    print("{:<8} {:<15}".format('Key', 'Value'))
    for k, v in d.items():
        print("{:<8} {:<15}".format(k, v))

