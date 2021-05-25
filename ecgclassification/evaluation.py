import matplotlib.pyplot as plt
import itertools

from sklearn.metrics import confusion_matrix
import datetime
from scipy import signal
import numpy as np

import ecgclassification
import ecgclassification.config as config
from scipy.signal import savgol_filter

aami = ["Normal", "Supraventricular", "Ventricular", "Fusion beat", "Unkown"]


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        pass
        # print('Confusion matrix, without normalization')
    # print(cm)
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# Calcuate VEB/SVEB metrics
def calculate_metrics(cm, class_type):
    if class_type == 'VEB':
        tn = sum(cm[0]) - cm[0][2] + sum(cm[1]) - cm[1][2] + sum(cm[3]) - cm[3][2] + sum(cm[4]) - cm[4][2]
        fn = sum(cm[2]) - cm[2][2]
        tp = cm[2][2]
        fp = cm[0][2] + cm[1][2]
    if class_type == 'SVEB':
        tn = sum(cm[0]) - cm[0][1] + sum(cm[2]) - cm[2][1] + sum(cm[3]) - cm[3][1] + sum(cm[4]) - cm[4][1]
        fn = sum(cm[1]) - cm[1][1]
        tp = cm[1][1]
        fp = cm[0][1] + cm[2][1] + cm[3][1]
    acc = round((tp + tn) / (tp + tn + fp + fn), 4) * 100
    pp = round(tp / (tp + fp), 4) * 100
    se = round(tp / (tp + fn), 4) * 100
    sp = round(tn / (tn + fp), 4) * 100
    return {class_type + ' Acc': acc, class_type + ' Pp': pp, class_type + ' Se': se, class_type + ' sp': sp}


def calculate_accuracy(cm):
    general_acc = round(sum(cm[i][i] for i in range(len(cm[0]))) / cm.sum(), 4) * 100
    return calculate_metrics(cm, 'VEB'), calculate_metrics(cm, 'SVEB'), general_acc


def evaluate_model(m, x_tests, y_tests, input_format="ba_num", keras_evaluation=True, normalize_cm=None):
    if keras_evaluation:
        m.evaluate(x_tests, y_tests)
    prediction = m.predict(x_tests, verbose=2)
    prediction = prediction.argmax(axis=-1)
    if input_format == "ba_num":
        ba_pred = prediction
        ba_true = y_tests
        aami_pred = [ecgc.aami_num(ecgc.config.relsym[ba]) for ba in ba_pred]
        aami_true = [ecgc.aami_num(ecgc.config.relsym[ba]) for ba in ba_true]
    elif input_format == "aami_num":
        aami_pred = prediction
        aami_true = y_tests

    cm = confusion_matrix(y_true=aami_true, y_pred=aami_pred, normalize=normalize_cm, labels=[0, 1, 2, 3, 4])
    print(cm)
    cm_plot_labels = ['N', 'S', 'V', 'F', 'Q']
    plot_confusion_matrix(cm=cm, classes=cm_plot_labels, title='Confusion Matrix for AAMI Classes')
    veb, sveb, gen = calculate_accuracy(cm)
    print("VEB acc: {} SVEB acc: {} General acc: {}".format(veb, sveb, gen))


def print_metrics(pred, true):
    cm = confusion_matrix(y_true=true, y_pred=pred)
    veb, sveb, gen = calculate_accuracy(cm)
    print("General Acc: {}")
    d = {**veb, **sveb}
    for k in sorted(d.keys()):
        print(k + " " + d[k])


def get_timestamp(i):
    return datetime.timedelta(seconds=round(i / config.signal_frequency))