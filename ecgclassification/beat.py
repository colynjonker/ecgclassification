
import ecgclassification.config as config


# Heartbeat Class
class Beat:
    def __init__(self, ba, patient):
        self.ba = ba
        self.aami_num = aami_to_num(ba)
        self.patient = int(patient)


# Returns number representation of BA class
def ba_num(ba):
    return config.relsym.index(ba)


# Returns number presentation of BA class
def aami_to_num(ba):
    if ba in set('NRelj'):
        return 0
    elif ba in set('AaSJ'):
        return 1
    elif ba in set('V!E'):
        return 2
    elif ba == 'F':
        return 3
    elif ba in set('/fQ'):
        return 4
    else:
        return 5
