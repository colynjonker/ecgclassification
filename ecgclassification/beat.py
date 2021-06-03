
import ecgclassification.config as config


# Heartbeat Class
class Beat:
    def __init__(self, ba, patient):
        self.ba = ba
        self.aami_num = ba_num(ba)
        self.patient = int(patient)


# Returns number presentation of BA class
def ba_aami(ba):
    if ba in set('NReLj'):
        # Normal class includes annotations N, R, L ,e, j
        return 'N'
    elif ba in set('AaSJ'):
        # SVEB class includes annotations: A, a, S, J
        return 'SVEB'
    elif ba in set('VE!'):
        # TODO: Determine if class ! should be included in VEB classes
        # VEB class includes annotations: V, E, {!}
        return 'VEB'
    elif ba == 'F':
        # Fusion class includes annotations: F
        return 'F'
    elif ba in set('/fQ'):
        # Unknown class includes annotations: /, f, q
        return 'Q'


# Returns number presentation of BA class
def ba_num(ba):
    if ba in set('NReLj'):
        # Normal class includes annotations N, R, L ,e, j
        return 0
    elif ba in set('AaSJ'):
        # SVEB class includes annotations: A, a, S, J
        return 1
    elif ba in set('VE!'):

        # VEB class includes annotations: V, E, {!}
        return 2
    elif ba == 'F':
        # Fusion class includes annotations: F
        return 3
    elif ba in set('/fQ'):
        # Unknown class includes annotations: /, f, q
        return 4
    else:
        return 5


