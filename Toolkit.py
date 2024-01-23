import numpy as np
import uproot
import pandas as pd


def store_waveforms(path, filename, tree,
                     HV_scan = False, branch='wf0', folder = ''):  # Branches in .root-files tend to differ. look them up in the TBrowser
    # and enter them as touple of strings here
    root_file = uproot.open(path + filename + '.root' + tree)
    array = root_file[branch].array(library='np')

    if HV_scan == False:
        np.save('/home/maxi/Documents/UZH/MasterOfScience!/Code/Project/LightLevels/Arrays/' + filename, array)
    else:
        if folder == '':
            raise Exception("Specify Directory for storage!")
        np.save(folder+'/'+ filename + '_' +branch, array)
        print('done')
    return array


def Params(array,
           thresholds):  # return df with added collumns bsl (baseline), amplitude (baseline subtracted min value of peak*-1)
    # area ( sum of bin entries between thresholds)
    # Thresholds = (lower_threshold, upper_threshold) (touple)

    df = pd.DataFrame(array)
    bsl = np.median(array[:, :thresholds[0]], axis=1)
    df['bsl'] = bsl
    bsl_std = np.std(array[:, :thresholds[0]], axis=1, ddof=1)
    df["bsl_std"] = bsl_std

    amplitude = -(np.min(array[:, thresholds[0]:thresholds[1]], axis=1) - bsl)
    df['amplitude'] = amplitude

    area = -(np.sum(array[:, thresholds[0]: thresholds[1]], axis=1) - df.bsl * (thresholds[1] - thresholds[0]))
    df['area'] = area

    return df

def gain_HV(HV, a, k):
    # number of dinodes n
    n = 10
    return a ** n * (HV / (n + 1)) ** (k * n)


# Conversion Factors

F = 500 * 10 ** 6  # sampling  frequency: 500 MHz digitization speed (2 ns bins)
r = 2.0 / 2 ** 14  # ADC resolution: 14 bit ADC, 2V voltage range -> already accounted for by DAQ
Z = 50  # input  impedance: 50 Ohm termination to ground
e = 1.60218 * 10 ** (-19)  # electron charge
A = 10  # amplification factor: 10 times gain_ch0
adc_to_e = r / (F * Z * e * A)
