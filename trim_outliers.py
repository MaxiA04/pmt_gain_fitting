import numpy as np
import tqdm
from itertools import count

def trim(array):
    mean = sum(array)/len(array)
    variance =  sum((array - mean)**2)/len(array)
    trimmed_array = []
    threshold = 20*variance**.5
    for entry in array:
        if entry >= threshold or entry <= -.1*threshold:
            pass
        else:
            trimmed_array.append(entry)

    return np.array(trimmed_array)

def trim_percentile(area, percentile):
    upper_lim = np.percentile(area, percentile)
    lower_lim = -500
    trimmed_array = area[ (area <= upper_lim) & (area >= lower_lim)]
    print(lower_lim, upper_lim)
    return trimmed_array