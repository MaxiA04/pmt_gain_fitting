import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from scipy.special import erfc
import zfit

x = np.linspace(-2, 30, 550)

def normal(x, mu, sigma):
    return 1/(2*np.pi*sigma**2)**.5*zfit.z.exp(-.5*(x-mu)**2/sigma**2)

def emg(x, bsl_shift, bsl_width, c):
    exponential_part = c / 2 * zfit.z.exp(c ** 2 * bsl_width ** 2 / 2) * zfit.z.exp(-c * (x - bsl_shift))
    prod = tf.math.erfc(1 / 2 ** .5 * (c * bsl_width - (x - bsl_shift) / bsl_width))
    return exponential_part*prod

def total_charge_yield(x, lamb, mu0, sig0, mu, sig, c, w):
    n = 20

    term1 = zfit.z.exp(-lamb)*((1-w)*normal(x, mu0, sig0) + w*emg(x, mu0, sig0, c))
    term2 = lamb*zfit.z.exp(-lamb) * ((1-w)*normal(x, mu + mu0, zfit.z.sqrt(sig**2 + sig0**2))
                                  + w*emg(x, mu + mu0, zfit.z.sqrt(sig**2 + sig0**2), c))
    term3 = 0

    sum_over = np.arange(2,20,1.)
    print('hi')

    for i in sum_over:
        term3 += lamb**i*zfit.z.exp(-lamb)/np.math.factorial(i)*((1-w)*normal(x, i*mu+mu0, zfit.z.sqrt(i*sig**2 + sig0**2))
                                                             +w*emg(x, i*mu+mu0, zfit.z.sqrt(i*sig**2 + sig0**2), c))
    return term1 + term2 + term3

params = [3., 1., .5, 5., 2., .2, .3]

plt.plot(x, total_charge_yield(x, *params))
plt.yscale('log')
print('hello')
plt.show()
