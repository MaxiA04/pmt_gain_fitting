import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfcx, erfc


x = np.linspace(-3.5, 45,550)
# x = np.linspace(-5e5, 1e7, 550)


def normal(x, mu, sigma):
    return 1/(2*np.pi*sigma**2)**.5*np.exp(-.5*(x-mu)**2/sigma**2)

def emg(x, mu, sigma, c):

    return c/2*np.exp(c**2*sigma**2/2)*np.exp(-c*(x-mu))*erfc(1/2**.5*(c*sigma - (x -mu)/sigma))

# Apparently more robust parametrization (holds up for large values of c but breaks down as x grows)

def EN(x,mu,sigma,c):
    part1 = np.exp(-1/2*((x-mu)/sigma)**2)
    part2 = sigma*c*np.sqrt(np.pi/2)*erfcx(1/np.sqrt(2)*(sigma*c-((x-mu)/sigma)))
    return part1*part2

def total_charge_yield(x, lamb, mu0, sig0, mu, sig, c, w):
    n = 20

    term1 = np.exp(-lamb)*((1-w)*normal(x, mu0, sig0) + w*emg(x, mu0, sig0, c))
    term2 = lamb*np.exp(-lamb) * ((1-w)*normal(x, mu + mu0, np.sqrt(sig**2 + sig0**2))
                                  + w*emg(x, mu + mu0, np.sqrt(sig**2 + sig0**2), c))
    term3 = 0

    sum_over = np.arange(2,20,1)

    for i in sum_over:
        term3 += lamb**i*np.exp(-lamb)/np.math.factorial(i)*((1-w)*normal(x, i*mu+mu0, np.sqrt(i*sig**2 + sig0**2))
                                                             +w*emg(x, i*mu+mu0, np.sqrt(i*sig**2 + sig0**2), c))
    return term1 + term2 + term3

def extended_normal(x, lamb, mu0, sig0, mu, sig, c, w, k):
    return lamb**k*np.exp(-lamb)/np.math.factorial(k)*(1-w)*normal(x,k*mu + mu0,(k*sig**2 + sig0**2)**.5)

def extended_en(x, lamb, mu0, sig0, mu, sig, c, w, k):
    return lamb**k*np.exp(-lamb)/np.math.factorial(k)*w*(emg(x, k*mu+mu0, (k*sig**2+sig0**2)**.5, c))

# params = [lambda, q0, sig0, mu, sig, c, w]
params = [.05, 0.1, .6, 14, 5., 1., .05]
# params = [.8, 0.1, .4, .2 ,10., .05, 0.01]
# paramparm = [3., 100., 50., 3.e6, 3.e4, .2,.3]

# params = [2.4, 1e5, 1e4, 3e6, 1e6, 1., .5]
# plt.plot(x, emg(x, .01, 1, 3, 2))
# plt.title(fr'$\lambda$ = {params[0]}, $\mu$ = {params[3]}, $\sigma$ = {params[4]}')
# plt.plot(x, total_charge_yield(x, *params), color='k', label='Total PDF')
# for i in range(3):
#     if i == 0:
#         label_N = 'Noise'
#     else:
#         label_N = f'{i}PE Normal'
#     label_EMG = f'{i}PE EMG'
#     plt.plot(x, extended_normal(x,*params, i), linestyle='--', label=label_N)
#     plt.plot(x, extended_en(x, *params, i), linestyle='--', label=label_EMG)
# plt.xlabel('Charge Yield [A.U.]')
# plt.ylabel(r'Prob. Density')
# plt.legend()
# plt.ylim(1e-5, 1)
# plt.yscale('log')
# plt.show()
