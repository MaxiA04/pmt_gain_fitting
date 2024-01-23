import warnings
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from pmt_gain_fitting.FitterClass.Toolkit import Params, adc_to_e
from pmt_gain_fitting.FitterClass.trim_outliers import trim_percentile
from pmt_gain_fitting.FitterClass_deprecated.EMG_Model_np import total_charge_yield, extended_normal, extended_en



"""
Fitter for gain calculation of PMT R12699-406-M4 callibration data.
- fig_path:              Path to directory where you want the plots to be saved
- bias_voltage='1000V':                variable name
- upper_cut:             Upper boundary of fitting range (in initial ADC units, typically ranges from O(10²) - O(10³)
                         Call the 
Additional (optioonal) arguments:
- lamb, mu0, ..., w:     Initial guesses of paramaters
- wf:                    In case of simultaneous readout, i.e. single root file containing all PMT readouts, you might want to add
                            a specification to the file handle to identify the PMT. The ID should be passed to the object via a string
                            stored in the wf variable.
- lower_cut:             Lower boundary of fitting range (scaled to be between [lower_cut, 35]
- single_channel:        Default True. Indicates
"""


class MasterFitter(object):

    def __init__(self, array, fig_path, pmt_ID, trim=1e-3, bias_voltage='1000', lamb=.05,
                 mu0=.5, sig0=.4, mu=14, sig=4, c=.85, w=.1, suffix='', pretty_plot=True):
        self.array = array
        self.fig_path = fig_path
        self.bias_voltage = bias_voltage
        self.pmt_ID = pmt_ID
        self.suffix = suffix
        self.df_header = ['occupancy', 'occupancy_err', 'q0', 'q0_err', 'sig0', 'sig0_err', 'gain', 'gain_err',
                          'sig_gain', 'sig_gain_err', 'c', 'c_err', 'w', 'w_err']
        self.pretty_plot = pretty_plot
        ###########################
        # Adjust Parameters here! #
        ###########################

        # Call the Waveform method to estimate thresholds and enter them here
        thresholds = (140, 250)
        scaling_ceiling = 35
        self.bounds = np.array([[0.001, -3, 0, 5, 1, 1e-4, 1e-3],
                                [5, 4, 3, 30, 5, 1, 1]])
        self.nbins = 80

        # Initialize Parameters and scaling

        self.df = Params(self.array, thresholds)
        self.params = [lamb, mu0, sig0, mu, sig, c, w]
        self.param_names = ['lambda', 'mu0', 'sig0', 'mu', 'sig', 'c', 'w']
        self.trimmed_area = trim_percentile(self.df.area, 100 - trim)
        self.hist_bins = np.histogram(self.trimmed_area, bins=self.nbins)

        # Initialize Histogram

        self.hist = self.hist_bins[0]
        _bin_centers = (self.hist_bins[1][:-1] + self.hist_bins[1][1:]) / 2
        self.upper_cut = _bin_centers[-1]
        self.scaling_factor = scaling_ceiling / self.upper_cut
        self.bin_centers = self.scaling_factor * _bin_centers
        self.cut_off_range = (self.bin_centers[0], scaling_ceiling)
        self.bin_width = (self.bin_centers[1] - self.bin_centers[0])
        self.normalization = (sum(self.hist) * self.bin_width)
        self.normed_hist = self.hist / self.normalization

    def Waveforms(self, iterations, thresholds=(0, 0)):
        """
        Aid to visualize positions of signals in PMT samples.
        Plots events where the voltage dips below a set value of adc counts
        -iterations: Nr. of waveforms starting from the top to be evaluated.
        -thresholds: in [0,250]. Adds vertical lines to the plot. Use it to visualize peak/valley position
        """
        adc_threshold = 15480

        peaks_detected = 0
        xplot = np.linspace(0, np.shape(self.array)[1], np.shape(self.array)[1])

        for i in range(iterations):
            if np.amin(self.array[i]) <= adc_threshold:
                plt.plot(xplot, self.array[i])
                peaks_detected += 1
        if thresholds != (0, 0):
            plt.axvline(thresholds[0], linestyle='--', color='r')
            plt.axvline(thresholds[1], linestyle='--', color='r')
        print('Out of %s waveforms %s peaks where detected' % (iterations, peaks_detected))
        plt.title(f'First {iterations} WFs of ' + self.pmt_ID + '@' + self.bias_voltage)
        plt.savefig(self.fig_path + self.bias_voltage + 'waveforms' + self.pmt_ID + '.pdf')
        plt.show()
        plt.clf()

        return 0

    def histogram(self):
        """
        Aid to visualize the range of the area under/over the signal peak.
        Call this function to decide which range to set in the histogramming of the data.

        Ex.: Set cut at 4500 -> adjust variable scaling_factor to cut_off_range[1]/4500
        :return figure:
        """

        hist, bins = np.histogram(self.df.area, self.nbins, range=(-500, 2e4))
        bin_centers = (bins[:-1] + bins[1:]) / 2
        yerr = (hist + 1) ** .5
        plt.errorbar(bin_centers, hist, yerr=yerr, fmt='.', color='k')
        plt.yscale('log')
        plt.title(self.bias_voltage + ' ' + self.pmt_ID)
        plt.show()
        return 0

    def optimizer(self):
        popt, pcov = curve_fit(total_charge_yield, self.bin_centers, self.normed_hist, p0=self.params,
                               bounds=self.bounds, sigma=(self.normed_hist + 1) ** .5,
                               absolute_sigma=False)
        print(self.param_names, '\n', popt)
        if np.abs(self.bounds[0] - popt).min() <= 1e-3 or np.abs(self.bounds[0] - popt).min() <= 1e-3:
            index = np.abs(self.bounds[0] - popt).argmin()
            warnings.warn(
                f'Optimal Value near Boundary! \n {self.param_names[index]}' + f' = {popt[index]}' + f'\n boundaries: {self.bounds[0][index]}, {self.bounds[1][index]}')

        return popt, pcov

    def residuals(self, hist, bin_centers, popt):

        residuals = (hist - self.normalization * total_charge_yield(bin_centers, *popt)) / (1 + hist) ** .5
        return residuals

    def plot(self, nPE):
        popt, pcov = self.optimizer()
        occupancy = popt[0]
        occ_err = pcov[0][0] ** .5
        q0 = popt[1] / self.scaling_factor * adc_to_e
        q0_err = pcov[1][1] / self.scaling_factor * adc_to_e
        sig0 = popt[2] / self.scaling_factor * adc_to_e
        sig0_err = pcov[2][2] / self.scaling_factor * adc_to_e
        gain = popt[3] / self.scaling_factor * adc_to_e
        gain_err = pcov[3][3] ** .5 / self.scaling_factor * adc_to_e
        sig_gain = popt[4] / self.scaling_factor * adc_to_e
        sig_gain_err = pcov[4][4] ** .5 / self.scaling_factor * adc_to_e
        c = popt[5]
        c_err = pcov[5][5] ** .5 / self.scaling_factor * adc_to_e
        w = popt[6]
        w_err = pcov[6][6] ** .5 / self.scaling_factor * adc_to_e

        scaled_params = np.array([occupancy, occ_err, q0, q0_err, sig0, sig0_err, gain, gain_err,
                                  sig_gain, sig_gain_err, c, c_err, w, w_err])

        xplot = np.linspace(self.cut_off_range[0], 35, 500)

        fig, ax = plt.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': [4, 1]})

        residuals = self.residuals(self.hist, self.bin_centers, popt)

        ax[0].errorbar(self.bin_centers, self.hist, xerr=self.bin_width, yerr=self.hist ** .5, fmt='.', color='k')
        ax[0].plot(xplot, self.normalization * total_charge_yield(xplot, *popt),
                   label='Fit Function')
        ax[1].scatter(self.bin_centers, residuals, color='k', marker='.')

        for i in range(nPE + 1):
            ax[0].plot(xplot, self.normalization * extended_normal(xplot, *popt, i), linestyle='--',
                       label='%s PE Normal' % i)
            ax[0].plot(xplot, self.normalization * extended_en(xplot, *popt, i), linestyle='--',
                       label='%s PE EMG' % i)

        xticks = np.arange(0, 36, 5)
        res_ticks = [-3, 0, 3]
        labels = np.asarray(np.round(adc_to_e * xticks / self.scaling_factor / 1e6, 2), dtype=float)
        plt.xticks(ticks=xticks, labels=labels)
        ax[0].set_yscale('log')
        ax[1].set_xlabel('Signal Electrons (x 1e6)')
        ax[0].set_ylabel('Counts')
        ax[1].set_ylabel(r'$\sigma$')
        ax[0].set_ylim(1, 1e6)
        ax[0].legend(loc='upper right')
        ax[1].set_ylim(-5, 5)
        plt.axhline(-3, color='r', linestyle='--')
        plt.axhline(-1, color='g', linestyle='--')
        plt.axhline(3, color='r', linestyle='--')
        plt.axhline(1, color='g', linestyle='--')
        plt.axhline(0, color='k', linestyle='--')
        plt.yticks(ticks=res_ticks)
        fig.suptitle('PE Spectrum at ' + self.bias_voltage + 'V' + ' [%s]' % self.pmt_ID)
        if self.pretty_plot:
            plt.savefig(self.fig_path + self.pmt_ID + '_' + self.bias_voltage + 'V_fit' + self.suffix + '.pdf')
        plt.savefig(self.fig_path + self.pmt_ID + '_' + self.bias_voltage + 'V_fit' + self.suffix + '.png')
        plt.clf()
        plt.close()

        print('Gain = %s +- %s 10⁶' % (np.round(gain / 1e6, 3), np.round(gain_err / 1e6, 3)))
        print('Occupancy lambda = %s +- %s' % (occupancy, occ_err))
        return scaled_params
