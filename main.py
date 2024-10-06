import os

from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import pandas as pd
from obspy import read
from obspy.signal.invsim import cosine_taper
from obspy.signal.filter import highpass
from obspy.signal.trigger import classic_sta_lta, plot_trigger, trigger_onset
from scipy import signal
from matplotlib.colors import LogNorm
from scipy.signal import butter, filtfilt

OUTPUT_FILES_ROOT = './output_files/'
PATH_ROOT = './data/'  # 'C:/Users/dsavr/.pyweed/' #
RELATIVE_ROOT = 'lunar/test/data/'
PLOT_GRIDS = True


# FILTERS #####################################


def bandpass_filter(data, lowcut, highcut, fs, order=3):
    """Apply a band-pass filter to the data samples"""
    nyquist = 0.5 * fs  # Nyquist frequency
    low = lowcut / nyquist  # Normalize the lowcut frequency
    high = highcut / nyquist  # Normalize the highcut frequency
    b, a = butter(order, [low, high], btype='band', analog=False)  # Filter coefficients
    filtered_data = filtfilt(b, a, data)
    return filtered_data


def lowpass_filter(data, cutoff, fs, order=3):
    """Apply a low-pass filter to the data samples"""
    nyquist = 0.5 * fs  # Nyquist frequency
    cutoffn = cutoff/nyquist
    b, a = butter(order, cutoffn, btype='lowpass', analog=False)
    filtered_data = filtfilt(b, a, data)
    return filtered_data


class Model:
    def __init__(self):
        self.raw_data = None
        self.data = None
        self.raw_times = None
        self.times = None
        self.fname = None
        self.fs = None
        self._fig = None
        self._atimes = self.get_cat_arrivals()
        self.atime = None
        # SPECTROGRAM PARAMS
        self.sxx = None
        self.sxxt = None
        self.sxxf = None
        self.sc = None
        self.scm = None
        #  CHARACTERISTIC FUNCTION PARAMS
        # How long should the short-term and long-term window be, in seconds?
        self.sta_len = 120
        self.lta_len = 600
        self.ch_samps = None
        self.ch_times = None

    @staticmethod
    def get_cat_arrivals():
        """Get map filename to arrival time, from the training catalogs"""
        arrivals = {}
        cat = pd.read_csv('./data/lunar/training/catalogs/apollo12_catalog_GradeA_final.csv')
        moon_filename_to_arrival = {cat.iloc[i].filename: cat.iloc[i]['time_rel(sec)'] for i in range(len(cat))}
        arrivals.update(moon_filename_to_arrival)
        cat = pd.read_csv('./data/mars/training/catalogs/Mars_InSight_training_catalog_final.csv')
        mars_filename_to_arrival = {cat.iloc[i].filename.replace('.csv', ''): cat.iloc[i]['time_rel(sec)'] for i in
                                    range(len(cat))}
        arrivals.update(mars_filename_to_arrival)
        return arrivals

    def load_from_seed(self, seed_path, fname):
        st = read(seed_path+fname)
        self.fname = fname.replace('.mseed', '')
        self.raw_times = st.traces[0].times()
        self.times = self.raw_times
        self.raw_data = st.traces[0].data
        self.data = self.raw_data
        self.fs = 1 / (self.times[1] - self.times[0])
        self.atime = self._atimes.get(self.fname)
        self.analyze_capture()

    def analyze_capture(self):
        # DATA FILTERING
        self.data = bandpass_filter(self.data, 0.3, 1, self.fs)
        self.data = savgol_filter(self.data, window_length=2001, polyorder=2)
        self.get_spectrogram_analysis()
        #self.get_characteristic()

    def plot_capture(self):
        self._fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 1, figsize=(19.2, 10.8))
        self._fig.suptitle(self.fname, fontweight='bold')
        #self.plot_data(ax1, self.times, self.raw_data, f'Raw data, fs={round(self.fs, 2)}Hz')
        self.plot_data(ax1, self.times, self.data, f'Raw data, filtered')
        # PLOT ENVELOPE
        envelope = lowpass_filter(5 * np.abs(self.data), 0.01, self.fs)
        self.plot_data(ax1, self.times, envelope, 'Data envelope')
        # PLOT FFT
        self.plot_fft(ax2, self.times, self.data)
        # PLOT SPECTROGRAM
        self.plot_spectrogram(ax3)
        self.plot_spectral_content(ax4)
        if self.atime:
            self.plot_arrival(ax3, self.atime)
            self.plot_arrival(ax4, self.atime)
            self.plot_arrival(ax1, self.atime)
        self.plot_characteristic(ax5)
        plt.subplots_adjust(hspace=0.5)

    @staticmethod
    def plot_fft(ax, times, data):
        """Plot the FFT on the axis"""
        sampling_period = (times[1] - times[0])
        n = len(times)
        fft_values = np.fft.fft(data)
        magnitude = np.abs(fft_values) / n
        freqs = np.fft.fftfreq(n, d=sampling_period)
        # PLOT ONLY THE POSITIVE F
        positive_magnitude = magnitude[:n // 2]
        positive_freqs = freqs[:n // 2]
        ax.semilogy(positive_freqs, positive_magnitude, label='Fast Fourier Transform')
        ax.set_xlabel('Frequency [Hz]', fontweight='bold')
        ax.set_ylabel('Amplitude', fontweight='bold')
        ax.set_title(f'Frequency content of raw data', fontweight='bold')
        if PLOT_GRIDS:
            ax.grid()
        ax.legend()

    def get_spectrogram_analysis(self):
        self.sxxf, self.sxxt, self.sxx = signal.spectrogram(self.data, self.fs)
        self.sc = [sum([self.sxx[i][j] for i in range(len(self.sxx))]) for j in range(len(self.sxx[0]))]
        # MEDIAN FILTER
        self.scm = signal.medfilt(self.sc, kernel_size=11)

    def plot_spectrogram(self, ax):
        color_range = LogNorm(vmin=np.max(self.sxx) / 1000, vmax=np.max(self.sxx))
        vals = ax.pcolormesh(self.sxxt, self.sxxf, self.sxx, cmap=cm.jet, norm=color_range)
        ax.set_xlim([min(self.times), max(self.times)])
        ax.set_xlabel(f'Time [s]', fontweight='bold')
        ax.set_ylabel('Frequency [Hz]', fontweight='bold')
        cbar = plt.colorbar(vals, orientation='horizontal')
        cbar.set_label('Power ((m/s)^2/sqrt(Hz))', fontweight='bold')
        ax.set_title(f'Spectrogram', fontweight='bold')

    def plot_spectral_content(self, ax):
        ax.plot(self.sxxt, self.sc, label='Raw', color='red')
        # ax.plot(self.sxxt, lowpass_filter(self.sc, 0.1, 2), label='Filtered, low pass filter')
        # Apply Savitzky-Golay filter
        # ax.plot(self.sxxt, savgol_filter(self.sc, window_length=11, polyorder=3), label='Filtered. Golay')
        ax.plot(self.sxxt, self.scm, label='Filtered, median')
        ax.legend()
        ax.set_xlim([min(self.times), max(self.times)])
        ax.set_xlabel(f'Time [s]', fontweight='bold')
        ax.set_ylabel('Spectral content', fontweight='bold')
        ax.set_title(f'Spectral content', fontweight='bold')
        if PLOT_GRIDS:
            ax.grid()

    @staticmethod
    def plot_data(ax, times, data, label):
        ax.plot(times, data, label=label)
        ax.set_xlim([min(times), max(times)])
        ax.set_ylabel('Velocity (m/s)', fontweight='bold')
        ax.set_xlabel('Time [s]', fontweight='bold')
        ax.set_title(f'Raw data plot', fontweight='bold')
        if PLOT_GRIDS:
            ax.grid()
        ax.legend()

    def get_characteristic(self, data, freq):
        # Run Obspy's STA/LTA to obtain a characteristic function
        # This function basically calculates the ratio of amplitude between the␣short - term
        # and long-term windows, moving consecutively in time across the data
        # self.ch_samps = classic_sta_lta(data, int(self.sta_len * freq), int(self.lta_len * freq))
        self.ch_samps = self.sta_lta(data, int(self.sta_len * freq), int(self.lta_len * freq))
        self.ch_times = np.arange(len(self.ch_samps))/freq
        return self.ch_samps, self.ch_times

    def sta_lta(self, signal, nsta, nlta, demean=True):
        """
        Compute the STA/LTA ratio of a signal.

        Parameters:
        - signal (np.ndarray): Input seismic signal.
        - nsta (int): Number of samples in the short-term average window.
        - nlta (int): Number of samples in the long-term average window.
        - demean (bool): Whether to remove the mean from the signal before processing.

        Returns:
        - ratio (np.ndarray): STA/LTA ratio.
        """
        from scipy.signal import convolve

        if len(signal) < nlta:
            raise ValueError("Signal length must be at least as long as nlta")

        # Optional: Remove mean to center the signal around zero
        if demean:
            signal = signal - np.mean(signal)

        # Use absolute signal to ensure non-negative values
        abs_signal = np.abs(signal)

        # Define the window for STA and LTA
        window_sta = np.ones(nsta)
        window_lta = np.ones(nlta)

        # Compute STA and LTA using convolution
        sta = convolve(abs_signal, window_sta, mode='same') / nsta
        lta = convolve(abs_signal, window_lta, mode='same') / nlta

        # Avoid division by zero by adding a small epsilon to LTA
        epsilon = 1e-10
        ratio = sta / (lta + epsilon)

        return ratio

    def plot_characteristic(self, ax):
        # Plot characteristic function
        ax.plot(self.ch_times, self.ch_samps/np.mean(self.ch_samps), label='Characteristic function')
        ax.set_xlim([min(self.ch_times), max(self.ch_times)])
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Characteristic function', fontweight='bold')
        ax.grid()
        ax.legend()

    def plot_arrival(self, ax, label):
        ax.axvline(x=self.atime, c='red')
        ax.legend()

    def save_image(self, comment=None):
        if self._fig:
            image_path = OUTPUT_FILES_ROOT + RELATIVE_ROOT + self.fname + comment + '.png'
            print(f'Saving image {image_path}')
            if not os.path.exists(OUTPUT_FILES_ROOT + RELATIVE_ROOT):
                os.makedirs(OUTPUT_FILES_ROOT + RELATIVE_ROOT)
            plt.savefig(image_path)
            plt.close(self._fig)
            self._fig = None


if __name__ == '__main__':
    model = Model()
    for file in os.listdir(PATH_ROOT + RELATIVE_ROOT):
        if file.endswith('.mseed'):
            model.load_from_seed(PATH_ROOT + RELATIVE_ROOT, file)
            """for sta in np.linspace(100, 1000, 50):
                for lta in np.linspace(100, 2000, 50):"""
            model.sta_len = 200
            model.lta_len = 1000
            model.get_characteristic(model.scm, 2555 / (model.times[-1] - model.times[0]))
            model.plot_capture()
            model.save_image(f'sta={400}, lta={3000}')

