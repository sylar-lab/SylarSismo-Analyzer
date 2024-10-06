import os

from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import pandas as pd
from obspy import read
from scipy import signal
from matplotlib.colors import LogNorm
from scipy.signal import butter, filtfilt

PLANET = 'lunar'  # 'moon' or 'mars'
MODE = 'test'  # 'training' or 'test'

MOD_PARAMS = {'mars': {'filter_range': [0.3, 1.5],
                       'median_len': 7,
                       'ch_cutoff': 1e-3,
                       'ch_trigger': 0.5},
              'lunar': {'filter_range': [0.5, 0.9],
                        'median_len': 21,
                        'ch_cutoff': 1.4e-3,
                        'ch_trigger': 0.46}
              }

OUTPUT_FILES_ROOT = './output_files/'
PATH_ROOT = './data/'  # 'C:/Users/dsavr/.pyweed/' #
RELATIVE_ROOT = f'{PLANET}/{MODE}/data/'
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
    cutoffn = cutoff / nyquist
    b, a = butter(order, cutoffn, btype='lowpass', analog=False)
    filtered_data = filtfilt(b, a, data)
    return filtered_data


class Model:
    def __init__(self):
        self.fname = None
        self.raw_data = None
        self.data = None
        self.raw_times = None
        self.times = None
        self.fs = None
        self._fig = None
        self._atimes = self.get_arrivals_from_catalog()
        self.cat_atime = None
        # SPECTROGRAM PARAMS
        self.sxx = None
        self.sxxt = None
        self.sxxf = None
        self.sc = None
        self.scm = None
        #  CHARACTERISTIC FUNCTION PARAMS
        # How long should the short-term and long-term window be, in seconds
        self.sta_len = 120
        self.lta_len = 600
        self.median_len = 21
        self.ch_samps = None
        self.ch_times = None
        self.ch_cutoff = 0.1
        self.ch_trigger = 0.5
        self.filter_range = [0.2, 1]
        # OUTPUT VALUES
        self.arrival_times = []
        self.img_dir = ''

    @staticmethod
    def get_arrivals_from_catalog():
        """From the training catalogs provided by NASA, get the arrival times"""
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
        """Load the data from a .mseed file"""
        st = read(seed_path + fname)
        self.fname = fname.replace('.mseed', '')
        self.raw_times = st.traces[0].times()
        self.times = self.raw_times
        self.raw_data = st.traces[0].data
        self.data = self.raw_data
        self.fs = 1 / (self.times[1] - self.times[0])
        self.cat_atime = self._atimes.get(self.fname)
        self.analyze_capture()

    def analyze_capture(self):
        """Process the loaded data, by applying filters, calculating spectral content, etc"""
        # DATA FILTERING
        self.data = bandpass_filter(self.data, *self.filter_range, self.fs)
        self.data = savgol_filter(self.data, window_length=2001, polyorder=2)
        self.data = self.normalize(self.data)
        self.sxxf, self.sxxt, self.sxx = signal.spectrogram(self.data, self.fs)
        self.calculate_spectral_content()
        self.calculate_characteristic()
        self.scan_for_arrivals(self.ch_trigger)

    def scan_for_arrivals(self, trigger_level):
        """Scan the characteristic function for arrivals"""
        self.arrival_times.clear()
        # data_median = np.median(self.data)
        noise_floor = np.percentile(self.data, 75)
        samp_n = len(self.ch_samps)
        i = 0
        while i < samp_n - 1:
            while self.ch_samps[i] < trigger_level and i < samp_n - 1:
                # REACH THE FIRST TRIGGER
                i += 1
            while self.ch_samps[i] > noise_floor and self.ch_samps[i] > self.ch_samps[i - 1] and i > 0:
                # FIND THE START TIME
                i -= 1
            print(f'Found new arrival time: {self.ch_times[i]}')
            self.arrival_times.append(self.ch_times[i])
            while self.ch_samps[i] < trigger_level and i < samp_n - 1:
                # PASS THROUGH THE TRIGGER SO SAME INSTANT IS NOT REPEATED
                i += 1
            while self.ch_samps[i] >= trigger_level and i < samp_n - 1:
                # GO ON UNTIL PEAK IS PASSED
                i += 1
            while self.ch_samps[i] < trigger_level and i < samp_n - 1:
                # REACH THE NEXT TRIGGER OR TERMINATION
                i += 1

    @staticmethod
    def normalize(data):
        """Normalize the data"""
        normalized_data = data / np.max(data)
        return normalized_data

    def plot_capture(self):
        """Creates the final image with all the plots"""
        self._fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 1, figsize=(19.2, 10.8))
        self._fig.suptitle(self.fname, fontweight='bold')
        # self.plot_data(ax1, self.times, self.raw_data, f'Raw data, fs={round(self.fs, 2)}Hz')
        self.plot_data(ax1, self.times, self.data, f'Raw data, filtered')
        # PLOT ENVELOPE
        envelope = lowpass_filter(5 * np.abs(self.data), 0.01, self.fs)
        self.plot_data(ax1, self.times, envelope, 'Data envelope')
        # PLOT FFT
        self.plot_fft(ax2, self.times, self.data)
        # PLOT SPECTROGRAM
        self.plot_spectrogram(ax3)
        self.plot_spectral_content(ax4)
        if self.cat_atime:
            self.plot_arrival(ax3, self.cat_atime)
            self.plot_arrival(ax4, self.cat_atime)
            self.plot_arrival(ax1, self.cat_atime)
        self.plot_characteristic(ax5)
        for ar in self.arrival_times:
            self.plot_arrival(ax5, ar)
        plt.subplots_adjust(hspace=1)

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

    def calculate_spectral_content(self):
        """Calculates the spectral content from the spectrogram"""
        self.sc = [sum([self.sxx[i][j] for i in range(len(self.sxx))]) for j in range(len(self.sxx[0]))]
        # APPLY MEDIAN FILTER
        self.scm = signal.medfilt(self.sc, kernel_size=self.median_len)

    def plot_spectrogram(self, ax):
        """Plots the spectrogram"""
        color_range = LogNorm(vmin=np.max(self.sxx) / 1000, vmax=np.max(self.sxx))
        vals = ax.pcolormesh(self.sxxt, self.sxxf, self.sxx, cmap=cm.jet, norm=color_range)
        ax.set_xlim([min(self.times), max(self.times)])
        # ax.set_xlabel(f'Time [s]', fontweight='bold')
        ax.set_ylabel('Frequency [Hz]', fontweight='bold')
        cbar = plt.colorbar(vals, orientation='horizontal')
        cbar.set_label('Power ((m/s)^2/sqrt(Hz))', fontweight='bold')
        ax.set_title(f'Spectrogram', fontweight='bold')

    def plot_spectral_content(self, ax):
        """Plot the spectral content of the data"""
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
        """Plot the RAW data"""
        ax.plot(times, data, label=label)
        ax.set_xlim([min(times), max(times)])
        ax.set_ylabel('Normalized velocity [m/s]', fontweight='bold')
        ax.set_xlabel('Time [s]', fontweight='bold')
        ax.set_title(f'Raw data plot', fontweight='bold')
        if PLOT_GRIDS:
            ax.grid()
        ax.legend()

    def calculate_characteristic(self):
        """Get the characteristic function, which will be used to deduce sismic events programmatically"""
        med = np.median(self.scm)
        epsilon = 1e-3  # to avoid division by 0
        self.ch_samps = (self.scm + epsilon) / (med + epsilon)
        self.ch_times = np.linspace(self.times[0], self.times[-1], len(self.scm))
        self.ch_samps = (self.ch_samps + epsilon) / (med + epsilon)
        # self.ch_samps = self.sta_lta(self.ch_samps, int(self.sta_len * freq), int(self.lta_len * freq))
        # self.ch_samps = classic_sta_lta(self.ch_samps, int(self.sta_len * freq), int(self.lta_len * freq))
        self.ch_samps = lowpass_filter(self.ch_samps, self.ch_cutoff, 1/(self.ch_times[1] - self.ch_times[0]))
        self.ch_samps = self.normalize(self.ch_samps)

    @staticmethod
    def sta_lta(data, nsta, nlta, demean=False):
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
        if len(data) < nlta:
            raise ValueError("Signal length must be at least as long as nlta")
        # Optional: Remove mean to center the signal around zero
        if demean:
            data = data - np.mean(data)
        # Use absolute signal to ensure non-negative values
        abs_signal = np.abs(data)
        # Define the window for STA and LTA
        window_sta = np.ones(nsta)
        window_lta = np.ones(nlta)
        # Compute STA and LTA using convolution
        sta = convolve(abs_signal, window_sta, mode='same') / nsta
        lta = convolve(abs_signal, window_lta, mode='same') / nlta
        # Avoid division by zero by adding a small epsilon to LTA
        epsilon = 1e-10
        ratio = (sta + epsilon) / (lta + epsilon)
        return ratio

    def plot_characteristic(self, ax):
        """Plot characteristic function"""
        ax.plot(self.ch_times, self.ch_samps, label='Characteristic function')
        ax.set_xlim([min(self.ch_times), max(self.ch_times)])
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Characteristic function', fontweight='bold')
        # ax.set_ylim([1e-1, 1e0])
        ax.grid()
        ax.legend()

    @staticmethod
    def plot_arrival(ax, a_time):
        ax.axvline(x=a_time, c='red')

    def save_image(self, comment=''):
        """Save the current plots"""
        if self._fig:
            self.img_dir = f'med{self.median_len}_fil{self.filter_range}_trig{self.ch_trigger}_chcut{self.ch_cutoff}/'
            out_dir = OUTPUT_FILES_ROOT + RELATIVE_ROOT + self.img_dir
            out_name = self.fname + comment + '.png'
            image_path = out_dir + out_name
            print(f'Saving image {image_path}')
            if not os.path.exists(OUTPUT_FILES_ROOT + RELATIVE_ROOT + self.img_dir):
                os.makedirs(OUTPUT_FILES_ROOT + RELATIVE_ROOT + self.img_dir)
            plt.savefig(image_path)
            plt.close(self._fig)
            self._fig = None


if __name__ == '__main__':
    # CREATE and CONFIGURE the model
    model = Model()
    model.filter_range = MOD_PARAMS[PLANET]['filter_range']
    model.median_len = MOD_PARAMS[PLANET]['median_len']
    model.ch_trigger = MOD_PARAMS[PLANET]['ch_trigger']
    model.ch_cutoff = MOD_PARAMS[PLANET]['ch_cutoff']
    # LIST OF FILENAMES - ARRIVAL FOR THE CATALOG
    fnames = []
    ar_times = []
    # SCAN DATA DIRECTORIES
    for file in os.listdir(PATH_ROOT + RELATIVE_ROOT):
        # ONLY .mseed FILES ARE SUPPORTED
        if file.endswith('.mseed'):
            # LOAD THE MODEL WITH THE DATA
            model.load_from_seed(PATH_ROOT + RELATIVE_ROOT, file)
            # CREATE A DATA PLOT
            model.plot_capture()
            # SAVE THE PLOT
            model.save_image()
            # ADD ARRIVALS TO CATALOG
            ar_times.extend(model.arrival_times)
            fnames.extend([model.fname for _ in model.arrival_times])
    # CREATE CATALOG IN CSV FORMAT
    detect_df = pd.DataFrame(data={'filename': fnames,
                                   'time_rel(sec)': ar_times})
    detect_df.head()
    cat_path = OUTPUT_FILES_ROOT + RELATIVE_ROOT + model.img_dir + f'{PLANET}_catalog.csv'
    print(f'Saving catalog to {cat_path}')
    detect_df.to_csv(cat_path, index=False)
