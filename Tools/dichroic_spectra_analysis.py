'''
Written by Oleksandra Rebrysh.
09/10/2024
'''

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from astropy.modeling import models, fitting, polynomial


class DichroicSpectra:
    # Define the Sodium D-line wavelength range (e.g., 5890 - 5896 Å) within FWHM
    na_line_min = 5800
    na_line_max = 5980

    def __init__(self, path, filename, ripple_threshold = 0.005, columns_to_load = None, sheet_name = None, skiprows = None, delimiter = None):
        self.data = os.path.join(path,filename)
        self.ripple_threshold = ripple_threshold
        self.wavelength, self.intensity = self.LoadData(self.data, columns_to_load, sheet_name, skiprows)
        self.ripple = None
        self.wavelength_A_excl = None
        self.spectral_scale = None


    @staticmethod
    def LoadData(file_path, columns_to_load, sheet_name, skiprows):

        '''
        This function checks the extension of the file and loads data correspondingly.
        The wavelength and T/R is then splitted into 2 1D numpy arrays.
        '''

        # Check the file format
        _, file_extension = os.path.splitext(file_path)
        acceptable_formats = ['.xlsx', '.xls', '.xlsm', '.txt', '.csv']

        if file_extension not in acceptable_formats:
            raise ValueError(f"Invalid file format: {file_extension}. Please provide an Excel file (.xlsx or .xls) or a Text file (.txt).")

        if file_extension in ['.xlsx', '.xls', '.xlsm']:
            data = pd.read_excel(file_path, sheet_name=sheet_name, usecols=columns_to_load, skiprows = skiprows)
        elif file_extension in ['.txt', '.csv']:
            data = pd.read_csv(file_path, usecols=columns_to_load, header = None, skiprows = skiprows, delim_whitespace=True)

        data = data.dropna()
        print(data)

        if data.shape[1] < 2:
            raise ValueError("Data must contain at least two columns for wavelength and intensity.")

        wavelength = data.iloc[:, 0].to_numpy()
        intensity = data.iloc[:, 1].to_numpy() #Transmission or Reflection

        if np.max(intensity) <= 1:
            intensity = intensity*100

        return wavelength, intensity

    def PlotRawData(self, xmin, xmax, title, label = None, ifsave = False, path = None, name = None, figure = None):
        from matplotlib.ticker import MultipleLocator, AutoMinorLocator

        if figure is None:
            plt.figure()

        if label:
            plt.plot(self.wavelength, self.intensity, label = label, linewidth = 1)
            plt.legend()
        else:
            plt.plot(self.wavelength, self.intensity)
        plt.title(title)
        plt.xlabel('Wavelength, nm')
        plt.ylabel('T (R), %')
        plt.xlim(xmin,xmax)
        plt.grid(which='major', color='#DDDDDD', linewidth=0.8)
        plt.grid(which='minor', color='#EEEEEE', linestyle=':', linewidth=0.5)


        ax = plt.gca()
        ax.xaxis.set_minor_locator(MultipleLocator(50))

        if ifsave:
            if path is None or name is None:
                raise ValueError("Error: You must provide a path and a filename when saving the plot.")

            save_path = os.path.join(path, name)
            plt.savefig(save_path, dpi = 600)

        return


    # def auto_round(self, number):
    #     from decimal import Decimal, ROUND_05UP, getcontext
    #
    #     decimal_number = Decimal(str(number))
    #     if decimal_number == decimal_number.to_integral():
    #         # If it's an integer, no rounding needed
    #         return decimal_number
    #     else:
    #         # Determine precision based on significant decimal places
    #         precision = Decimal('1e-{0}'.format(abs(decimal_number.as_tuple().exponent)))
    #         # Perform rounding with ROUND_HALF_UP
    #         rounded_number = decimal_number.quantize(precision, rounding=ROUND_05UP)
    #
    #     return rounded_number

    def LocalMaximum(self):

        '''
        This function seeks for peaks/spikes.
        Return:
        peaks:
        self.wavelength[peaks]: wavelength at which a peak is located.
        fwhm: fwhm in wavelength units
        '''
        from scipy.signal import find_peaks, peak_widths
        peaks, _ = find_peaks(self.intensity, height = 0.007)
        print(self.wavelength[peaks])

        fwhm = peak_widths(self.intensity, peaks, rel_height = 0.5)

        return peaks, self.wavelength[peaks], fwhm[0]*(self.wavelength[1]-self.wavelength[0])


    def Mask(self, w_min, w_max):

        '''
        The boolean mask.
        Return:
        cut_mask: numpy array
        wavelength_masked:
        intensity_masked: transmission/reflection
        '''

        if w_min < np.min(self.wavelength) or w_max > np.max(self.wavelength):
            raise ValueError(f"Wavelength range must be within {np.min(self.wavelength)} nm and {np.max(self.wavelength)} nm")

        cut_mask = np.where((self.wavelength >= w_min)&(self.wavelength <= w_max))

        wavelength_masked = self.wavelength[cut_mask]
        intensity_masked = self.intensity[cut_mask]

        return cut_mask, wavelength_masked, intensity_masked

    def MeanTranRef(self, wmin, wmax):

        '''
        This function calculates mean value of transmission/reflection.

        NOTE: Add check if data is not in %, then either convert to % or change the print output.
        '''

        _, _, intensity_masked = self.Mask(wmin, wmax)
        mean_TR = np.mean(intensity_masked)
        print(f'Mean Transmission/Reflection on the range {wmin}-{wmax} nm: {mean_TR}%')

        return mean_TR

    def Ripple(self, w_min = 368, w_max = 935, Notch = False, saveTable = False, path = None, name = None, verbose = False):

        '''
        This function calculates the ripple as ΔΤ(R)/Δλ.
        Notch: True applies only to the LGS spectrum.

        Return:
        ripple: numpy array
        '''

        _, wavelength_mskd, SCI_spectra = self.Mask(w_min, w_max)
        wavelength_A = wavelength_mskd*10

        self.spectral_scale = np.diff(wavelength_A)
        if not np.all(self.spectral_scale <= 1):
            # raise ValueError("Spectral scale is more than 1 Ångström! Please check the data or scaling.")
            print(f"Spectral scale is more than 1 Ångström! Please check the data or scaling. Average spectral scale is {np.mean(self.spectral_scale)}")
            pass

        if Notch:
            na_indices = np.where((wavelength_A >= DichroicSpectra.na_line_min) & (wavelength_A <= DichroicSpectra.na_line_max))
            wavelength_A_excl = np.delete(wavelength_A, na_indices)
            spectra_excl = np.delete(SCI_spectra, na_indices)
        else:
            wavelength_A_excl = wavelength_A
            spectra_excl = SCI_spectra
        self.wavelength_A_excl = wavelength_A_excl
        # print(wavelength_A_excl.shape)
        # print(spectra_excl.shape)

        ripple = np.zeros((len(wavelength_A_excl)-1))
        for i in range(len(wavelength_A_excl)-1):

            ripple[i] = (spectra_excl[i+1]- spectra_excl[i])/(wavelength_A_excl[i+1]-wavelength_A_excl[i])

        self.ripple = ripple
        # print(self.ripple.shape)

        total_ripple_values = len(ripple)
        high_ripple_indices = np.where(np.abs(self.ripple) > self.ripple_threshold)
        print(self.ripple_threshold)
        # print(high_ripple_indices[0])
        num_exceeding = len(high_ripple_indices[0])  # Count how many values exceed the threshold
        percentage_exceeding = (num_exceeding / total_ripple_values) * 100

        print(f'Out of Band ripple at resolution {np.mean(self.spectral_scale)} Å:{np.percentile(abs(self.ripple), 98)}%')
        print(f'Out of Band ripple exceeding the threshold at resolution {np.mean(self.spectral_scale)} Å: {percentage_exceeding}%')

        high_ripple_wavelengths = wavelength_A_excl[:-1][high_ripple_indices]
        high_ripple = ripple[high_ripple_indices]
        if saveTable == True:
            ripple_table = pd.DataFrame({'wavelength, A': wavelength_A_excl[:-1][np.where(abs(self.ripple))],
                                            'ripple, %/A': self.ripple})
            if path is None or name is None:
                raise ValueError("Error: You must provide a path and a filename when saving the table.")

            save_path = os.path.join(path, name)
            ripple_table.to_csv(save_path)

        if verbose == True:
            for wl in high_ripple_wavelengths:
                print(f"Ripple exceeds {self.ripple_threshold}%/Å at wavelength: {wl:.2f} Å")

        return ripple


    def PlotRipple(self, ifsave = False, path = None, name = None):

        '''
        This function plots ripples (slopes) in % as a function of wavelength.
        '''

        if self.wavelength_A_excl is None or self.ripple is None:
            print("You need to call Ripple() before PlotRipple()")
            return

        plt.figure()
        plt.scatter(self.wavelength_A_excl[:-1], self.ripple, marker = '.', s = 5, c = 'black')
        plt.axhline(-self.ripple_threshold, color='r', linestyle='--', label=f"{self.ripple_threshold}%/Å limit")
        plt.axhline(self.ripple_threshold, color='r', linestyle='--')
        plt.title('Out-of-band ripple')
        plt.xlabel('Wavelength, A')
        plt.ylabel(f'Ripple, %/Å')
        plt.legend()
        if ifsave:
            if path is None or name is None:
                raise ValueError("Error: You must provide a path and a filename when saving the plot.")

            save_path = os.path.join(path, name)
            plt.savefig(save_path, dpi=600)

        return

    def FitNotch(self, ifsave = False, path = None, name = None):

        '''
        This function fits the Notch in a super smart way, using some machine learning. (Standard fitting does not do its work properly :))
        FWHM calculated and printed within this function.
        NOTE: does not work properly if spectra are in transmission. Add condition to check T or R data and fit accordingly to it.
        '''

        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

        #Going hard way as none of the standard fitting works well :(

        notchzoom_mask = np.where((self.wavelength >= 560)&(self.wavelength <= 620))
        wavelength_zoom = self.wavelength[notchzoom_mask]
        intensity_zoom = self.intensity[notchzoom_mask]

        X = wavelength_zoom.reshape(-1, 1)  # Reshape for the model
        # y = 100 - intensity_zoom
        y = intensity_zoom

        # Define the kernel
        kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=1, length_scale_bounds=(1e-2, 1e2))

        # Create Gaussian Process model
        gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)
        # Fit to the data
        gp.fit(X, y)
        # Make a prediction over the range of wavelengths
        x_pred = np.linspace(np.min(wavelength_zoom), np.max(wavelength_zoom), 1000).reshape(-1, 1)
        y_pred, sigma = gp.predict(x_pred, return_std=True)

        #FWHM
        max_value = np.max(y_pred)
        # max_value = np.min(y_pred)
        half_max = max_value/2

        indices_above_half_max = np.where(y_pred >= half_max)[0]
        if len(indices_above_half_max) < 2:
            print("Unable to find two points above half maximum for FWHM calculation.")
            return None

        fwhm_start_index = indices_above_half_max[0]
        fwhm_end_index = indices_above_half_max[-1]

        fwhm = x_pred[fwhm_end_index] - x_pred[fwhm_start_index]
        fwhm_value = fwhm[0]
        print(f'FWHM: {fwhm_value} nm')

        plt.figure()

        plt.plot(wavelength_zoom, y, linewidth = 2, label = 'Data')
        # plt.plot(wavelength_zoom, fit_notch(wavelength_zoom), label = 'Gaussian fit')
        plt.plot(x_pred, y_pred, c = 'red', label = 'Gaussian Regressor Fit', linewidth = 0.8)
        plt.legend()
        plt.xlabel('Wavelength, nm')
        plt.ylabel('T (R), %')

        if ifsave:
                if path is None or name is None:
                    raise ValueError("Error: You must provide a path and a filename when saving the plot.")

                save_path = os.path.join(path, name)
                plt.savefig(save_path, dpi = 600)

        return

    def FindTransitionRegion(self, multiplier, image = False, ifsave = False, path = None, name = None):
        '''
        Automatically find the transition region between reflection and transmission using derivatives.
        '''

        _, wavelength_mskd, intensity_mskd = self.Mask(900, 1000)

        # Compute the derivative (rate of change)
        derivative = np.gradient(intensity_mskd, wavelength_mskd)

        # Threshold to detect significant changes (adjust based on data characteristics)
        derivative_threshold = np.max(np.abs(derivative)) * multiplier
        transition_indices = np.where(np.abs(derivative) >= derivative_threshold)[0]

        if len(transition_indices) == 0:
            print("Unable to detect transition region automatically.")
            return None

        w_transition_min = wavelength_mskd[transition_indices[0]]
        w_transition_max = wavelength_mskd[transition_indices[-1]]

        transition_width = w_transition_max - w_transition_min

        print(f"Transition region detected between {w_transition_min:.2f} nm and {w_transition_max:.2f} nm.")
        print(f"Transition region width: {transition_width:.2f} nm")

        if image:
            plt.figure()
            plt.plot(wavelength_mskd, intensity_mskd, c = 'black')
            plt.axvline(w_transition_min, color='r', linestyle='--', label='Transition Start')
            plt.axvline(w_transition_max, color='g', linestyle='--', label='Transition End')
            plt.title(f"Transition Region Detection")
            plt.xlabel("Wavelength (nm)")
            plt.ylabel("T (R), %")
            plt.legend()
            plt.grid(True)

            if ifsave:
                if path is None or name is None:
                    raise ValueError("Error: You must provide a path and a filename when saving the plot.")

                save_path = os.path.join(path, name)
                plt.savefig(save_path, dpi = 600)


        return w_transition_min, w_transition_max, transition_width

    def OpticalDensity(self, wmin, wmax):

        '''
        This function calculates optical density by using analytical formula.
        '''

        _, _, transmission = self.Mask(wmin, wmax)
        if transmission.dtype == object:
            transmission = transmission.astype(float)

        transmission = np.where(transmission <= 0, 1e-10, transmission)
        OD = - np.log10(transmission/100)

        print(f'Average OD on the range {wmin}-{wmax} nm: {np.mean(OD)}')

        return OD


    def ResolutionSelection(self, w_min, w_max, resolution):

        '''
        This function filters the data up to a certain resolution by skipping all the values in between.
        Be careful as it overwrites wavelength and T/R arrays, so all the methods called after this one will use new values.
        '''

        _, wavelength_mskd, intensity_mskd = self.Mask(w_min, w_max)

        filtered_wavelengths = [wavelength_mskd[0]]
        filtered_intensities = [intensity_mskd[0]]

        i = 1
        while i < len(wavelength_mskd):
            # Check if the difference matches the resolution
            if abs(wavelength_mskd[i] - filtered_wavelengths[-1] - resolution) < 1e-9:
                filtered_wavelengths.append(wavelength_mskd[i])
                filtered_intensities.append(intensity_mskd[i])
            i += 1


        #print(np.vstack((filtered_wavelengths, filtered_intensities)))
        self.wavelength = np.array(filtered_wavelengths)
        self.intensity = np.array(filtered_intensities)

        return self.wavelength, self.intensity





