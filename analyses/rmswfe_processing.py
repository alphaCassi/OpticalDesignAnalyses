#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

def process_txt(file_path, D, Fn, wavelengths):
    '''
    The function to process txt file of the rms wavefront error data.
    The encoding should be utf-8.
    The first column of the data is field points in mm. The second column is polychromatic errors.
    All other columns are errors for different wavelengths.

    Parameters:
    file_path: file path and name. Format: txt.
    D: diameter of the pupil [m] required to convert field coordinates from mm to arcsec.
    Fn: F-number of the system, required to convert field coordinates from mm to arcsec.
    wavelength: can be either a number or an array. Should be in nm.

    Output:
    field_data: field coordinates in arcsec.
    poly_data: polychromatic rms wavefront error in waves.
    errors: rms wavefront errors for different wavelength, in nm.
    '''

    data = np.loadtxt(file_path, skiprows=1)
    num_wavelengths = len(wavelengths)

    field_data = data[:, 0]*206265/(D * Fn)
    poly_data = data[:, 1]
    errors = np.zeros((data.shape[0], num_wavelengths))

    if isinstance(wavelengths, (int, float)):
        errors = data[:, 2:] * wavelengths
    else:
        for i in range(num_wavelengths):
            errors[:, i] = data[:, i+2] * wavelengths[i]  # Skipping first two columns

    return field_data, poly_data, errors


def plot_processed_data(field_data, errors, wavelength, savepath, filename, ymax, vertical_line = False, vline_x = None, vline_label = None):

    '''
    The function for plotting the RMS wavefront error versus field.

    Parameters:
    field_data: field points in arcsec.
    errors: rms wavefront errors in nm.
    wavelength: can be either a number or an array. Should be in nm.
    savepath: path where to save the plot.
    filename: name for the plot.
    ymax: maximum y coordinate of the figure.
    vertical line: boolean. If True, the vertical line will be plotted.
    vline_x: x coordinate for the vertical line. None by default.
    vline_label: label for the vertical line: None by default.

    '''

    # plt.figure(figsize=(10, 6))
    if isinstance(wavelength, (int, float)):
        plt.plot(field_data, errors, label = f'{wavelength} nm')
    else:
        for i in range(errors.shape[1]):
            plt.plot(field_data, errors[:, i], label = f'{wavelength[i]} nm')

    if vertical_line == True:
        plt.vlines(vline_x, 0, ymax, colors = 'red')
        plt.text(vline_x, ymax - 10, vline_label, rotation='vertical', verticalalignment='top', size = 14)

    plt.ylim(0, ymax)
    plt.xlabel('+Y Field, arcsec', size = 16)
    plt.ylabel('RMS WFE, nm', size = 16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.title('RMS wavefront error vs Field', size = 18)
    plt.legend(loc='upper left', prop={'size': 14})
    plt.grid(True)

    plt.savefig(savepath+filename, dpi = 600)
    # plt.show()

    return

def average_wfe(data, wavelength):

    '''
    The function to calculate the average rms wavefront errors across the FoV or defined field.

    Parameters:
    data: wavefront error data.
    wavelength: wavelength in nm to convert wfe from waves to nm rms.

    Output:
    rms_av_waves: average rms wfe in waves.
    rms_av_nm: average rms wfe in nm.

    '''
    rms_av_waves = np.mean(data)
    rms_av_nm = rms_av_waves * wavelength

    return rms_av_waves, rms_av_nm


#Example of the use

# if __name__ == "__main__":
#
#     file_path_sci = 'D:\PhD\VTB\pics\\RMSvsField_DM277_vis_data.txt'
#     wavelength_sci = [480, 587.6, 650, 550]
#     field_data, poly_errors, errors = process_txt(file_path_sci, 8000, 15, wavelength_sci)
#     print(field_data)
#
#     savepath = "D:\PhD\VTB\pics\\"
#     filename = "test.png"
#     plt.figure(figsize = (10, 6))
#     plot_processed_data(field_data, errors, wavelength_sci, savepath, filename, 110, vertical_line = True, vline_x = 15, vline_label = "Sci fov radius")
#     plt.show()
#
#     rms_av_waves, rms_av_nm = average_wfe(poly_errors, wavelength[3])
