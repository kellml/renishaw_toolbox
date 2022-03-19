# Use this script to average consolidated spectra and plot with st.dev error bars (stdev-sample) and export
#   a single (x, y) data
# Assumes column data is (wavelength, data1, data2, ...) format


# Import libraries #####################################################################################################


# Import libraries:
import sys
import matplotlib.pyplot as plt
import numpy as np
from numpy import ndarray
import tkinter as tk
from tkinter import filedialog
import statistics as st
import os
import glob


# Functions ############################################################################################################


# Functions: (generally in order they are called)


def consolidate_end():
    print('\n' + 'All imported txt files have been consolidated and saved as \'consolidated_data.txt\' in ' +
          'directory: ' + '\n' +
          os.getcwd() + '\n' +
          'Program will now process the consolidated data.' + '\n')


def txt_consolidate():
    # GUI select PL file to import:
    tk_frame = tk.Tk()  # create instance of Tkinter frame
    tk_frame.attributes("-topmost", True)  # bring window to front
    tk_frame.withdraw()  # hides the extra tKinter GUI window besides the dialog box
    specified_directory = filedialog.askdirectory()  # dialog box to choose one file
    current_path()

    # Change working directory to selected directory:
    os.chdir(specified_directory)  # change directory
    chosen_path()

    # Create list of .txt files in the directory:
    filenames_list = [i for i in glob.glob(f'*.txt')]
    print('Consolidating all ' + str(len(filenames_list)) + ' .txt files in directory:' + '\n')

    # Dummy file import of first .txt file for getting dimensions of empty array:
    dummy = np.loadtxt(filenames_list[0])

    # Form empty array of desired size:
    file_data: ndarray = np.zeros((len(dummy),
                                   int(len(filenames_list)) + 1))  # '+1' because first column is x and others are y

    # Import each file in the directory: (this loop works, but np.insert and/or enumerate may be cleaner?)
    import_index = 0
    for i in range(0, len(filenames_list)):  # indexes 0-2 in the 3 element filenames list
        if import_index == 0:  # imports wavelength and spectrum data
            first_file_data = np.loadtxt(filenames_list[i])
            file_data[:, 0] = first_file_data[:, 0]
            file_data[:, 1] = first_file_data[:, 1]
            import_index += 1
            print('Imported .txt file #' + str(i + 1) + '/' + str(len(filenames_list)) + '.')
        else:  # imports only spectrum data
            subsequent_file_data = np.loadtxt(filenames_list[i])
            file_data[:, i + 1] = subsequent_file_data[:, 1]
            import_index += 1
            print('Imported .txt file #' + str(i + 1) + '/' + str(len(filenames_list)) + '.')

    # Save the consolidated file_data to a .txt file in the same directory and display end message:
    np.savetxt('consolidated_data.txt', file_data, delimiter='\t')  # tab-delimited .txt
    consolidate_end()


def renishaw_data_type(prompt):
    specified_data_type = input(prompt)
    if specified_data_type == 'p':
        print('Consolidated PL data will be imported, normalized, averaged, and saved with ' +
              'standard deviations.' + '\n')
        import_marker = 0  # 0 for PL, 1 for Raman
        raw_data = np.loadtxt('consolidated_data.txt')  # numpy array of .txt file data, 'unpack' for 2 column arrays

    elif specified_data_type == 'r':
        print('Consolidated Raman data will be imported, normalized, averaged, and saved with ' +
              'standard deviations.' + '\n')
        import_marker = 1  # 0 for PL, 1 for Raman
        raw_data = np.loadtxt('consolidated_data.txt')  # numpy array of .txt file data, 'unpack' for 2 column arrays
    else:
        print('Invalid input. Program will terminate.')
        sys.exit()
    return specified_data_type, import_marker, raw_data


def current_path():
    print('Original working directory:\n' + os.getcwd() + '\n')


def chosen_path():
    print('Directory changed to:\n' + os.getcwd() + '\n')


def consolidated_data_summary():
    summary_directory = 'consolidated_data_summary'
    os.mkdir(summary_directory)
    os.chdir(summary_directory)


def wl_to_ev(wavelength):
    # Constants:
    h = 4.14e-15  # eV-s
    c = 3e8  # m/s
    # Wavelength to eV conversion:
    ev = (h * c) / wavelength * 1e9
    return ev


def find_nearest(array, value):  # find index of nearest value in array from specified input
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx  # must return value of idx otherwise calling function doesn't work


def pl_norm_spec():
    x_array = wl_to_ev(raw_data[:, 0])  # convert wavelength data to eV

    # Convert wavelength (column 1) to eV: (separately for untreated and treated here)
    pl_ev = wl_to_ev(raw_data[:, 0])
    raw_pl = np.insert(raw_data, 1, pl_ev,
                       axis=1)  # insert eV column to right of the wavelength column
    # (new column index '1')

    # Normalize each spectrum to 654.38 nm (Si Raman peak): (if/else later for choice of not normalizing)
    wl_norm_value = float(input('Enter wavelength in nm to normalize to.' + '\n' +
                                '(e.g. Si Raman peak is 654.38 nm with 633 nm excitation): '))
                                # float type to operate with np functions
    wl_norm_index = find_nearest(np.asarray(raw_pl[:, 0]), wl_norm_value)

    print('Imported spectrum will be normalized to closest value to: '
          + str(wl_norm_value) + ' nm.\n')
    print('Closest value found to ' + str(wl_norm_value) + ' nm was: '
          + str(raw_pl[wl_norm_index, 0]) + ' nm.\n')
    print('Index within the array for the found value was: ' + str(wl_norm_index) + '.\n')
    return x_array, wl_norm_value, wl_norm_index, raw_pl


def raman_norm_spec():
    x_array = raw_data[:, 0]

    raw_raman = raw_data

    # Normalize each spectrum to 654.38 nm (Si Raman peak): (if/else later for choice of not normalizing)
    wn_norm_value = float(input('Enter wavenumber in cm-1 to normalize to.' + '\n' +
                                '(e.g. Si has Raman band at 520 cm-1): '))
                                # float type to operate with np functions
    wn_norm_index = find_nearest(np.asarray(raw_raman[:, 0]), wn_norm_value)

    print('Imported spectrum will be normalized to closest value to: '
          + str(wn_norm_value) + ' nm.\n')
    print('Closest value found to ' + str(wn_norm_value) + ' nm was: '
          + str(raw_raman[wn_norm_index, 0]) + ' nm.\n')
    print('Index within the array for the found value was: ' + str(wn_norm_index) + '.\n')
    return x_array, wn_norm_value, wn_norm_index, raw_raman


def pl_process():
    # Loop to normalize each of the 8 flakes' PL data:
    wl_ev_column_indices = 2  # column index 2:end is PL data; first 2 columns are wl and eV and don't need normalizing

    norm_pl = raw_pl_data
    for i in range(wl_ev_column_indices, len(raw_pl_data[0, :])):
        norm_pl[:, i] = raw_pl_data[:, i] / raw_pl_data[wl_norm_idx, i]  # normalized pl array

    # Calculate average spectra with error regions showing standard deviation (matches excel stdev.s)
    norm_pl_shape = norm_pl.shape
    avgs = np.zeros((len(norm_pl), 1))
    stdevs = np.zeros((len(norm_pl), 1))

    # Loop for each row in columns 2-end; calc avg. and st.dev:
    for i in range(0, len(norm_pl)):
        avgs[i] = st.mean(norm_pl[i, wl_ev_column_indices:norm_pl_shape[1]])
        stdevs[i] = st.stdev(norm_pl[i, wl_ev_column_indices:norm_pl_shape[1]])

    return norm_pl, avgs, stdevs


def raman_process():
    # Loop to normalize each of the 8 flakes' PL data:
    wn_column_indices = 1  # column index 2:end is PL data; first 2 columns are wl and eV and don't need normalizing

    norm_raman = raw_raman_data
    for i in range(wn_column_indices, len(raw_raman_data[0, :])):
        norm_raman[:, i] = raw_raman_data[:, i] / raw_raman_data[wn_norm_idx, i]  # normalized pl array

    # Calculate average spectra with error regions showing standard deviation (matches excel stdev.s)
    norm_raman_shape = norm_raman.shape
    avgs = np.zeros((len(norm_raman), 1))
    stdevs = np.zeros((len(norm_raman), 1))

    # Loop for each row in columns 2-end; calc avg. and st.dev:
    for i in range(0, len(norm_raman)):
        avgs[i] = st.mean(norm_raman[i, wn_column_indices:norm_raman_shape[1]])
        stdevs[i] = st.stdev(norm_raman[i, wn_column_indices:norm_raman_shape[1]])

    return norm_raman, avgs, stdevs


def plot_consolidated_data():
    # Plot and save figures of raw data, normalized data, and average data (with stdev error bars):
    if import_value == 0:  # PL
        print('Saving plots of raw data, normalized data, and average data (with stdev error bars)' + '\n')
        # Raw data plot:
        raw_plot = np.loadtxt('raw_consolidated_data.txt')
        for i in range(2, len(raw_plot[0, :])):  # seems to be plotting normalized data
            plt.figure(1)
            plt.plot(raw_plot[:, 1], raw_plot[:, i])
            plt.title('Raw PL data')
            plt.xlabel('Energy (eV)')
            plt.ylabel('Intensity (counts)')
        plt.figure(1).savefig('raw_pl.png')
        plt.show()

        # Normalized data plot:
        for i in range(2, len(norm_data[0, :])):
            plt.figure(2)
            plt.plot(norm_data[:, 1], norm_data[:, i])
            plt.title('Normalized PL data')
            plt.xlabel('Energy (eV)')
            plt.ylabel('Intensity, norm. to ' + str(wl_norm_val) + ' (arb. units)')
        plt.figure(2).savefig('normalized_pl.png')
        plt.show()

        # Average data plot: (only 3 columns: x, avg_y, stdev_y)
        plt.figure(3)
        plt.plot(stat_data[:, 0], stat_data[:, 1], color='dodgerblue')
        plt.fill_between(stat_data[:, 0], np.add(stat_data[:, 1], stat_data[:, 2]),
                         np.subtract(stat_data[:, 1], stat_data[:, 2]), color='aliceblue')
        plt.title('Average PL data')
        plt.xlabel('Energy (eV)')
        plt.ylabel('Intensity, norm. to ' + str(wl_norm_val) + ' (arb. units)')
        plt.figure(3).savefig('average_stdev_pl.png')
        plt.show()

    elif import_value == 1:  # Raman
        print('Saving plots of raw data, normalized data, and average data (with stdev error bars)' + '\n')
        # Raw data plot:
        raw_plot = np.loadtxt('raw_consolidated_data.txt')
        for i in range(1, len(raw_plot[0, :])):  # seems to be plotting normalized data
            plt.figure(1)
            plt.plot(raw_plot[:, 0], raw_plot[:, i])
            plt.title('Raw Raman data')
            plt.xlabel('Raman shift (cm-1)')
            plt.ylabel('Intensity (counts)')
        plt.figure(1).savefig('raw_raman.png')
        plt.show()

        # Normalized data plot:
        for i in range(1, len(norm_data[0, :])):
            plt.figure(2)
            plt.plot(norm_data[:, 0], norm_data[:, i])
            plt.title('Normalized Raman data')
            plt.xlabel('Raman shift (cm-1)')
            plt.ylabel('Intensity, norm. to ' + str(wn_norm_val) + ' (arb. units)')
        plt.figure(2).savefig('normalized_raman.png')
        plt.show()

        # Average data plot: (only 3 columns: x, avg_y, stdev_y)
        plt.figure(3)
        plt.plot(stat_data[:, 0], stat_data[:, 1], color='dodgerblue')
        plt.fill_between(stat_data[:, 0], np.add(stat_data[:, 1], stat_data[:, 2]),
                         np.subtract(stat_data[:, 1], stat_data[:, 2]), color='aliceblue')
        plt.title('Average Raman data')
        plt.xlabel('Raman shift (cm-1)')
        plt.ylabel('Intensity, norm. to ' + str(wn_norm_val) + ' (arb. units)')
        plt.figure(3).savefig('average_stdev_raman.png')
        plt.show()


def program_close():
    os.chdir('..')  # change from summary directory to directory of imported file
    print('Program has completed successfully:\n' +
          '1. Consolidated raw, normalized, and averaged data (and figures) saved in \'consolidated_data_summary\'' +
          ' directory in: \n' +
          os.getcwd() + '\n' +
          '2. Modify the \'summary\' directory name immediately to avoid future error in data overwrite.\n'
          )


# Actual Program #######################################################################################################


# Consolidate directory of .txt files:
txt_consolidate()


# Specify which data is imported:
import_data_type, import_value,\
    raw_data = renishaw_data_type('What type of experiment was the consolidated data for? ' + '\n' +
                                  'Enter \'p\' for PL. (mandatory eV conversion from wavelength)' + '\n' +
                                  'Enter \'r\' for Raman. (no eV conversion and can use on any (x, y) data)' + '\n' +
                                  'Enter value here: ')


# Make directory named 'consolidated_summary' and change CWD to it:
consolidated_data_summary()


# Specify normalization value and save named copy of raw data as .txt:
# (because raw_pl_data is not raw and is normalized --> figure out)
if import_value == 0:
    x, wl_norm_val, wl_norm_idx, raw_pl_data = pl_norm_spec()  # setup PL data normalization
    np.savetxt('raw_consolidated_data.txt', raw_pl_data, delimiter='\t')  #
elif import_value == 1:
    x, wn_norm_val, wn_norm_idx, raw_raman_data = raman_norm_spec()  # setup Raman data normalization
    np.savetxt('raw_consolidated_data.txt', raw_raman_data, delimiter='\t')  #
else:
    print('I don\'t know what to do.')  # should be unreachable


# Normalize data, calculate average and stdevs, and save consolidated normlaized data and avg/stdev:
if import_value == 0:
    norm_data, avg_data, stdev_data = pl_process()  # normalize, average, and stdev PL data

    stat_data: ndarray = np.zeros((len(x), 3))
    stat_data[:, 0] = x  # eV
    stat_data[:, 1] = avg_data[:, 0]
    stat_data[:, 2] = stdev_data[:, 0]

    np.savetxt('norm_consolidated_data.txt', norm_data, delimiter='\t')  # normalized data, tab-delimited .txt
    np.savetxt('avg_stdev_data.txt', stat_data, delimiter='\t')  # average and stdev data, tab-delimited .txt
elif import_value == 1:
    norm_data, avg_data, stdev_data = raman_process()  # normalize, average, and stdev Raman data

    stat_data: ndarray = np.zeros((len(x), 3))
    stat_data[:, 0] = x  # cm-1
    stat_data[:, 1] = avg_data[:, 0]
    stat_data[:, 2] = stdev_data[:, 0]

    np.savetxt('norm_consolidated_data.txt', norm_data, delimiter='\t')  # normalized data, tab-delimited .txt
    np.savetxt('avg_stdev_data.txt', stat_data, delimiter='\t')  # average and stdev data, tab-delimited .txt
else:
    print('I don\'t know what to do.')  # should be unreachable


# Save plots of raw data, normalized data, and average data (with stdev error bars):
plot_consolidated_data()


# Change back to directory of imported file and print closing message:
program_close()


# End of program #######################################################################################################

