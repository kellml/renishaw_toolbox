# Use this script to consolidate a directory of PL or Raman data to one .txt file
# Assumes imported .txts are in column format (x, y)
# Once data is consolidated, can use txt_data_average_stdev.py to average (and st.devs) data and plot/save as .txt
# Averaged data can then be used for fits of single (x, y) data such as in double_lorentzian_pl.py
# Warning: will overwrite the consolidated._data.txt if re-run on directory


# Import libraries #####################################################################################################


# Import libraries
import os
import glob
import numpy as np
import tkinter as tk
from tkinter import filedialog


# Functions ############################################################################################################


# Functions:
from numpy import ndarray


def current_path():
    print('Original working directory:\n' + os.getcwd() + '\n')


def chosen_path():
    print('Directory changed to:\n' + os.getcwd() + '\n')


def txt_consolidate_begin():
    print('Consolidating all ' + str(len(filenames_list)) + ' .txt files in directory:' + '\n')


def consolidate_end():
    print('\n' + 'All ' + str(len(filenames_list)) + ' imported '
          '.txt files have been consolidated and saved as \'consolidated_data.txt\' in directory: ' + '\n' +
          os.getcwd() + '\n')


# Actual Program #######################################################################################################


# GUI select PL file to import: (eventually do directories of .txt file)
root = tk.Tk()
root.withdraw()  # hides the extra tKinter GUI window besides the dialog box
root.attributes("-topmost", True)  # bring window to front
chosen_directory = filedialog.askdirectory()  # dialog box to choose one file
current_path()

# Change working directory to selected directory:
os.chdir(chosen_directory)  # change directory
chosen_path()


# Create list of .txt files in the directory:
filenames_list = [i for i in glob.glob(f'*.txt')]
txt_consolidate_begin()

# Dummy file import of first .txt file for getting dimensions of empty array:
dummy = np.loadtxt(filenames_list[0])

# Form empty array of desired size:
file_data: ndarray = np.zeros((len(dummy),
                      int(len(filenames_list)) + 1))  # '+1' because first column is wavelength and others are data

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


# Save the consolidated file_data to a .txt file in the same directory:
np.savetxt('consolidated_data.txt', file_data, delimiter='\t')  # tab-delimited .txt

consolidate_end()


# End of program #######################################################################################################

