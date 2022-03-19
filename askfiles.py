# Import and plot many .txt files of presumably (x, avg_y, stdev_y) format from a single directory.
# Files are imported in the order that they are listed in the dialog box upon selection.
# All selected files will be plotted.


import tkinter as tk
from tkinter import filedialog
import numpy as np
from numpy import ndarray
import matplotlib.pyplot as plt


# Program:

# GUI select PL file to import:
tk_frame = tk.Tk()  # create instance of Tkinter frame
tk_frame.attributes("-topmost", True)  # bring window to front
tk_frame.withdraw()  # hides the extra tKinter GUI window besides the dialog box
test = filedialog.askopenfilenames(defaultextension='.txt')  # dialog box to choose one file


for i in range(0, len(test)):
    import_data: ndarray = np.loadtxt(test[i])
    # (x, y) dataset
    if import_data.shape[1] == 2:
        plt.plot(import_data[:, 0], import_data[:, 1], label=('Data ' + str(i + 1)))  # so legend starts at #1
    # (x, y_avg, y_stdev) dataset
    elif import_data.shape[1] == 3:
        plt.plot(import_data[:, 0], import_data[:, 1], label=('Data ' + str(i + 1)))  # so legend starts at #1
        plt.fill_between(import_data[:, 0], np.add(import_data[:, 1], import_data[:, 2]),
                         np.subtract(import_data[:, 1], import_data[:, 2]), alpha=0.3)
    else:
        print('Dimensions of selected .txt file are not in format (x, y) or (x, y_avg, y_stdev).' + '\n' +
              'Ensure that selected files are of desired format (2- or 3-column).' + '\n' +
              'Program will terminate.')

plt.legend(loc='upper left')
plt.show()
