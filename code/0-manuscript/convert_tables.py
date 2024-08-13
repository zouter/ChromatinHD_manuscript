import pandas as pd
import sys
import glob
import shutil
folder = sys.argv[1]

for file in glob.glob(folder + '*.csv'):
    shutil.copy(file, file.replace('.csv', '.xlsx'))