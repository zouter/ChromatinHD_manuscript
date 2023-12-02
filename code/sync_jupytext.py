import os
import glob
import shlex

files = []
for file in glob.glob("./code/**/*.ipynb", recursive=True):
    if file.endswith(".ipynb"):
        files.append(shlex.quote(file))
os.system(f"jupytext --sync {' '.join(files)}")
