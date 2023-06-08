import os
import sys
import imageio
import chromatinhd as chd

# Get the arguments from the terminal
directory = sys.argv[1]
n_plots = int(sys.argv[2])

# Set up the paths for image directory and output GIF
folder_data_preproc = chd.get_output() / "data" / "hspc"
image_dir = folder_data_preproc / f'plots/{directory}'
output_gif = folder_data_preproc / f'plots/{directory}.gif'

# Create a list of all the PNG images in the directory
images = []
for filename in os.listdir(image_dir)[:n_plots]:
    if filename.endswith('.png'):
        image_path = os.path.join(image_dir, filename)
        images.append(imageio.imread(image_path))

# Create the GIF from the list of images
imageio.mimsave(output_gif, images, fps=5)

print('GIF created!')