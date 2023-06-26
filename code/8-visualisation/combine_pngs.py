#%%
import os
import sys
import chromatinhd as chd
from PIL import Image

# set folder paths
folder_root = chd.get_output()
folder_data_preproc = folder_root / "data" / "hspc"

# Get the arguments from the terminal
# dir_1 = folder_data_preproc / "plots/likelihood_continuous"
# dir_2 = folder_data_preproc / "plots/likelihood_continuous_128"
# dir_out = folder_data_preproc / "plots/test"
# dir_1 = folder_data_preproc / 'plots' / sys.argv[1]
# dir_2 = folder_data_preproc / 'plots' / sys.argv[2]
# dir_out = folder_data_preproc / 'plots' / sys.argv[3]
# os.makedirs(dir_out, exist_ok=True)

# def combine_pngs(dir1, dir2, output_dir):
#     # Get the list of PNG files in the first directory
#     files1 = [f for f in os.listdir(dir1) if f.endswith(".png")]

#     for file1 in files1:
#         # Check if the corresponding file exists in the second directory
#         file2 = os.path.join(dir2, file1)
#         if os.path.isfile(file2):
#             # Open both images
#             img1 = Image.open(os.path.join(dir1, file1))
#             img2 = Image.open(file2)

#             # Combine the images vertically
#             combined_img = Image.new("RGB", (max(img1.width, img2.width), img1.height + img2.height))
#             combined_img.paste(img1, (0, 0))
#             combined_img.paste(img2, (0, img1.height))

#             # Save the combined image
#             output_file = os.path.join(output_dir, file1)
#             combined_img.save(output_file, "PNG")

#             print(f"Combined {file1} and {file2} into {output_file}")

# combine_pngs(dir_1, dir_2, dir_out)
# ...

def combine_pngs(dir1, dir2, dir3, dir4, dir5, output_dir):
    # Get the list of PNG files in the directories
    files1 = [f for f in os.listdir(dir1) if f.endswith(".png")]
    files2 = [f for f in os.listdir(dir2) if f.endswith(".png")]
    files3 = [f for f in os.listdir(dir3) if f.endswith(".png")]
    files4 = [f for f in os.listdir(dir4) if f.endswith(".png")]
    files5 = [f for f in os.listdir(dir5) if f.endswith(".png")]

    for i, (file1, file2, file3, file4, file5) in enumerate(zip(files1, files2, files3, files4, files5)):
        # Open images from each directory
        img1 = Image.open(os.path.join(dir1, file1))
        img2 = Image.open(os.path.join(dir2, file2))
        img3 = Image.open(os.path.join(dir3, file3))
        img4 = Image.open(os.path.join(dir4, file4))
        img5 = Image.open(os.path.join(dir5, file5))

        # Create a blank canvas for the combined image grid
        nrows = 2
        ncols = 3
        grid_width = max(img1.width, img2.width, img3.width, img4.width, img5.width)
        grid_height = max(img1.height, img2.height, img3.height, img4.height, img5.height)
        combined_img = Image.new("RGB", (grid_width * ncols, grid_height * nrows))

        # Paste images into the combined image grid
        combined_img.paste(img1, (0, 0))
        combined_img.paste(img2, (0, grid_height))
        combined_img.paste(img3, (grid_width, 0))
        combined_img.paste(img4, (grid_width, grid_height))
        combined_img.paste(img5, (grid_width * 2, 0))

        # # Calculate the row and column indices for the current image
        # row_index = i // 3
        # col_index = i % 3

        # # Calculate the position to paste the combined image in the grid
        # paste_x = col_index * grid_width
        # paste_y = row_index * grid_height // 2

        # # Paste the combined image into the grid at the appropriate position
        # combined_img.paste(combined_img, (paste_x, paste_y))

        # Save the combined image
        output_file = os.path.join(output_dir, file1)
        combined_img.save(output_file, "PNG")

        print(f"Combined images from directories to {output_file}")

# Set the directories for the five image sets
dir_1 = folder_data_preproc / 'plots' / sys.argv[1]
dir_2 = folder_data_preproc / 'plots' / sys.argv[2]
dir_3 = folder_data_preproc / 'plots' / sys.argv[3]
dir_4 = folder_data_preproc / 'plots' / sys.argv[4]
dir_5 = folder_data_preproc / 'plots' / sys.argv[5]

# Set the output directory for the combined image grid
dir_out = folder_data_preproc / 'plots' / sys.argv[6]
os.makedirs(dir_out, exist_ok=True)

# Call the function to combine the images into a grid
combine_pngs(dir_1, dir_2, dir_3, dir_4, dir_5, dir_out)
