#%%
import os
import chromatinhd as chd

from PIL import Image

# set folder paths
folder_root = chd.get_output()
folder_data_preproc = folder_root / "data" / "hspc"

dir_quant = folder_data_preproc / "plots/evaluate_pseudo"
dir_conti = folder_data_preproc / "plots/cut_sites_evaluate_pseudo_continuous"
dir_out = folder_data_preproc / "plots/test"
os.makedirs(dir_out, exist_ok=True)

def combine_pngs(dir1, dir2, output_dir):
    # Get the list of PNG files in the first directory
    files1 = [f for f in os.listdir(dir1) if f.endswith(".png")]
    print(files1)

    for file1 in files1:
        # Check if the corresponding file exists in the second directory
        file2 = os.path.join(dir2, file1)
        if os.path.isfile(file2):
            # Open both images
            img1 = Image.open(os.path.join(dir1, file1))
            img2 = Image.open(file2)

            # Combine the images vertically
            combined_img = Image.new("RGB", (max(img1.width, img2.width), img1.height + img2.height))
            combined_img.paste(img1, (0, 0))
            combined_img.paste(img2, (0, img1.height))

            # Save the combined image
            output_file = os.path.join(output_dir, file1)
            combined_img.save(output_file, "PNG")

            print(f"Combined {file1} and {file2} into {output_file}")

combine_pngs(dir_conti, dir_quant, dir_out)

 # %%
