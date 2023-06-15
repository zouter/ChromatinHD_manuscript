#!/bin/bash

# Set your variables
old_folder="/home/wsaelens/projects/chromatinhd/chromatinhd_manuscript/"
new_folder="/home/wsaelens/NAS2/wsaelens/projects/chromatinhd/chromatinhd_manuscript/"
subfolder_path="${old_folder}/$1"

if [ ! -d "${subfolder_path}" ]; then
    echo "Subfolder not found in the old_folder. Exiting..."
    exit 1
fi

# Extract the subfolder name from the path
subfolder_name=$(basename "${subfolder_path}")

# Check if the subfolder already exists in the new_folder
if [ -d "${new_folder}/${subfolder_name}" ]; then
    read -p "The subfolder already exists in the new_folder. Do you want to remove it? (y/n): " answer
    if [ "${answer}" != "y" ]; then
        echo "Aborting the operation."
        exit 1
    fi
    rm -rf "${new_folder}/${subfolder_name}"
fi

# Create the new_folder if it doesn't exist
mkdir -p "${new_folder}"

# Move the subfolder to the new_folder
mv "${subfolder_path}" "${new_folder}/${subfolder_name}"

# Remove the original subfolder path
parent_folder=$(dirname "${subfolder_path}")
rm -rf "${subfolder_path}"

# Create a symlink from the old location to the new location
ln -s "${new_folder}/${subfolder_name}" "${parent_folder}/${subfolder_name}"

echo "Subfolder has been moved and symlinked successfully."