"""
CAUTION ! this file deletes other files.

Delete all subfolders of the /runs directory that are either empty, or only contain the hparams.json
"""

import os
import shutil

root_folder = 'runs'

def main():
    # Recursively traverse the directory and check for empty subfolders
    for root, dirs, files in os.walk(root_folder, topdown=False):
        for dir_name in dirs:
            dir_path = os.path.join(root, dir_name)
            file_list = os.listdir(dir_path)

            # Check if the folder is empty or contains only 'hparams.json'
            if not file_list or (len(file_list) == 1 and file_list[0] == 'hparams.json'):
                print(f"Deleting empty or 'hparams.json' folder: {dir_path}")
                shutil.rmtree(dir_path)

    print("Cleanup complete.")

if __name__ == '__main__':
    main()