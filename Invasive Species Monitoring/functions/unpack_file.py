import os
import py7zr
import zipfile

def unpack_file(file_path, output_folder):
    """
    Unpacks a .7z or .zip file to an output directory.

    Parameters:
    file_path (str): The path to the file you want to unpack.
    output_folder (str): The directory where the unpacked files will be stored.
    
    Returns:
    None
    """
    # Check if the file exists
    print("Checking if the file exists...")
    if not os.path.exists(file_path):
        print(f"File {file_path} not found.")
        return

    # Check if the output folder exists, if not, create it
    print("Checking if the output folder exists...")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    try:
        # Identify the file type and unpack it
        print("Unpacking the file...")
        if file_path.endswith('.7z'):
            with py7zr.SevenZipFile(file_path, mode='r') as archive:
                archive.extractall(path=output_folder)
            print(f".7z file {file_path} successfully unpacked in {output_folder}")
        elif file_path.endswith('.zip'):
            with zipfile.ZipFile(file_path, 'r') as archive:
                archive.extractall(path=output_folder)
            print(f".zip file {file_path} successfully unpacked in {output_folder}")
        else:
            print(f"Unsupported file format: {file_path}")
    except Exception as e:
        print(f"An error occurred during unpacking: {e}")

# Example of use
# file_path_7z = "/kaggle/input/invasive-species-monitoring/train.7z"  # Replace with the path to your .7z file
# output_folder_7z = "/kaggle/working/train"  # Replace with the directory where you want the .7z files to be unpacked

# file_path_zip = "/kaggle/input/some-zip-file/train.zip"  # Replace with the path to your .zip file
# output_folder_zip = "/kaggle/working/train_zip"  # Replace with the directory where you want the .zip files to be unpacked

# unpack_file(file_path_7z, output_folder_7z)
# unpack_file(file_path_zip, output_folder_zip)
