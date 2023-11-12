import os
import shutil
import subprocess


def print_red_bold(text):
    print(f'\033[1;31m{text}\033[0m')


def setup_kaggle_credentials(source_file_path, destination_dir_path='/root/.kaggle/'):
    """
    Copies the kaggle.json file to the specified destination directory and sets the required permissions.
    
    Parameters:
    - source_file_path (str): The path to the source kaggle.json file.
    - destination_dir_path (str, optional): The path to the destination directory. Default is '/root/.kaggle/'.
    
    Returns:
    - None
    """
    # Determine the destination file path
    destination_file_path = os.path.join(destination_dir_path, 'kaggle.json')

    # Create the destination directory if it doesn't exist
    os.makedirs(destination_dir_path, exist_ok=True)

    # Copy the kaggle.json file to the destination directory
    shutil.copy(source_file_path, destination_file_path)

    # Check if the file was copied successfully
    if os.path.exists(destination_file_path):
        print_red_bold(f'File {source_file_path} successfully copied to {destination_file_path}')
    else:
        print_red_bold(f'Error copying file {source_file_path} to {destination_file_path}')
        exit(1)  # Exit the script if the file copy fails

    # Issue the chmod 600 command on the kaggle.json file
    chmod_command = f'chmod 600 {destination_file_path}'
    process = subprocess.run(chmod_command.split(), check=True)

    # Check if the command was successful
    if process.returncode == 0:
        print_red_bold(f'\nSuccessfully changed permissions for {destination_file_path}')
    else:
        print_red_bold(f'\nError changing permissions for {destination_file_path}')
        

# Example usage:
# setup_kaggle_credentials('/workspaces/Week 5/kaggle.json')
