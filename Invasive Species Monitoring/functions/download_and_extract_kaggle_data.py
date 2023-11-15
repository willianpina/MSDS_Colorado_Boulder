import os
import zipfile
import tarfile
import kaggle 
from tqdm import tqdm

def download_and_extract_kaggle_data(competition_name: str, save_dir: str = 'raw_data') -> None:
    
    #Download, extract Kaggle competition data with progress bars, and then delete the downloaded file.
    
    #Parameters:
    #- competition_name (str): Name of the Kaggle competition.
    #- save_dir (str, optional): Directory where the data should be saved. Defaults to 'raw_data'.
    
    
    # Download the competition data with progress
    kaggle.api.competition_download_cli(competition_name, quiet=False)
    
    # Check the file extension of the downloaded data
    if os.path.exists(f"{competition_name}.zip"):
        file_path = f"{competition_name}.zip"
        is_zip = True
    elif os.path.exists(f"{competition_name}.tar.gz"):
        file_path = f"{competition_name}.tar.gz"
        is_zip = False
    else:
        raise ValueError("Downloaded file is neither .zip nor .tar.gz")
    
    # Ensure the save directory exists
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Extract the data with progress bar
    if is_zip:
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            total_files = len(zip_ref.namelist())
            with tqdm(total=total_files, unit="file") as pbar:
                for member in zip_ref.namelist():
                    zip_ref.extract(member, save_dir)
                    pbar.update(1)
    else:
        with tarfile.open(file_path, 'r:gz') as tar_ref:
            total_files = len(tar_ref.getnames())
            with tqdm(total=total_files, unit="file") as pbar:
                for member in tar_ref.getnames():
                    tar_ref.extract(member, save_dir)
                    pbar.update(1)
    
    try:
        os.remove(file_path)
    except (PermissionError, OSError) as e:
        print(f"Permission denied when trying to delete {file_path}. Please delete it manually.")

    
    print(f"\nData for {competition_name} has been downloaded, extracted to {save_dir}, and the downloaded file has been deleted.")

# Download the data for the Kaggle competition
# download_and_extract_kaggle_data('open-problems-single-cell-perturbations')