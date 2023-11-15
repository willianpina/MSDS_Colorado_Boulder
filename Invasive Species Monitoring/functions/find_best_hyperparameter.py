import os
import json

def find_best_hyperparameter(directories):
    best_auc = 0
    best_hps = None
    best_trial_dir = None

    for dir in directories:
        # Iterate over subdirectories within the main directory
        for subdir in os.listdir(dir):
            trial_dir = os.path.join(dir, subdir)
            trial_file = os.path.join(trial_dir, 'trial.json')
            
            if os.path.exists(trial_file):
                with open(trial_file, 'r') as file:
                    trial_data = json.load(file)
                    try:
                        # Here, you need to adjust for the correct metric name
                        # For example, 'val_auc' or another name as per your case
                        auc = trial_data['metrics']['metrics']['val_auc']['observations'][0]['value'][0]

                        if auc > best_auc:
                            best_auc = auc
                            best_hps = trial_data['hyperparameters']
                            best_trial_dir = trial_dir
                    except KeyError as e:
                        print(f"Key not found in {trial_file}: {e}")
    return best_trial_dir, best_hps, best_auc
