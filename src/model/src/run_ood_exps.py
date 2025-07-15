import yaml
import subprocess

ood_subsets = [
    # "english",
    "Image description",
    "Discourse tasks",
    "Word Fluency",
    "spanish",
    "mandarin",
]

# Path to the config file
config_path = '../config/config.yaml'

for ood in ood_subsets:
    # Load current config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Modify the config value
    prev_ood = config['training']['leave_out']
    prev_ood = prev_ood.replace(" ","")
    ood_name = ood.replace(" ","")

    config["paths"]["checkpoints_path"] = config["paths"]["checkpoints_path"].replace(prev_ood,ood_name)
    config["paths"]["final_model"] = config["paths"]["final_model"].replace(prev_ood,ood_name)
    config["paths"]["results_path"] = config["paths"]["results_path"].replace(prev_ood,ood_name)

    config['training']['leave_out'] = ood

    # Save the updated config
    with open(config_path, 'w') as f:
        yaml.dump(config, f)
    
    print(f'Starting script with OOD subset={ood}')

    # Run the target script
    subprocess.run(['python', 'run_train.py'])

    print(f'Finished script with OOD subset={ood}')
