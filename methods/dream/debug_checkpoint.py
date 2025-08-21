import torch
import glob

# Find checkpoint file
dataset = 'dunnhumby'
fold_id = 0
para_path = glob.glob('./models/'+dataset+'/*')
checkpoint_file = []
for path in para_path:
    path_l = path.split('-')
    if path_l[2] == str(fold_id) and path_l[4] == '1':
        checkpoint_file.append(path)

# Load and inspect checkpoint
checkpoint = torch.load(checkpoint_file[0], map_location=torch.device('cpu'))
print("Checkpoint keys:")
for key, value in checkpoint['state_dict'].items():
    if 'decoder' in key:
        print(f"{key}: {value.shape}")