import os
import json
from tqdm import tqdm

data_dir = "/home/local/ASUAD/agupt374/research_directory/Playground/Kinematic_Modelling/Sequential_Joint_Estimation/data/data_sim"
labels_path = "/home/local/ASUAD/agupt374/research_directory/Playground/Kinematic_Modelling/Sequential_Joint_Estimation/data/labels_sim/labels_sim.json"
missing_file_log = os.path.join(data_dir, "missing_labels.txt")

# Load labels
with open(labels_path, 'r') as f:
    labels = json.load(f)

# Track missing folders
missing = []
folders = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]


for folder_name in tqdm(folders, desc="Processing folders"):
    folder_path = os.path.join(data_dir, folder_name)
    if not os.path.isdir(folder_path):
        continue

    if folder_name not in labels:
        missing.append(folder_name)
        print(f"Missing label for folder: {folder_name}")
        continue

    obj = labels[folder_name]
    joint_count = obj["dof"] + 1
    joint_params = []

    for j in range(1, 4):
        axis = obj.get(f"joint {j} axis", [-1, -1, -1])
        position = obj.get(f"joint {j} position", [-1, -1, -1])
        joint_type = obj.get(f"joint {j} type", -1)  # 0: rotation, 1: translation
        joint_params.append(axis + position + [joint_type])

    joint_deltas = []
    for t in range(12):
        step_deltas = []
        for j in range(1, 4):
            q = obj.get(f"joint {j} q", [-1] * 12)
            step_deltas.append(q[t])
        joint_deltas.append(step_deltas)

    meta = {
        "joint_count": joint_count,
        "joint_params": joint_params,  # shape (3, 7)
        "joint_deltas": joint_deltas   # shape (12, 3)
    }

    # Save meta.json inside the folder
    meta_path = os.path.join(folder_path, "meta.json")
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)

    # print(f"Processed folder: {folder_name} with joint_count: {joint_count}")
    # break

# Log missing folders
if missing:
    with open(missing_file_log, 'w') as f:
        for name in missing:
            f.write(name + '\n')
    print(f"Missing {len(missing)} folders. Logged to {missing_file_log}")
else:
    print("All folders processed successfully.")
