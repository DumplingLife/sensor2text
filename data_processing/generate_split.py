import os
import random
import csv

base_dir = "data/imagebind_targets"

def write_csv(file_list, csv_writer):
    for file_path in file_list:
        csv_writer.writerow([file_path])

"""
def random_split_8():
    num_train_intervals = int(16789 * 0.75 * 1/8) # 16789 total
    train_file, test_file = "data/train_random_8.csv", "data/test_random_8.csv"
    assert not os.path.exists(train_file)
    assert not os.path.exists(test_file)
    
    file_list = []
    for subdir in os.listdir(base_dir):
        file_list += [os.path.join(subdir, file) for file in os.listdir(os.path.join(base_dir, subdir)) if file.endswith(".npy")]
    
    possible_start_idx_list = []
    for subdir in os.listdir(base_dir):
        for file in os.listdir(os.path.join(base_dir, subdir)):
            if file.endswith(".npy"):
                idx = int(file.split(".")[0])
                if idx % 8 == 0:
                    possible_start_idx_list.append((subdir, idx))
    
    train_interval_starts = random.sample(possible_start_idx_list, num_train_intervals)
    train_files = []
    for subdir, start_idx in train_interval_starts:
        for idx in range(start_idx, start_idx+8):
            if f"{idx:03d}.npy" in os.listdir(os.path.join(base_dir, subdir)):
                train_files.append(f"{subdir}/{idx:03d}.npy")

    train_files.sort()
    test_files = sorted(list(set(file_list) - set(train_files)))

    with open(train_file, "w", newline="") as train_csv, open(test_file, "w", newline="") as test_csv:
        train_writer, test_writer = csv.writer(train_csv), csv.writer(test_csv)
        for file_path in train_files:
            train_writer.writerow([file_path])
        for file_path in test_files:
            test_writer.writerow([file_path])
"""

def random_split_8():
    num_train_intervals = int(16789 * 0.70 * 1/8)  # 70% for training
    num_val_intervals = int(16789 * 0.15 * 1/8)    # 15% for validation
    
    train_file = "data/train_random_8.csv"
    val_file = "data/val_random_8.csv"
    test_file = "data/test_random_8.csv"
    
    assert not os.path.exists(train_file)
    assert not os.path.exists(val_file)
    assert not os.path.exists(test_file)
    
    file_list = []
    for subdir in os.listdir(base_dir):
        file_list += [os.path.join(subdir, file) for file in os.listdir(os.path.join(base_dir, subdir)) if file.endswith(".npy")]
    
    possible_start_idx_list = []
    for subdir in os.listdir(base_dir):
        for file in os.listdir(os.path.join(base_dir, subdir)):
            if file.endswith(".npy"):
                idx = int(file.split(".")[0])
                if idx % 8 == 0:
                    possible_start_idx_list.append((subdir, idx))
    
    train_interval_starts = random.sample(possible_start_idx_list, num_train_intervals)
    remaining_intervals = list(set(possible_start_idx_list) - set(train_interval_starts))
    val_interval_starts = random.sample(remaining_intervals, num_val_intervals)
    
    train_files = []
    for subdir, start_idx in train_interval_starts:
        for idx in range(start_idx, start_idx+8):
            if f"{idx:03d}.npy" in os.listdir(os.path.join(base_dir, subdir)):
                train_files.append(f"{subdir}/{idx:03d}.npy")
    
    val_files = []
    for subdir, start_idx in val_interval_starts:
        for idx in range(start_idx, start_idx+8):
            if f"{idx:03d}.npy" in os.listdir(os.path.join(base_dir, subdir)):
                val_files.append(f"{subdir}/{idx:03d}.npy")
    
    test_files = sorted(list(set(file_list) - set(train_files) - set(val_files)))
    
    train_files.sort()
    val_files.sort()
    
    with open(train_file, "w", newline="") as train_csv, open(val_file, "w", newline="") as val_csv, open(test_file, "w", newline="") as test_csv:
        train_writer = csv.writer(train_csv)
        val_writer = csv.writer(val_csv)
        test_writer = csv.writer(test_csv)
        
        for file_path in train_files:
            train_writer.writerow([file_path])
        for file_path in val_files:
            val_writer.writerow([file_path])
        for file_path in test_files:
            test_writer.writerow([file_path])

def subject_split():
    train_file, test_file = "data/train.csv", "data/test.csv"
    assert not os.path.exists(train_file)
    assert not os.path.exists(test_file)
    train_subjects = ["S00", "S02_1", "S02_2", "S02_3", "S03", "S04", "S05", "S06", "S07"]
    
    train_files = []
    test_files = []    
    for subdir in os.listdir(base_dir):
        file_paths = [os.path.join(subdir, file) for file in os.listdir(os.path.join(base_dir, subdir)) if file.endswith(".npy")]
        if subdir in train_subjects:
            train_files += file_paths
        else:
            test_files += file_paths 
    train_files.sort()
    test_files.sort()

    with open(train_file, "w", newline="") as train_csv, open(test_file, "w", newline="") as test_csv:
        train_writer, test_writer = csv.writer(train_csv), csv.writer(test_csv)
        for file_path in train_files:
            train_writer.writerow([file_path])
        for file_path in test_files:
            test_writer.writerow([file_path])

# subject_split()
random_split_8()