import os

# Configuration
DATASET_ROOT = "dataset_hitl"
CLASSES = ['0', '1', '2']

def count_files(directory):
    """Counts image files in a directory."""
    if not os.path.exists(directory):
        return 0
    return len([
        f for f in os.listdir(directory)
        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp'))
    ])

def get_counts_dict(base_path):
    """Returns a dict of counts {'0': x, '1': x, '2': x} and the total."""
    counts = {}
    total = 0
    for cls in CLASSES:
        path = os.path.join(base_path, cls)
        c = count_files(path)
        counts[cls] = c
        total += c
    return counts, total

def get_unlabeled_counts(directory):
    """Parses filenames starting with (0), (1), (2) to count classes in a flat folder."""
    counts = {'0': 0, '1': 0, '2': 0}
    total = 0

    if os.path.exists(directory):
        files = [f for f in os.listdir(directory) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        for f in files:
            # Check prefix like "(0)_image.jpg"
            if f.startswith('(0)'):
                counts['0'] += 1
            elif f.startswith('(1)'):
                counts['1'] += 1
            elif f.startswith('(2)'):
                counts['2'] += 1
            total += 1

    return counts, total

def print_section(title, counts_dict, total):
    """Helper to format the output line nicely."""
    details = ", ".join([f"{k}: {v:>3}" for k, v in counts_dict.items()])
    print(f"{title:<10} | {details}  (Total: {total})")

# --- Main Execution ---
print(f"\n{'='*60}")
print(f"DATASET DISTRIBUTION SUMMARY: {DATASET_ROOT}")
print(f"{'='*60}\n")

# 1. Labeled Data (Train, Val, Test are all here now)
print("LABELED:")
print("-" * 60)

# Train
train_counts, train_total = get_counts_dict(os.path.join(DATASET_ROOT, "labeled", "train"))
print_section("Train", train_counts, train_total)

# Val
val_counts, val_total = get_counts_dict(os.path.join(DATASET_ROOT, "labeled", "val"))
print_section("Val", val_counts, val_total)

# Test (Updated Path)
test_counts, test_total = get_counts_dict(os.path.join(DATASET_ROOT, "labeled", "test"))
print_section("Test", test_counts, test_total)

print("\n")

# 2. Unlabeled Data
print("UNLABELED POOL:")
print("-" * 60)
unlabeled_path = os.path.join(DATASET_ROOT, "unlabeled")
pool_counts, pool_total = get_unlabeled_counts(unlabeled_path)
print_section("Pool Size", pool_counts, pool_total)

print(f"\n{'='*60}\n")

# ============================================================
# DATASET DISTRIBUTION SUMMARY: dataset_hitl
# ============================================================

# LABELED:
# ------------------------------------------------------------
# Train      | 0:  31, 1:  32, 2:  32  (Total: 95)
# Val        | 0:   8, 1:   9, 2:   8  (Total: 25)
# Test       | 0:  52, 1:  55, 2:  54  (Total: 161)


# UNLABELED POOL:
# ------------------------------------------------------------
# Pool Size  | 0: 171, 1: 182, 2: 177  (Total: 530)

# ============================================================