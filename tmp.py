import os
import shutil
import argparse
import random
import cv2  # OpenCV for reading/writing images
import albumentations as A # The augmentation library
import numpy as np
from tqdm import tqdm # A progress bar for long loops

# --- Configuration ---

# 1. Define the fixed IDs for your 44 original background images
# These sets now control which ORIGINAL images are used for each subset.
# This ensures your test set is always generated from a completely separate pool of images.
source_train_ids = {5, 16, 13, 32, 7, 23, 29, 18, 22, 40, 27, 14, 33, 20, 25, 39, 36, 34, 42, 1, 10, 37, 3, 6, 9, 28}
source_test_ids = {17, 8, 30, 43, 24, 4, 26, 38} # This is your "Golden Test Set" source pool
source_valid_ids = {2, 31, 12, 11, 15, 19, 21, 41, 35, 44}

# 2. Define the desired number of images to GENERATE for each class in each subset
# This is where you control the final dataset size.
# Example: 1000 images for 'A', 1000 for 'B', 1000 for 'background' in the training set.
SAMPLES_PER_CLASS = {
    "train": 1000,
    "valid": 200,
    "test": 200
}

# --- Augmentation Pipeline ---

# 3. Define the data augmentation pipeline using Albumentations.
# This is a powerful and flexible way to create diverse training data.
# You can easily comment out or add more augmentations here.
transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.RandomBrightnessContrast(p=0.3),
    A.GaussNoise(p=0.2),
    A.GaussianBlur(p=0.2),
    # You can add more, like A.RandomResizedCrop, A.ShiftScaleRotate, etc.
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

# --- Helper Functions ---

def add_synthetic_defect(image, defect_type='A'):
    """
    Placeholder function for adding your synthetic defects.
    You need to REPLACE THIS with your own logic.
    
    Args:
        image (np.array): The input image as a NumPy array.
        defect_type (str): 'A' or 'B'.
        
    Returns:
        tuple: A tuple containing:
            - modified_image (np.array): The image with the defect.
            - bboxes (list): A list of bounding boxes in YOLO format. 
                              e.g., [[x_center, y_center, width, height], ...]
            - class_labels (list): A list of class labels corresponding to bboxes.
                                   e.g., [0] for class 'A', [1] for class 'B'.
    """
    # ===================================================================
    # TODO: REPLACE THIS EXAMPLE LOGIC WITH YOUR ACTUAL CODE
    # This example just draws a rectangle and returns its coordinates.
    h, w, _ = image.shape
    modified_image = image.copy()
    bboxes = []
    class_labels = []

    if defect_type == 'A':
        # Example: Add a white square for defect 'A'
        x1, y1 = random.randint(0, w-50), random.randint(0, h-50)
        x2, y2 = x1 + 30, y1 + 30
        cv2.rectangle(modified_image, (x1, y1), (x2, y2), (255, 255, 255), -1)
        
        # Convert to YOLO format: [x_center, y_center, width, height]
        x_center = (x1 + x2) / 2 / w
        y_center = (y1 + y2) / 2 / h
        width = (x2 - x1) / w
        height = (y2 - y1) / h
        bboxes.append([x_center, y_center, width, height])
        class_labels.append(0) # Class 0 for 'A'

    elif defect_type == 'B':
        # Example: Add a gray circle for defect 'B'
        cx, cy = random.randint(50, w-50), random.randint(50, h-50)
        radius = 20
        cv2.circle(modified_image, (cx, cy), radius, (128, 128, 128), -1)

        # Convert to YOLO format
        x_center = cx / w
        y_center = cy / h
        width = (radius * 2) / w
        height = (radius * 2) / h
        bboxes.append([x_center, y_center, width, height])
        class_labels.append(1) # Class 1 for 'B'

    return modified_image, bboxes, class_labels
    # ===================================================================


def save_yolo_label(label_path, bboxes, class_labels):
    """Saves bounding boxes and labels to a .txt file in YOLO format."""
    with open(label_path, 'w') as f:
        for bbox, label in zip(bboxes, class_labels):
            f.write(f"{label} {' '.join(map(str, bbox))}\n")


# --- Main Logic ---

def generate_and_split_dataset(source_dir, output_dir):
    """Main function to generate the complete, augmented, and balanced dataset."""
    
    # Create all necessary output directories
    for subset in ["train", "valid", "test"]:
        os.makedirs(os.path.join(output_dir, 'images', subset), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'labels', subset), exist_ok=True)
        
    # Find all original background images
    all_source_files = [f for f in os.listdir(source_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    if len(all_source_files) == 0:
        print(f"❌ Error: No source images found in {source_dir}")
        return

    # Split the original files into three source pools based on their IDs
    source_pools = { "train": [], "valid": [], "test": [] }
    for file in all_source_files:
        base_name = os.path.splitext(file)[0]
        try:
            num = int(base_name)
            if num in source_train_ids:
                source_pools["train"].append(file)
            elif num in source_valid_ids:
                source_pools["valid"].append(file)
            elif num in source_test_ids:
                source_pools["test"].append(file)
        except ValueError:
            print(f"⚠️ Warning: Could not parse ID from '{file}', skipping.")
            continue
            
    print("✅ Source images have been split into pools:")
    print(f"   - Training pool: {len(source_pools['train'])} images")
    print(f"   - Validation pool: {len(source_pools['valid'])} images")
    print(f"   - Golden Test pool: {len(source_pools['test'])} images")

    # Generate the dataset
    for subset in ["train", "valid", "test"]:
        print(f"\nGenerating '{subset}' set...")
        
        image_out_dir = os.path.join(output_dir, 'images', subset)
        label_out_dir = os.path.join(output_dir, 'labels', subset)
        
        source_files_pool = source_pools[subset]
        if not source_files_pool:
            print(f"⚠️ Warning: No source files for '{subset}' subset. Skipping generation.")
            continue

        num_samples_per_class = SAMPLES_PER_CLASS[subset]
        
        # The main generation loop
        for i in tqdm(range(num_samples_per_class * 3)): # *3 for 3 classes
            # Pick a random source image from the correct pool
            source_file = random.choice(source_files_pool)
            source_path = os.path.join(source_dir, source_file)
            image = cv2.imread(source_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # Albumentations uses RGB

            # Decide which class to generate for this iteration to ensure balance
            class_type = i % 3 
            
            new_bboxes = []
            new_labels = []
            
            # --- Generate image and label based on class type ---
            if class_type == 0: # Background
                # For background, there are no defects
                transformed = transform(image=image, bboxes=[], class_labels=[])
                transformed_image = transformed['image']
                
            elif class_type == 1: # Defect A
                modified_image, bboxes, class_labels = add_synthetic_defect(image, 'A')
                transformed = transform(image=modified_image, bboxes=bboxes, class_labels=class_labels)
                transformed_image = transformed['image']
                new_bboxes = transformed['bboxes']
                new_labels = transformed['class_labels']

            else: # class_type == 2 -> Defect B
                modified_image, bboxes, class_labels = add_synthetic_defect(image, 'B')
                transformed = transform(image=modified_image, bboxes=bboxes, class_labels=class_labels)
                transformed_image = transformed['image']
                new_bboxes = transformed['bboxes']
                new_labels = transformed['class_labels']

            # --- Save the new image and label ---
            base_filename = os.path.splitext(source_file)[0]
            # Create a unique new filename to avoid overwriting
            new_filename_base = f"{base_filename}_aug_{i}"
            
            # Save image
            img_to_save = cv2.cvtColor(transformed_image, cv2.COLOR_RGB2BGR)
            image_save_path = os.path.join(image_out_dir, f"{new_filename_base}.png")
            cv2.imwrite(image_save_path, img_to_save)

            # Save corresponding label file (even if it's empty for background)
            label_save_path = os.path.join(label_out_dir, f"{new_filename_base}.txt")
            save_yolo_label(label_save_path, new_bboxes, new_labels)

    print("\n✅ Dataset generation complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate and split an augmented dataset from source background images.")
    parser.add_argument('--source_dir', type=str, required=True, help='Path to the directory with your original 44 background images.')
    parser.add_argument('--output_dir', type=str, required=True, help='Path to the root directory where the new dataset will be saved (e.g., "YOLO_dataset").')
    args = parser.parse_args()

    generate_and_split_dataset(args.source_dir, args.output_dir)