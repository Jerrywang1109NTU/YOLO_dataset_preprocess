import numpy as np
# these parameters are for images modification settings
replacement_right = np.array([
    [227, 215, 207, 198, 207],
    [199, 207, 199, 199, 215],
    [215, 199, 207, 215, 199],
    [215, 199, 208, 215, 227],
    [208, 199, 216, 215, 215]
], dtype=np.uint8)
replacement_right = np.tile(replacement_right, (2, 2))
replacement_left = np.array([
    [208, 200, 199, 208, 199],
    [208, 183, 208, 183, 199],
    [200, 208, 216, 208, 208],
    [216, 200, 208, 200, 183],
    [208, 200, 216, 200, 208]
], dtype=np.uint8)
replacement_left = np.tile(replacement_left, (2, 2))
replacement_right = np.stack([replacement_right]*3, axis=-1)
replacement_left = np.stack([replacement_left]*3, axis=-1)
w_b = 8
h_b = 10
w_g = 8
h_g = 10
# these parameters are for random labels settings
gray_count = 8
bright_count = 8
min_manhattan_distance = 0.10
num_combinations = 40
w_l = 0.02  # default width for labels
h_l = 0.05  # default width and height for labels
# saving files for YOLO format dataset
YOLO_save_dir_image = "dataset_0_bk/YOLO_data_6_24_0/images"
YOLO_save_dir_label = "dataset_0_bk/YOLO_data_6_24_0/labels"