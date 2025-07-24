import os

def merge_txt_modify_first_char(folder1, folder2, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    files1 = set(f for f in os.listdir(folder1) if f.endswith('.txt'))
    files2 = set(f for f in os.listdir(folder2) if f.endswith('.txt'))

    common_files = files1 & files2  # Only process same-name files

    for filename in common_files:
        file1_path = os.path.join(folder1, filename)
        file2_path = os.path.join(folder2, filename)
        output_path = os.path.join(output_folder, filename)

        with open(output_path, 'w') as out:
            # Process folder1 lines, first char -> '0'
            with open(file1_path, 'r') as f1:
                for line in f1:
                    line = line.rstrip('\n')
                    if line:  # Only modify non-empty lines
                        out.write('0' + line[1:] + '\n')
            
            # Process folder2 lines, first char -> '1'
            with open(file2_path, 'r') as f2:
                for line in f2:
                    line = line.rstrip('\n')
                    if line:
                        out.write('1' + line[1:] + '\n')

        print(f"✅ Merged and modified: {filename}")

# 修改成你自己的路径
folder1 = 'dataset_labels/labels_bright_all'
folder2 = 'dataset_labels/labels_gray_all'
output_folder = 'dataset_labels/labels_all'

merge_txt_modify_first_char(folder1, folder2, output_folder)
