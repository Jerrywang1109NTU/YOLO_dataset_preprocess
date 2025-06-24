import subprocess

print("Step 1: 处理 label")
subprocess.run(["python3", "parity_labels_combine.py"])

print("Step 2: 处理 image")
subprocess.run(["python3", "parity_images_combine.py"])

print("Step 3: 划分数据集")
subprocess.run(["python3", "parity_data_split.py"])
