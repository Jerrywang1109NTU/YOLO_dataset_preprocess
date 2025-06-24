import subprocess

print("Step 1: 处理 label")
subprocess.run(["python3", "labels_combine_random.py"])

print("Step 2: 处理 image")
subprocess.run(["python3", "images_combine_random.py"])

print("Step 3: 划分数据集")
subprocess.run(["python3", "data_split_random.py"])
