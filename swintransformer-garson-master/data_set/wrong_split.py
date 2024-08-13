import os
import shutil

src_dir = r"G:\resnet-garson-master\data_set\beans_data\wrong_classified"
dst_dir = r"G:\resnet-garson-master\data_set\beans_data\wrong_train"
pod_dir = os.path.join(dst_dir, "pod")
no_pod_dir = os.path.join(dst_dir, "no pod")

if not os.path.exists(pod_dir):
    os.makedirs(pod_dir)

if not os.path.exists(no_pod_dir):
    os.makedirs(no_pod_dir)

for root, dirs, files in os.walk(src_dir):
    for file in files:
        if file.endswith(".jpg"):
            if "-2.jpg" in file:
                src_file_path = os.path.join(root, file)
                dst_file_path = os.path.join(pod_dir, file)
                shutil.move(src_file_path, dst_file_path)
            elif "-3.jpg" in file:
                src_file_path = os.path.join(root, file)
                dst_file_path = os.path.join(no_pod_dir, file)
                shutil.move(src_file_path, dst_file_path)
