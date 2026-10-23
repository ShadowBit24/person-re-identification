import os
from shutil import copyfile
import argparse
import shutil

download_path = "raw_dataset/cuhk03-np"

parser = argparse.ArgumentParser(description='prepare')
parser.add_argument('--cuhk03', action='store_true', help='prepare dataset CUHK03-NP')
opt = parser.parse_args()

if not os.path.isdir(download_path):
    print('please change the download_path')

if opt.cuhk03:
    save_path = "./dataset/cuhk03-np_prepare/"
else:
    print('Please specify the dataset to prepare using --cuhk03 flag')

if not os.path.exists(save_path):
    os.makedirs(save_path)

# -----------------------------------------
# query
query_path = download_path + '/detected/query'
query_save_path = save_path + '/query'
if not os.path.exists(query_save_path):
    os.makedirs(query_save_path)

for root, dirs, files in os.walk(query_path, topdown=True):
    for name in files:
        if not name.endswith('-checkpoint.png'):
            src_path = query_path + '/' + name
            dst_path = query_save_path + '/' + name
            copyfile(src_path, dst_path)

# -----------------------------------------
# gallery
gallery_path = download_path + '/detected/bounding_box_test'
gallery_save_path = save_path + '/gallery'
if not os.path.exists(gallery_save_path):
    os.makedirs(gallery_save_path)

for root, dirs, files in os.walk(gallery_path, topdown=True):
    for name in files:
        if not name.endswith('-checkpoint.png'):
            src_path = gallery_path + '/' + name
            dst_path = gallery_save_path + '/' + name
            copyfile(src_path, dst_path)

# ---------------------------------------
# train_all
train_path = download_path + '/detected/bounding_box_train'
train_save_path = save_path + '/train_all'
if not os.path.exists(train_save_path):
    os.makedirs(train_save_path)

for root, dirs, files in os.walk(train_path, topdown=True):
    for name in files:
        if not name.endswith('-checkpoint.png'):
            src_path = train_path + '/' + name
            dst_path = train_save_path + '/' + name
            copyfile(src_path, dst_path)

# ---------------------------------------
# train_val
train_path = download_path + '/detected/bounding_box_train'
train_save_path = save_path + '/train'
val_save_path = save_path + '/val'
if not os.path.exists(train_save_path):
    os.makedirs(train_save_path)
    os.makedirs(val_save_path)

for root, dirs, files in os.walk(train_path, topdown=True):
    for name in files:
        if not name.endswith('-checkpoint.png'):
            src_path = train_path + '/' + name
            if 's1' in name:
                dst_path = val_save_path + '/' + name
            else:
                dst_path = train_save_path + '/' + name
            copyfile(src_path, dst_path)

print("Done!")
