import os
import json
import pandas as pd
from shutil import copy2
import random

from options.train_options import TrainOptions

BUILD = False

opt = TrainOptions().parse()

if not os.path.exists('datasets/bald2hairy') or \
        not os.path.exists('datasets/bald2hairy/testA') or \
        not os.path.exists('datasets/bald2hairy/trainA') or \
        not os.path.exists('datasets/bald2hairy/testB') or \
        not os.path.exists('datasets/bald2hairy/trainB'):
    os.makedirs('datasets/bald2hairy/testA')
    os.makedirs('datasets/bald2hairy/trainA')
    os.makedirs('datasets/bald2hairy/testB')
    os.makedirs('datasets/bald2hairy/trainB')


def bald_frame_builder(data):
    col_list = ['File_Name']
    balds = data[(data['Bald'] == 1) & (data['Male'] == 1)]

    df = balds[col_list].replace(to_replace=r'.jpg', value='.png', regex=True)
    df.to_csv("GANiry_bald.csv", index=False)
    df.to_json("GANiry_bald.json", orient="table", index=False)


def hairy_frame_builder(data):
    # col_list = ['File_Name', 'Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Gray_Hair', 'Straight_Hair', 'Wavy_Hair'] # 6 attributes
    col_list = ['File_Name', 'Black_Hair', 'Blond_Hair', 'Straight_Hair', 'Wavy_Hair']  # 4 attributes

    df = data.loc[
        (data['Male'] == 1) & (((data['Black_Hair'] == 1) | (data['Blond_Hair'] == 1)) & (
                    (data['Straight_Hair'] == 1) | (data['Wavy_Hair'] == 1)))]

    df = df[col_list].replace(to_replace=r'.jpg', value='.png', regex=True)
    df.to_csv("GANiry_hairy.csv", index=False)
    df.to_json("GANiry_hairy.json", orient="table", index=False)


def copy_bald_data():
    bald_json = open("GANiry_bald.json", 'r')
    bald_data = json.load(bald_json)['data']
    random.Random(opt.random_seed).shuffle(bald_data)

    for data in bald_data[:opt.test_split]:
        path = os.path.join(opt.celeba_path, data['File_Name'])
        copy2(path, "datasets/bald2hairy/testA/")

    for data in bald_data[opt.test_split:]:
        path = os.path.join(opt.celeba_path, data['File_Name'])
        copy2(path, "datasets/bald2hairy/trainA/")


def copy_hairy_data():
    hairy_json = open("GANiry_hairy.json", 'r')
    hairy_data = json.load(hairy_json)['data']
    random.Random(opt.random_seed).shuffle(hairy_data)

    for data in hairy_data[:opt.test_split]:
        path = os.path.join(opt.celeba_path, data['File_Name'])
        copy2(path, "datasets/bald2hairy/testB/")

    for data in hairy_data[opt.test_split:]:
        path = os.path.join(opt.celeba_path, data['File_Name'])
        copy2(path, "datasets/bald2hairy/trainB/")


if __name__ == "__main__":
    data = pd.read_csv("list_attr_celeba.txt", sep="\s+", skiprows=[0])
    if BUILD:
        bald_frame_builder(data)
        hairy_frame_builder(data)
    else:
        copy_bald_data()
        copy_hairy_data()
