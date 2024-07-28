import numpy as np
import pandas as pd
import argparse
import time
import os
import shutil

'''
Created on:
    2024-06-07 15:00:00
Author: 
    Teslim  Olayiwola 
Description: 
    Utility functions to split the numpy deepmd-kit format into training and test sets

It is important to state the following:
    1. The box.npy stored the cell data from .cell file
    2. The energy.npy stored the energy data from .en file
    3. The coord.npy stored the coordinates data from .xyz file
    4. The force.npy stored the force data from .for file

The goal is to convert the convert the arrays of deepmd .npy files to training and test sets

'''

def main(args):

    # check time
    start_time = time.time()

    # use if the files are in the same path and thus their absolute path is not provided 
    if args.PATH is not None: 
        args.FILES = [f"{args.PATH}/{file}" for file in args.CONFIGURATIONS]

    # check if the CONFIGURATIONS exists
    for config in args.FILES:
        assert os.path.exists(config), f"Configuration {config} does not exist"

    # in each configuration, there box.npy, energy.npy, coord.npy, force.npy in set.000, sample 80% for training and 20% for testing and save in train and test folders
    for config in args.FILES:
        print(f'+======= Processing configuration {config} =======+')
        # check if the set.000 exists
        assert os.path.exists(f"{config}/{args.FOLDER}/set.000"), f"Configuration {config} does not have set.000 folder"
        # check if the box.npy, energy.npy, coord.npy, force.npy exists
        for file in ["box.npy", "energy.npy", "coord.npy", "force.npy"]:
            assert os.path.exists(f"{config}/{args.FOLDER}/set.000/{file}"), f"Configuration {config} does not have {file} file"

        # load the box, energy, coord, force data
        box = np.load(f"{config}/{args.FOLDER}/set.000/box.npy")
        energy = np.load(f"{config}/{args.FOLDER}/set.000/energy.npy")
        coord = np.load(f"{config}/{args.FOLDER}/set.000/coord.npy")
        force = np.load(f"{config}/{args.FOLDER}/set.000/force.npy")

        # shuffle the data
        if args.SHUFFLE:
            np.random.seed(args.SEED)
            idx = np.random.permutation(len(box))
            box = box[idx]
            energy = energy[idx]
            coord = coord[idx]
            force = force[idx]

        # split the data into training and test sets
        nfiles = len(box)
        ntrain = int(args.TRAIN_SIZE*nfiles)

        # consistency check
        assert len(box) == len(energy) == len(coord) == len(force), "Number of steps mismatch." # expect the same number of steps
        assert coord.shape[1] == force.shape[1], "Coordinates data mismatch."
        print(f'Number of steps and coordinates matches')

        # split into training and test data
        box_train = box[:ntrain]
        energy_train = energy[:ntrain]
        coord_train = coord[:ntrain]
        force_train = force[:ntrain]

        box_test = box[ntrain:]
        energy_test = energy[ntrain:]
        coord_test = coord[ntrain:]
        force_test = force[ntrain:]

        print(f'Data split into training and test sets')
        # create the train and test folders
        if not os.path.exists(f"{config}/train"):
            os.mkdir(f"{config}/train")
        if not os.path.exists(f"{config}/test"):
            os.mkdir(f"{config}/test")

        # copy type.raw and type_map.raw in {config}/{args.FOLDER} to {config}/train and {config}/test
        if os.path.exists(f"{config}/{args.FOLDER}/type.raw"):
            shutil.copy(f"{config}/{args.FOLDER}/type.raw", f"{config}/train")
            shutil.copy(f"{config}/{args.FOLDER}/type.raw", f"{config}/test")
        else:
            print(f"type.raw file not found in the folder {config}/{args.FOLDER}")

        if os.path.exists(f"{config}/{args.FOLDER}/type_map.raw"):
            shutil.copy(f"{config}/{args.FOLDER}/type_map.raw", f"{config}/train")
            shutil.copy(f"{config}/{args.FOLDER}/type_map.raw", f"{config}/test")
        else:
            print(f"type_map.raw file not found in the folder {config}/{args.FOLDER}")

        
        # ********************************** train set **********************************
        # split the datas (cell_data, energy_data, ) and save them in set.000, set.010, set.020, ...
        if args.SPLIT_TRAIN_DATA:
            nline_per_set_train = int(len(box_train) // args.SPLIT_COUNT)
            for i in range(args.SPLIT_COUNT):

                if not os.path.exists(f"{config}/train/set.{i:0>3}"):
                    os.mkdir(f"{config}/train/set.{i:0>3}")

                iset_box_train = box_train[i*nline_per_set_train:(i+1)*nline_per_set_train]
                np.save(f"{config}/train/set.{i:0>3}/box.npy", iset_box_train.astype(np.float32))

                iset_energy_train = energy_train[i*nline_per_set_train:(i+1)*nline_per_set_train]
                np.save(f"{config}/train/set.{i:0>3}/energy.npy", iset_energy_train.astype(np.float32))

                iset_coord_train = coord_train[i*nline_per_set_train:(i+1)*nline_per_set_train]
                np.save(f"{config}/train/set.{i:0>3}/coord.npy", iset_coord_train.astype(np.float32))

                iset_force_train = force_train[i*nline_per_set_train:(i+1)*nline_per_set_train]
                np.save(f"{config}/train/set.{i:0>3}/force.npy", iset_force_train.astype(np.float32))

                # print
                print(f'Saved set.{i:0>3} for training sets')

        else:
            if not os.path.exists(f'{config}/train/set.000'):
                os.makedirs(f'{config}/train/set.000')

            np.save(f"{config}/train/set.000/box.npy", box_train)
            np.save(f"{config}/train/set.000/energy.npy", energy_train)
            np.save(f"{config}/train/set.000/coord.npy", coord_train)
            np.save(f"{config}/train/set.000/force.npy", force_train)

            print(f'Saved set.000 for training sets')


        # ********************************** test set **********************************
        if args.SPLIT_TEST_DATA:
            nline_per_set_test = int(len(box_test) // args.SPLIT_COUNT)
            for i in range(args.SPLIT_COUNT):

                if not os.path.exists(f"{config}/test/set.{i:0>3}"):
                    os.mkdir(f"{config}/test/set.{i:0>3}")

                iset_box_test = box_test[i*nline_per_set_test:(i+1)*nline_per_set_test]
                np.save(f"{config}/test/set.{i:0>3}/box.npy", iset_box_test.astype(np.float32))

                iset_energy_test = energy_test[i*nline_per_set_test:(i+1)*nline_per_set_test]
                np.save(f"{config}/test/set.{i:0>3}/energy.npy", iset_energy_test.astype(np.float32))

                iset_coord_test = coord_test[i*nline_per_set_test:(i+1)*nline_per_set_test]
                np.save(f"{config}/test/set.{i:0>3}/coord.npy", iset_coord_test.astype(np.float32))

                iset_force_test = force_test[i*nline_per_set_test:(i+1)*nline_per_set_test]
                np.save(f"{config}/test/set.{i:0>3}/force.npy", iset_force_test.astype(np.float32))

                # print
                print(f'Saved set.{i:0>3} for test sets')

        else:
            if not os.path.exists(f'{config}/test/set.000'):
                os.makedirs(f'{config}/test/set.000')

            np.save(f"{config}/test/set.000/box.npy", box_test)
            np.save(f"{config}/test/set.000/energy.npy", energy_test)
            np.save(f"{config}/test/set.000/coord.npy", coord_test)
            np.save(f"{config}/test/set.000/force.npy", force_test)

            print(f'Saved set.000 for test sets')


    print(f'Execution time: {time.time() - start_time} seconds')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="convert cp2k aimd files to deepmd-kit format")
    parser.add_argument("--FILES", type=str, nargs='+', default=['AEM2Cl6Wat900_NVT-04000-08000', 'AEM2Cl6Wat900_NVT-08000-10000'], help="LIST of NCONFIGURATIONS")
    parser.add_argument("--PATH", type=str, default=None, help="parent folder for the CONFIGURATIONS")
    parser.add_argument("--FOLDER", type=str, default=None, help="folder holding the files of each CONFIGURATIONS")
    parser.add_argument("--TRAIN_SIZE", type=float, default=0.8, help="number of sets to split the data")
    parser.add_argument("--SHUFFLE", action="store_true", help="shuffle the data before splitting")
    parser.add_argument("--SEED", type=int, default=42, help="random seed for shuffling")
    parser.add_argument("--SPLIT_TRAIN_DATA", action="store_true", help="split the training data into sets")
    parser.add_argument("--SPLIT_TEST_DATA", action="store_true", help="split the test data into sets")
    parser.add_argument("--SPLIT_COUNT", type=int, default=5, help="number of sets to split the data")
    args = parser.parse_args()
    main(args)


# python split_train_test.py  --PATH /work/tolayi1/ml4forcefield/basic/  --CONFIGURATIONS AEM2Cl6Wat900_NVT-04000-08000 AEM2Cl6Wat900_NVT-08000-10000 --TRAIN_SIZE 0.8 --SHUFFLE True --SEED 42 --SPLIT_TRAIN_DATA True --SPLIT_TEST_DATA False --SPLIT_COUNT 5