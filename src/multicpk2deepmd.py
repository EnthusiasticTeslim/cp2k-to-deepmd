import numpy as np
import pandas as pd
import argparse
import time
import os
from utils import read_cell_or_energy_data, read_forces_data, read_position_data

'''
Created on:
    2024-06-07 15:00:00
Author: 
    Teslim  Olayiwola 
Description: 
    Utility functions for convert cp2k data to deepmd-kit format

It is important to state the following:
    1. The cell data is stored in a .cell file
    2. The energy data is stored in a .en file
    3. The coordinates data is stored in a .xyz file
    4. The force data is stored in a .for file

The goal is to convert the data to the deepmd-kit format which is a .npy file format
    1. The cell data is stored in a coord.npy file has shape (NSTEPS, 3*3)
    2. The energy data is stored in a energy.npy file has shape (NSTEPS,)
    3. The coordinates data is stored in a coord.npy file has shape (NSTEPS, NATOMS*3)
    4. The force data is stored in a force.npy file has shape (NSTEPS, NATOMS*3)

type.raw
type_map.raw
set.000/coord.npy
set.000/energy.npy
set.000/force.npy

NOTE: if the file reads from zero, 1 must
'''

def main(args):

    # check time
    start_time = time.time()

    # check that args.FILES has same length as args.NSTEPS
    assert len(args.FILES) == len(args.NSTEPS), "Number of files and NSTEPS must be the same"

    # use if the files are in the same path and thus their absolute path is not provided 
    if args.PATH is not None: 
        args.FILES = [f"{args.PATH}/{file}" for file in args.FILES]

    cell_data = []
    energy_data = []
    coord_data = []
    force_data = []
    for rng in range(len(args.FILES)):
        print(f'+======= Processing file {args.FILES[rng]} =======+')
        # read cell and energy data
        print('**** Loading cell data *****')
        cell = pd.DataFrame(read_cell_or_energy_data(filename=f'{args.FILES[rng]}-Cell.cell', nsteps=args.NSTEPS[rng], STDPRINT=args.STDPRINT, TIMESTEP=args.TIMESTEP, type='cell'))
        print('**** Loading energy data *****')
        energy = pd.DataFrame(read_cell_or_energy_data(filename=f'{args.FILES[rng]}-Energy.en', nsteps=args.NSTEPS[rng], STDPRINT=args.STDPRINT, TIMESTEP=args.TIMESTEP, type='energy'))
        # read the coordinates & forces data
        if args.ZERO_DATA:
            args.NSTEPS = args.NSTEPS+1
        print('**** Loading coordinates data *****')
        coord = read_position_data(filename=f'{args.FILES[rng]}-Trajectory.xyz', nsteps=args.NSTEPS[rng], STDPRINT=args.STDPRINT)
        print('**** Loading force data *****')
        kind_to_element, ordering_letter, ordering_number, force = read_forces_data(filename=f'{args.FILES[rng]}-Forces.for', nsteps=args.NSTEPS[rng], STDPRINT=args.STDPRINT)

        # append to data to a list
        cell_data.append(cell.iloc[:, 2:-1].to_numpy())
        energy_data.append(energy['Cons Qty[a.u.]'].to_numpy())
        coord_data.append(coord)
        force_data.append(force)

    print(f'xxxxxx Processing the {len(args.FILES)} files xxxxxx')
    # convert the list to numpy array
    cell_data  = np.concatenate(cell_data, axis=0)
    energy_data = np.concatenate(energy_data, axis=0)
    coord_data = np.concatenate(coord_data, axis=0)
    force_data = np.concatenate([idx[1:, :] for idx in force_data], axis=0) # removed the starting ones to match the others

    # consistency check
    assert len(cell_data) == len(energy_data) == len(coord_data) == len(force_data), "Number of steps mismatch." # expect the same number of steps
    assert coord_data.shape[1] == force_data.shape[1], "Coordinates data mismatch."
    print(f'Number of steps and coordinates matches')

    # write files
    if args.SAVE_PATH is None and args.PATH is not None:
        args.SAVE_PATH = args.PATH
    # check if the folder does not exist
    if not os.path.exists(args.SAVE_PATH):
        os.makedirs(args.SAVE_PATH)
	
    print(f'Writing files to {args.SAVE_PATH}')

    # split the datas (cell_data, energy_data, ) and save them in set.000, set.010, set.020, ...
    if args.SPLIT:
        nline_per_set = int(len(cell_data) // args.SPLIT_COUNT)
        for i in range(args.SPLIT_COUNT):
            if not os.path.exists(f"{args.SAVE_PATH}/set.{i:0>3}"):
                os.mkdir(f"{args.SAVE_PATH}/set.{i:0>3}")
            
            iset_cell = cell_data[i*nline_per_set:(i+1)*nline_per_set]
            np.save(f"{args.SAVE_PATH}/set.{i:0>3}/box.npy", iset_cell.astype(np.float32))

            iset_energy = energy_data[i*nline_per_set:(i+1)*nline_per_set]
            np.save(f"{args.SAVE_PATH}/set.{i:0>3}/energy.npy", iset_energy.astype(np.float32))

            iset_coord = coord_data[i*nline_per_set:(i+1)*nline_per_set]
            np.save(f"{args.SAVE_PATH}/set.{i:0>3}/coord.npy", iset_coord.astype(np.float32))

            iset_force = force_data[i*nline_per_set:(i+1)*nline_per_set]
            np.save(f"{args.SAVE_PATH}/set.{i:0>3}/force.npy", iset_force.astype(np.float32))

    else:
        if not os.path.exists(f'{args.SAVE_PATH}/set.000'):
            os.makedirs(f'{args.SAVE_PATH}/set.000')

        np.save(f"{args.SAVE_PATH}/set.000/box.npy", cell_data)
        np.save(f"{args.SAVE_PATH}/set.000/energy.npy", energy_data)
        np.save(f"{args.SAVE_PATH}/set.000/coord.npy", coord_data)
        np.save(f"{args.SAVE_PATH}/set.000/force.npy", force_data)

    
    with open(f'{args.SAVE_PATH}/type_map.raw', 'w') as f: # type_map.raw is the ordering atoms in alphabets
        for item in kind_to_element.values():
            f.write(f"{item}\n")

    with open(f'{args.SAVE_PATH}/type.raw', 'w') as f: # type.raw is the ordering atoms in numbers
        for item in ordering_number:
            f.write(f"{item-1}\n") # -1 to match the 0-based indexing

    # with open(f'{args.SAVE_PATH}/mapping.txt', 'w') as f:
    #     for key, value in kind_to_element.items():
    #         f.write(f"{key} : {value}\n")

    print(f'Execution time: {time.time() - start_time} seconds')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="convert cp2k aimd files to deepmd-kit format")
    parser.add_argument("--FILES", type=str, nargs='+', default=['AEM2Cl6Wat900_NVT-04000-08000', 'AEM2Cl6Wat900_NVT-08000-10000'], help="name of the cp2k output files")
    parser.add_argument("--PATH", type=str, default=None, help="folder containing the cp2k output files")
    parser.add_argument("--SAVE_PATH", type=str, help="folder to save the deepmd-kit files")
    parser.add_argument("--NSTEPS", type=int, nargs='+', default=[4000, 2000], help="number of steps")
    parser.add_argument("--STDPRINT", type=int, default=1, help="stdprint")
    parser.add_argument("--TIMESTEP", type=float, default=0.5, help="timestep in fs")
    parser.add_argument("--ZERO_DATA", action="store_true", help="if the file reads from zero, 1 must")
    parser.add_argument("--SPLIT", action="store_true", help="split the data into 10 sets")
    parser.add_argument("--SPLIT_COUNT", type=int, default=10, help="number of sets to split the data")
    args = parser.parse_args()
    main(args)
