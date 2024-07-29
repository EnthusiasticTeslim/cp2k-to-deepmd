# cp2k-to-deepmd
This repository contains code that facilitates the conversion of CP2K AIMD data to the DeepMD format.

## Requirements

### 1. Converting raw AIMD data to DeepMD

The `multicpk2deepmd.py` script allows for the conversion of raw CP2K AIMD files to the DeepMD format (`.npy`). The following inputs are required:

1. **--PATH**: str, main directory where the AIMD files are located, e.g., `main_path='/work/basic'`
2. **--SAVE_PATH**: str, directory to save the output files. If not specified, the `PATH` is used.
3. **--FILES**: list, names of the AIMD files to be converted, e.g., `'RUN_02/AEM2Cl6Wat900_NVT' 'RUN_03/AEM2Cl6Wat900_NVT'`. Note that the examples contain two different AIMD files, one in `/work/basic/RUN_02` and the other in `/work/basic/RUN_03`.
4. **--NSTEPS**: int, number of steps in each AIMD file, e.g., 4000 2000
5. **--STDPRINT**: int, specifies how info is written, same as the STDPRINT and TRAJMDPRINT in CP2K.
6. **--TIMESTEP**: float, timestep in fs.
7. **--ZERO_DATA**: flag, set if the file reads from zero.
8. **--SPLIT**: int, number of splits to divide the data into sets.
9. **--SPLIT_COUNT**: int, number of sets to split the data into.

### Usage
```bash
~$python ./src/multicpk2deepmd.py --PATH ${main_path}/frame01  --SAVE_PATH ${main_path}/frame01/new_mlpData/allFrames
          --NSTEPS 4000 2000
          --FILES 'cp2kFromMike/AEM2Cl6Wat900_NVT-04000-08000' 'cp2kFromMike/AEM2Cl6Wat900_NVT-08000-10000'
```
if `--SPLIT` is not set, the output will be saved in `--SAVE_PATH` and will contain the following files:
```bash
  - type.raw
  - type_map.raw
  - set.000/coord.npy
  - set.000/energy.npy
  - set.000/force.npy
```

If `--SPLIT` is set, the files will be split into `--SPLIT_COUNT` sets and saved in `--SAVE_PATH`, resulting in:
```bash
  - type.raw
  - type_map.raw
  - set.000/coord.npy
  - set.000/energy.npy
  - set.000/force.npy
  - set.001/coord.npy
  - set.001/energy.npy
  - set.001/force.npy
  - set.002/coord.npy
  - set.002/energy.npy
  - set.002/force.npy
```
> [!IMPORTANT]  
> Execute the code `multicpk2deepmd.py` on a single aimd trajectory (i.e. frame) at a time.
