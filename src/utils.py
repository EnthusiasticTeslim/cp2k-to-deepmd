'''
Created on 2024-06-07 15:00:00
Author: Teslim  Olayiwola (
Description: Utility functions for reading data from cp2k output files
'''
import numpy as np
from typing import List

def read_cell_or_energy_data(filename, nsteps=10, STDPRINT=1, TIMESTEP=0.5, columns=None, type='cell'):
  """
  Reads data from a file.

  Args:
      filename: The path to the file.exi
      columns: A list of column names. If None, use the specified columns in the file.
      nsteps: The number of steps in the file.
      STDPRINT: The number of steps between each printed output.
      TIMESTEP: The time step between each step.

  Returns:
      A dictionary where keys are column names and values are lists containing data from 
      each row (excluding the header row).
  """

  # set the default columns
  if columns is None:
    if type == 'cell':
      columns = ['Step', 'Time [fs]', 'Ax [Angstrom]', 'Ay [Angstrom]', 'Az [Angstrom]', 'Bx [Angstrom]', 'By [Angstrom]', 'Bz [Angstrom]', 'Cx [Angstrom]', 'Cy [Angstrom]', 'Cz [Angstrom]', 'Volume [Angstrom^3]']
    elif type == 'energy':
      columns = ['Step Nr.', 'Time[fs]', 'Kin.[a.u.]', 'Temp[K]', 'Pot.[a.u.]', 'Cons Qty[a.u.]', 'UsedTime[s]']
    else:
      print("Unknown type. Please specify between 'cell' and 'energy'.")

  # read the data from the file  
  data = {}

  with open(filename, 'r') as file:
    # Skip the header row (first line)
    next(file)
    
    # Read data from subsequent rows
    for row in file:
      values = row.strip().split()

      # Check if the number of values matches the number of columns
      if len(values) != len(columns):
        print(f"Warning: Row with unmatched number of values found. Skipping.")
        continue

      # Create entries in the dictionary
      for i, value in enumerate(values):
        if columns[i] not in data:
          data[columns[i]] = []
        
        try:
          value = float(value) # convert to float
          data[columns[i]].append(value)
        except ValueError:
          print(f"Warning: Could not convert '{value}' to float. Skipping.")
          continue
        
  # Check if the number of steps is consistent with the number of rows
  if type == 'cell':
    step = 'Step'
    time = 'Time [fs]'
  elif type == 'energy':
    step = 'Step Nr.'
    time = 'Time[fs]'


  # check if the Time [fs] matches the expected time
  if time in data:
    print(f"Succesfully matched {(data[time][-1] - data[time][0])+TIMESTEP} fs & {int(data[step][-1] - data[step][0]) + 1} steps from {type} file of {nsteps} steps / {data[time][1] - data[time][0]} fs timestep.")
  else:
    print(f"No '{time}' found in {type} file.")

  
  # if time in data:
  #   if data[time][-1] - data[time][0] == TIMESTEP*nsteps:
  #     print(f"Succesfully matched {data[time][-1] - data[time][0]} fs from {type} file of {nsteps} fs total.")
  #   else:
  #     print(f"Warning: Matched {data[time][-1] - data[time][0]} fs from {type} file of {nsteps} fs total.")
  # else:
  #   print(f"No '{time}' found in {type} file.")

  return data


def read_forces_data(filename, nsteps=10, STDPRINT=1):
    """
    Reads data from a file.

    Args:
        filename: The path to the file.
        nsteps: The number of steps in the file.
        STDPRINT: The number of steps between each printed output.

    Returns:
        kind_to_element: A dictionary mapping kind to element.
        data: A numpy array containing the data showing the atomic forces with shape (nsteps, num_atoms * 3).
    """

    with open(filename, 'r') as file:
        lines = file.readlines()

    kind_to_element = {}
    arrangement = []
    frames = []
    current_frames = []
    header_found = False

    for line in lines:
        if "Atom" in line and "Kind" in line and "Element" in line:
            header_found = True
            if current_frames:
                frames.append(current_frames)
                current_frames = []
            continue
        
        if not header_found or "ATOMIC FORCES" in line:
            continue

        parts = line.split()
        if len(parts) < 6:
            continue

        atom, kind, element, x, y, z = parts[:6]
        kind_to_element[int(kind)] = element
        arrangement.append([int(kind), element])
        current_frames.append([float(x), float(y), float(z)])

    if current_frames:
        frames.append(current_frames)

    num_frames = len(frames)
    num_atoms = len(frames[0])
    data = np.zeros((num_frames, num_atoms * 3))
    ordering_number = [x[0] for x in arrangement[:int(len(arrangement)/num_frames)]] # get the 'number' ordering of the atoms in the file
    ordering_letter = [x[1] for x in arrangement[:int(len(arrangement)/num_frames)]] # get the 'alphabet'  ordering of the atoms in the file

    assert num_atoms * 3 == data.shape[1], "Data shape mismatch."
    assert num_frames == int(nsteps / STDPRINT) + 1, "Number of frames does not match the expected number of printed frames."

    print(f"Successfully loaded xyz forces with {num_atoms} atoms each within {num_frames} frames")

    for i, frame in enumerate(frames):
        data[i] = np.array(frame).flatten()

    return kind_to_element, ordering_letter, ordering_number, data


def read_position_data(filename, nsteps=10, STDPRINT=1) -> np.ndarray:
    """
    Reads XYZ data from a file with three distinct sections.

    Args:
        filename: The path to the file.
        nsteps: The number of steps in the file.
        STDPRINT: The number of steps between each printed output.

    Returns:
        data: A numpy array containing the XYZ data with shape (number of distinct sections, 3 * number of rows).
    """
    
    with open(filename, 'r') as file:
        lines = file.readlines()

    frames = []
    current_frame = []
    frame_started = False

    for line in lines:
        if "i =" in line and "time =" in line and "E =" in line: # standard print in cp2k
            if frame_started:
                if current_frame:
                    frames.append(current_frame)
                    current_frame = []
            frame_started = True
            continue

        if frame_started:
            parts = line.split()
            if len(parts) != 4:
                continue
            
            atom, x, y, z = parts
            current_frame.append([float(x), float(y), float(z)])

    if current_frame:
        frames.append(current_frame)

    num_frames = len(frames)
    num_atoms = len(frames[0])
    data = np.zeros((num_frames, num_atoms * 3))


    assert num_atoms * 3 == data.shape[1], "Data shape mismatch."
    assert num_frames == int(nsteps / STDPRINT), "Number of frames does not match the expected number of steps."

    print(f"Successfully loaded xyz forces with {num_atoms} atoms each within {num_frames} frames")


    for i, frame in enumerate(frames):
        data[i] = np.array(frame).flatten()

    return data

