from tqdm import tqdm
import torch
from torch_geometric.data import Data
import h5py
import numpy as np
import os
from collections import defaultdict
import glob
import multiprocessing
from multiprocessing import Process, Manager, Queue
import argparse
from torch_geometric.data.collate import collate
import logging

properties_list = ['DIP', 'HLgap', 'KSE', 'atNUM', 'atPOL', 'atXYZ', 'eAT', 'eC', 'eEE', 'eH', 'eKIN', 'eKSE', 'eL',
                   'eNE', 'eNN', 'ePBE0+MBD', 'eTS', 'eX', 'eXC', 'eXX', 'hCHG', 'hDIP', 'mPOL', 'totFOR']
logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))


def get_args():
    # fmt: off
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--data', default=None, type=str, help='DB to load from')
    parser.add_argument('--root', default='/Users/jnc/Documents/torchmd_clip/qm7x/gdb7x_pbe0.db', type=str, help='Root folder of hdf5 files')
    parser.add_argument('--multi', type=bool, default=False, help='Use multiprocessing or not')

    args = parser.parse_args()

    return args


def create_data_multi(root):
    def worker(file, data_list):
        """ The worker function, invoked in a process. 'nums' is a
            list of numbers to factor. The results are placed in
            a dictionary that's pushed to a queue.
        """

        temp_list = []
        fDFT = h5py.File(file, 'r')
        ## get IDs of DFTB and DFT files and loop through
        DFTmol_ids = list(fDFT.keys())

        for molid in tqdm(DFTmol_ids):
            ## get IDs of individual configurations/conformations of molecule
            DFTconf_ids = list(fDFT[molid].keys())

            for confid in DFTconf_ids:
                property_buffer = defaultdict()
                for properties in fDFT[molid][confid].keys():
                    if properties in properties_list:
                        if properties == 'atNUM':
                            temp_file = torch.from_numpy(fDFT[molid][confid][properties][:]).long()
                        else:
                            temp_file = torch.tensor(fDFT[molid][confid][properties][:], dtype=torch.float32)

                        if properties == 'atXYZ':
                            properties = 'pos'
                        elif properties == 'atNUM':
                            properties = 'z'
                        elif properties == 'ePBE0+MBD':
                            properties = 'y'
                        elif properties == 'totFOR':
                            properties = 'dy'

                        if temp_file.ndim == 1 and temp_file.shape[0] == 1:
                            temp_file = temp_file.unsqueeze(1)
                        property_buffer[properties] = temp_file
                temp_list.append(Data(**property_buffer))
        data_list.put(temp_list)

    # Each process will get 'chunksize' nums and a queue to put his out
    # dict into
    data_list = Queue()
    procs = []

    files = glob.glob(os.path.join(root, "qm7x/*.hdf5"))
    nprocs = len(files)

    logging.info(f"Found {len(files)} hdf5 files.")
    logging.info(f"Start {nprocs} processes.")

    for i in range(nprocs):
        p = multiprocessing.Process(
                target=worker,
                args=(files[i],
                      data_list))
        procs.append(p)
        p.start()

    # Collect all results into a single result dict. We know how many dicts
    # with results to expect.
    data_list_f = []
    for i in range(nprocs):
        data_list_f.extend(data_list.get())

    # Wait for all worker processes to finish
    for p in procs:
        p.join()

    data, slices = collate(data_list_f)
    torch.save((data, slices), os.path.join(root, "data"))

def create_data(root):

    data_list = []

    files = glob.glob(os.path.join(root, "qm7x/*.hdf5"))
    logging.info(f"Found {len(files)} hdf5 files. Processing...")

    for file in tqdm(files):
        fDFT = h5py.File(file, 'r')
        ## get IDs of DFTB and DFT files and loop through
        DFTmol_ids = list(fDFT.keys())

        for molid in tqdm(DFTmol_ids):
            ## get IDs of individual configurations/conformations of molecule
            DFTconf_ids = list(fDFT[molid].keys())

            for confid in DFTconf_ids:
                property_buffer = defaultdict()
                for properties in fDFT[molid][confid].keys():
                    if properties in properties_list:
                        if properties == 'atNUM':
                            temp_file = torch.from_numpy(fDFT[molid][confid][properties][:]).long()
                        else:
                            temp_file = torch.tensor(fDFT[molid][confid][properties][:], dtype=torch.float32)

                        if properties == 'atXYZ':
                            properties = 'pos'
                        elif properties == 'atNUM':
                            properties = 'z'
                        elif properties == 'ePBE0+MBD':
                            properties = 'y'
                        elif properties == 'totFOR':
                            properties = 'dy'

                        if temp_file.ndim == 1 and temp_file.shape[0] == 1:
                            temp_file = temp_file.unsqueeze(1)
                        property_buffer[properties] = temp_file
                data_list.append(Data(**property_buffer))

    data, slices = collate(data_list)
    torch.save((data, slices), os.path.join(root, "data"))

def collate(data_list):
    if len(data_list) == 1:
        return data_list[0], None

    data, slices, _ = collate(
        data_list[0].__class__,
        data_list=data_list,
        increment=False,
        add_batch=False,
    )

    return data, slices


def create_npz(file):
    folder = f'/datasets/qm7x/{file}'
    property_buffer = defaultdict(list)

    fDFT = h5py.File(file, 'r')
    ## get IDs of DFTB and DFT files and loop through
    DFTmol_ids = list(fDFT.keys())
    for molid in DFTmol_ids:
        ## get IDs of individual configurations/conformations of molecule
        DFTconf_ids = list(fDFT[molid].keys())

        for confid in DFTconf_ids:
            for properties in fDFT[molid][confid].keys():
                temp_array = []
                temp_file = fDFT[molid][confid][properties]
                for j in range(temp_file.shape[0]):
                    temp_array.append(temp_file[j])
                property_buffer[properties].append(np.array(temp_array, dtype=object))

    np.savez(folder, **property_buffer)

if __name__ == "__main__":
    args = get_args()
    if args.multi:
        logging.info("Use multiprocessing...")
        create_data_multi(args.root)
    else:
        create_data(args.root)

