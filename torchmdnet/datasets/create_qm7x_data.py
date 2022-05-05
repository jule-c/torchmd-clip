from tqdm import tqdm
import torch
from torch_geometric.data import Data
import h5py
import os
from collections import defaultdict
import glob
import multiprocessing
from multiprocessing import Queue
import argparse
from torch_geometric.data.collate import collate
import logging
from joblib import Parallel, delayed
import sys

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

properties_list_selected = ['DIP', 'HLgap', 'KSE', 'atNUM', 'atPOL', 'atXYZ', 'eAT', 'eC', 'eEE', 'eH', 'eKIN', 'eKSE', 'eL',
                   'eNE', 'eNN', 'ePBE0+MBD', 'eTS', 'eX', 'eXC', 'eXX', 'hCHG', 'hDIP', 'mPOL', 'totFOR']

properties_list_all = ['atNUM', 'atXYZ', 'sRMSD', 'ePBE0+MBD', 'eAT', 'ePBE0', 'eMBD', 'eTS', 'eNN', 'eKIN',
                       'eNE', 'eEE', 'eXC', 'eX', 'eC', 'eXX', 'eKSE', 'KSE', 'eH', 'eL', 'HLgap', 'DIP', 'vTQ',
                       'vIQ', 'vEQ', 'mC6', 'mPOL', 'totFOR', 'hVOL', 'hRAT', 'hCHG', 'hVDIP', 'atC6', 'atPOL', 'vdwR']

logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))

def get_args():
    # fmt: off
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--root', default='/Users/jnc/Documents/torchmd_clip', type=str, help='Root folder of hdf5 files.')
    parser.add_argument("--multi", default=False, action="store_true", help="Use multiprocessing.")
    parser.add_argument("--selected-props", default=False, action="store_true", help="Use selected properties.")
    parser.add_argument("--all-props", default=False, action="store_true", help="Use all properties.")
    parser.add_argument("--efc-only", default=False, action="store_true", help="Use only energies, forces, charges despite positions and atomic numbers.")

    args = parser.parse_args()

    return args

def worker(data_list, file, used_properties):

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
                if properties in used_properties:
                    if properties == 'atNUM':
                        temp_file = torch.from_numpy(fDFT[molid][confid][properties][:]).long()
                    else:
                        temp_file = torch.tensor(fDFT[molid][confid][properties][:], dtype=torch.float32)
                        if properties in ['vTQ', 'vIQ', 'vEQ']:
                            temp_file = torch.norm(temp_file)

                    if properties == 'atXYZ':
                        properties = 'pos'
                    elif properties == 'atNUM':
                        properties = 'z'
                    elif properties == 'ePBE0+MBD':
                        properties = 'y'
                    elif properties == 'totFOR':
                        properties = 'dy'
                    elif properties == 'hCHG':
                        properties = 'q'

                    if temp_file.ndim == 1 and temp_file.shape[0] == 1:
                        temp_file = temp_file.unsqueeze(1)
                    property_buffer[properties] = temp_file
            temp_list.append(Data(**property_buffer))
    data_list += temp_list

def create_data_multi(root, args):

    if args.efc_only:
        logging.info("Using only energies, forces and charges as properties!")
        used_properties = ['atNUM', 'atXYZ', 'hCHG', 'totFOR', 'ePBE0+MBD']
        file_name = 'processed/qm7x_m_efc.pt'
    elif args.all_props:
        logging.info("Using all properties!")
        used_properties = properties_list_all
        file_name = 'processed/qm7x_m_all.pt'
    elif args.selected_props:
        logging.info("Using selected properties!")
        used_properties = properties_list_selected
        file_name = 'processed/qm7x_m_selected.pt'

    prompt = input(f'Output will be stored in {os.path.join(root, file_name)}. Type "ok", if this folder exists and you want to proceed.\n')
    if prompt != 'ok':
        sys.exit('Stopped program. Please create the output folder.')

    files = glob.glob(os.path.join(root, "qm7x/*.hdf5"))
    assert len(files) >= 1, (f"No HDF5 file has been found. Save HDF5 file into {os.path.join(root, 'qm7x')}")
    assert len(files) <= 8, (f"More than eight HDF5 files have been found. Please save only QM7X specific HDF5 in {os.path.join(root, 'qm7x')}")
    logging.info(f"Found {len(files)} hdf5 file(s)!")

    logging.info(f"Building {len(files)} process(es)...")

    data_list = []

    Parallel(n_jobs=len(files), backend='threading')(delayed(worker)(data_list, file, used_properties) for file in files)

    logging.info("Collating the list of Data files into one Data file...")
    data, slices = _collate(data_list)

    logging.info(f"Saving the data to {os.path.join(root, file_name)}...")
    torch.save((data, slices), os.path.join(root, file_name))

    logging.info("Finished!")

def create_data_single(root, args):

    if args.efc_only:
        logging.info("Using only energies, forces and charges as properties!")
        used_properties = ['atNUM', 'atXYZ', 'hCHG', 'totFOR', 'ePBE0+MBD']
        file_name = 'processed/qm7x_m_efc.pt'
    elif args.all_props:
        logging.info("Using all properties!")
        used_properties = properties_list_all
        file_name = 'processed/qm7x_m_all.pt'
    elif args.selected_props:
        logging.info("Using all properties!")
        used_properties = properties_list_selected
        file_name = 'processed/qm7x_m_selected.pt'

    data_list = []

    files = glob.glob(os.path.join(root, "qm7x/*.hdf5"))
    assert len(files) >= 1, (f"No HDF5 file has been found. Save HDF5 file into {os.path.join(root, 'qm7x')}")
    assert len(files) <= 8, (f"More than eight HDF5 files have been found. Please save only QM7X specific HDF5 in {os.path.join(root, 'qm7x')}")

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
                    if properties in used_properties:
                        if properties == 'atNUM':
                            temp_file = torch.from_numpy(fDFT[molid][confid][properties][:]).long()
                        else:
                            temp_file = torch.tensor(fDFT[molid][confid][properties][:], dtype=torch.float32)
                            if properties in ['vTQ', 'vIQ', 'vEQ']:
                                temp_file = torch.norm(temp_file)

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

    logging.info("Collating the list of Data files into one Data file...")
    data, slices = _collate(data_list)

    logging.info(f"Saving the data to {os.path.join(root, file_name)}...")
    torch.save((data, slices), os.path.join(root, file_name))

    logging.info("Finished!")

def _collate(data_list):
    if len(data_list) == 1:
        return data_list[0], None

    data, slices, _ = collate(
        data_list[0].__class__,
        data_list=data_list,
        increment=False,
        add_batch=False,
    )

    return data, slices


if __name__ == "__main__":
    args = get_args()
    if args.multi:
        logging.info("Using multiprocessing...")
        create_data_multi(args.root, args)
    else:
        logging.info("Using single processing...")
        create_data_single(args.root, args)

