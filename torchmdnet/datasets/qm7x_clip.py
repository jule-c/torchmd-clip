import pdb
from tqdm import tqdm
import torch
from torch_geometric.data import Data, Dataset
import h5py
from collections import defaultdict
import os
from typing import List
import glob
from torch_geometric.data.collate import collate
from torch_geometric.data.separate import separate
import logging
from joblib import Parallel, delayed
import time


logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
properties_list = ['DIP', 'HLgap', 'KSE', 'atNUM', 'atPOL', 'atXYZ', 'eAT', 'eC', 'eEE', 'eH', 'eKIN', 'eKSE', 'eL',
                   'eNE', 'eNN', 'ePBE0+MBD', 'eTS', 'eX', 'eXC', 'eXX', 'hCHG', 'hDIP', 'mPOL', 'totFOR']


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
    data_list += temp_list


class QM7X_clip(Dataset):

    element_numbers = {"H": 1, "C": 6, "N": 7, "O": 8, "Cl": 17, "S": 16}

    self_energies = {"H": -13.641404161,
                     "C": -1027.592489146,
                     "N": -1484.274819088,
                     "O": -2039.734879322,
                     "Cl": -12516.444619523,
                     "S": -10828.707468187,
                     }

    all_props = True
    atom_only = False
    mol_only = False

    def __init__(self, root, transform=None, pre_transform=None, dataset_arg="efc_only"):
        super(QM7X_clip, self).__init__(root, transform, pre_transform)

        logging.info(f"Using {dataset_arg}!")
        assert ((self.atom_only != self.mol_only) or
                (self.mol_only is False and self.atom_only is False)
                ), "Don't set atom_only and mol_only to True!"
        logging.info("Set all_props, atom_only and mol_only in qm7x_clip.py accordingly!")
        logging.info(f"Currently all_props: {self.all_props}, atom_only: {self.atom_only}, mol_only: {self.mol_only}.")
        time.sleep(8)

        self.all_props = dataset_arg == "all_props"
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):

        if self.all_props:
            if self.atom_only:
                return ["qm7x_all_minmax_atom_only.pt"]
            elif self.mol_only:
                return ["qm7x_all_minmax_mol_only.pt"]
            else:
                return ["qm7x_all_minmax_all.pt"]
        else:
            return ["qm7x_efc.pt"]

    def process(self):

        if not self.all_props:
            logging.info("Using only energies, forces and charges as properties!")
            used_properties = ['atNUM', 'atXYZ', 'hCHG', 'totFOR', 'ePBE0+MBD']
            file_name = 'processed/qm7x_efc.pt'
        else:
            logging.info("Using all properties! If you want to use only certain properties, set self.all_props in __init__ to False!")
            used_properties = properties_list
            file_name = 'processed/qm7x_all.pt'

        files = glob.glob(os.path.join(self.root, "qm7x/*.hdf5"))
        assert len(files) >= 1, (f"No HDF5 file has been found. Save HDF5 file into {os.path.join(self.root, 'qm7x')}")
        assert len(files) <= 8, (
            f"More than eight HDF5 files have been found. Please save only QM7X specific HDF5 in {os.path.join(self.root, 'qm7x')}")
        logging.info(f"Found {len(files)} hdf5 file(s)!")

        logging.info(f"Building {len(files)} process(es)...")

        data_list = []

        Parallel(n_jobs=len(files), backend='threading')(
            delayed(worker)(data_list, file, used_properties) for file in files)

        logging.info("Collating the list of Data files into one Data file...")
        data, slices = self._collate(data_list)

        logging.info(f"Saving the data to {os.path.join(self.root, file_name)}...")
        torch.save((data, slices), os.path.join(self.root, file_name))

        logging.info("Finished!")

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self._collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def get(self, idx: int) -> Data:

        data = separate(
            cls=self.data.__class__,
            batch=self.data,
            idx=idx,
            slice_dict=self.slices,
            decrement=False,
        )

        return data

    def _collate(self, data_list: List[Data]):
        r"""Collates a Python list of :obj:`torch_geometric.data.Data` objects
        to the internal storage format of
        :class:`~torch_geometric.data.InMemoryDataset`."""
        if len(data_list) == 1:
            return data_list[0], None

        data, slices, _ = collate(
            data_list[0].__class__,
            data_list=data_list,
            increment=False,
            add_batch=False,
        )

        return data, slices

    def get_atomref(self, max_z=100):
        out = torch.zeros(max_z)
        out[list(self.element_numbers.values())] = torch.tensor(
            list(self.self_energies.values())
        )
        return out.view(-1, 1)

    def len(self):
        return len(self.data.y)
