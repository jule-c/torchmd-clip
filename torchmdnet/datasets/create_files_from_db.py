from schnetpack import AtomsData
import logging
import os
from joblib import Parallel, delayed
import pickle
import argparse
from tqdm import tqdm

properties_list = ['energies', 'forces', 'atomic_numbers', 'positions']

logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))

def get_args():
    # fmt: off
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--out-path', default='/Users/jnc/Documents/torchmd_clip/qm7x', type=str, help='Directory of output files.')
    parser.add_argument("--file", default='/Users/jnc/Documents/torchmd_clip/qm7x/gdb7x_pbe0.db', type=str, help='Path to DB file.')

    args = parser.parse_args()

    return args

def pickle_save(file, file_name):
    open_file = open(file_name, "wb")
    pickle.dump(file, open_file)
    open_file.close()

def worker(data, property, energies, forces, coords, embeds):
    if property == 'energies':
        energies += [data[i]['EDFT'] + data[i]['EMBD'] for i in tqdm(range(len(data)))]
    elif property == 'forces':
        forces += [data[i]['forces'] for i in tqdm(range(len(data)))]
    elif property == 'positions':
        coords += [data[i]['_positions'] for i in tqdm(range(len(data)))]
    elif property == 'atomic_numbers':
        embeds += [data[i]['_atomic_numbers'] for i in tqdm(range(len(data)))]

def create_files(file, out_path):

    logging.info("Loading db file with schnetpack Atomloader...")
    data = AtomsData(file)

    energies, forces, coords, embeds = [], [], [], []

    logging.info("Loading all processes...")
    Parallel(n_jobs=4, backend='threading')(delayed(worker)(data, property, energies, forces, coords, embeds) for property in properties_list)
    logging.info("Multiprocessing finished...")

    logging.info("Saving all files...")
    pickle_save(energies, os.path.join(out_path, 'energies.pkl'))
    pickle_save(forces, os.path.join(out_path, 'forces.pkl'))
    pickle_save(coords, os.path.join(out_path, 'coords.pkl'))
    pickle_save(embeds, os.path.join(out_path, 'embeds.pkl'))

if __name__ == "__main__":
    args = get_args()
    create_files(args.file, args.out_path)
