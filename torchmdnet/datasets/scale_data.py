from sklearn.preprocessing import MinMaxScaler
import torch
import argparse
import os

def get_args():
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--path', default='/Users/jnc/Documents/torchmd-clip/processed/qm7x_all.pt', type=str, help='Filepath to qm7x_all.pt.')
    parser.add_argument('--out-path', default='/Users/jnc/Documents/torchmd-clip/processed', type=str, help='Output directory')

    parser.add_argument("--mol-only", default=False, action="store_true", help="Use only mol properties")
    parser.add_argument("--atom-only", default=False, action="store_true", help="Use only atom properties.")

    args = parser.parse_args()

    return args

def main(path, out_path, atom_only, mol_only):

    scaler = MinMaxScaler()

    data = torch.load(path)
    batch = data[0]

    properties_list_all = ['atNUM', 'atXYZ', 'sRMSD', 'ePBE0+MBD', 'eAT', 'ePBE0', 'eMBD', 'eTS', 'eNN', 'eKIN',
                           'eNE', 'eEE', 'eXC', 'eX', 'eC', 'eXX', 'eKSE', 'KSE', 'eH', 'eL', 'HLgap', 'DIP', 'vTQ',
                           'vIQ', 'vEQ', 'mC6', 'mPOL', 'totFOR', 'hVOL', 'hRAT', 'hCHG', 'hVDIP', 'atC6', 'atPOL', 'vdwR']

    if atom_only or not mol_only:
        atomic_properties = torch.hstack([
            torch.norm(batch.dy, dim=-1).unsqueeze(1), batch.hVOL, batch.hRAT, batch.atPOL,
            batch.atC6, torch.norm(batch.hVDIP, dim=-1).unsqueeze(1), batch.q, batch.z.unsqueeze(1)
        ])
        atomic_properties_norm = scaler.fit_transform(atomic_properties)
        atomic_properties_norm = torch.tensor(atomic_properties_norm, dtype=torch.float32)

        data[0]['atom_properties'] = atomic_properties_norm
        data[1]['atom_properties'] = data[1]['pos']

    if mol_only or not atom_only:
        mol_properties = torch.hstack([
            batch.DIP, batch.HLgap, batch.eAT, batch.eC, batch.eEE, batch.eH,
            batch.eKIN, batch.eKSE, batch.eL, batch.ePBE0, batch.eMBD, batch.eNE, batch.eNN, batch.eTS, batch.eX,
            batch.eXC, batch.eXX, batch.mPOL, batch.mC6, batch.sRMSD, batch.vTQ.unsqueeze(1), batch.vIQ.unsqueeze(1),
            batch.vEQ.unsqueeze(1)
        ])
        mol_properties_norm = scaler.fit_transform(mol_properties)
        mol_properties_norm = torch.tensor(mol_properties_norm, dtype=torch.float32)

        data[0]['mol_properties'] = mol_properties_norm
        data[1]['mol_properties'] = data[1]['y']


    data_ = data[0].clone()
    for k, v in data_.items():
        if k in properties_list_all:
            data[0].pop(k)
            data[1].pop(k)

    del data_

    name = "atom_only" if atom_only else "mol_only"
    if not atom_only and not mol_only:
        name = "all"
    torch.save(data, os.path.join(out_path, f'qm7x_all_minmax_{name}.pt'))

if __name__ == "__main__":
    args = get_args()
    main(args.path, args.out_path, args.atom_only, args.mol_only)