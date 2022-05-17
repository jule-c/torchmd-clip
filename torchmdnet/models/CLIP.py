import torch
import numpy as np
import torch.nn.functional as F
from torch import nn, Tensor
from torch_scatter import scatter
from torch.autograd import grad
from typing import Optional, List
from torchmdnet.models.output_modules import EquivariantScalar



class ContrastiveLoss(nn.Module):
    def __init__(self):
        super(ContrastiveLoss, self).__init__()
        
    def _loss(self, logits, labels):
        #return F.nll_loss(logits, labels)
        return F.cross_entropy(logits, labels)

    def forward(self, molecule_embedding, properties_embedding, labels):
        molecule_loss = self._loss(molecule_embedding, labels)
        property_loss = self._loss(properties_embedding, labels)
        return (molecule_loss + property_loss) / 2


class CLOOB_Loss(nn.Module):
    def __init__(self):
        super(CLOOB_Loss, self).__init__()
        self.tau = 1/10
    
    def loob_loss(self, x, y):
        Ux = Hopfield(x, x)
        Uy = Hopfield(y, x)
        Vx = Hopfield(x, y)
        Vy = Hopfield(y, y)
        return InfoLOOB_loss(Ux, Uy, Vx, Vy, self.tau)

    def forward(self, molecule_embedding, properties_embedding):
        
        return self.loob_loss(molecule_embedding,properties_embedding)


def InfoLOOB_loss(Ux, Uy, Vx, Vy, tau):
    A = torch.clamp(Loo_softmax(Ux, Uy, tau), min=1e-300)
    B = torch.clamp(Loo_softmax(Vx, Vy, tau), min=1e-300)
    return -torch.mean(torch.log(A) + torch.log(B), dim=0)


def Hopfield(a, b, beta=8):
    softmax = nn.Softmax(dim=-1)
    att = softmax(beta * torch.matmul(b, a.T))
    out = torch.matmul(b.T, att).T
    return F.normalize(out, dim=1)


def Loo_softmax(x, y, tau):
    expmat = torch.matmul(x, y.T)
    mat = torch.exp((1 / tau) * expmat)
    num = torch.diagonal(mat)
    mask = torch.ones(x.size()[0], x.size()[0],device=mat.device.type)
    mask = mask - torch.eye(x.size()[0],device=mat.device.type)
    denom = torch.matmul(mask, mat)
    denom = torch.diagonal(denom)
    return torch.div(num, denom)


class CLIP(nn.Module):
    def __init__(
            self,
            molecule_encoder,
            atom_prop_encoder,
            mol_prop_encoder=None,
            hidden_dim=128,
            head=None,
            concatenation=None,
            regression=False,
            CLOOB=False,
            pretrain_atom_only=False,
            pretrain_mol_only=False,
    ):

        super(CLIP, self).__init__()

        if not regression:
            self.molecule_encoder = molecule_encoder
            self.atom_prop_encoder = atom_prop_encoder
            self.mol_prop_encoder = mol_prop_encoder
            if not CLOOB:
                self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

            self.regression = regression
            self.concatenation = concatenation
            self.head = head
            self.CLOOB = CLOOB
            self.scalar = EquivariantScalar(
                hidden_channels=hidden_dim,
                activation='silu'
            )
            self.pretrain_atom_only = pretrain_atom_only
            self.pretrain_mol_only = pretrain_mol_only

        else:
            self.molecule_encoder = molecule_encoder

    def forward(self,
                z: Tensor,
                pos: Tensor,
                batch: Optional[Tensor] = None,
                q: Optional[Tensor] = None,
                s: Optional[Tensor] = None,
                atom_properties: Optional[Tensor] = None,
                mol_properties: Optional[Tensor] = None,
                ):

        if self.regression:
            if self.head is None:
                molecule_embedding = self.molecule_encoder(z, pos, batch, q=q, s=s)
                atom_prop_embedding = self.atom_prop_encoder(atom_properties)
                mol_prop_embedding = self.mol_prop_encoder(mol_properties)

                if self.concatenation == 'linear':
                    output = molecule_embedding + atom_prop_embedding + mol_prop_embedding
                    return self.output(output)
                elif self.concatenation == 'hopfield':
                    molecule_embedding = molecule_embedding.unsqueeze(1)
                    atom_prop_embedding = atom_prop_embedding.unsqueeze(1)
                    mol_prop_embedding = mol_prop_embedding.unsqueeze(1)
                    output = torch.cat([molecule_embedding, atom_prop_embedding], dim=1)
                    query = self.W_q(self.query)
                    energy = torch.softmax(torch.einsum('bnd, hd -> bn', output, query), dim=-1)
                    value = self.W_v(output)
                    output = torch.einsum('bnd, bn -> bd', value, energy)
                    return self.output(output)
                else:
                    print('No concatenation type given!')
                    raise Exception

            elif self.head == 'molecule_only':
                molecule_embedding = self.molecule_encoder(z, pos, batch, q=q, s=s)
                return molecule_embedding

            elif self.head == 'properties_only':
                atom_prop_embedding = self.atom_prop_encoder(atom_properties)
                mol_prop_embedding = self.mol_prop_encoder(atom_properties)
                return atom_prop_embedding, mol_prop_embedding

        else:

            pos.requires_grad_(True)
            x, vec, z, pos, batch = self.molecule_encoder(z, pos, batch=batch, q=q, s=s)

            out = self.scalar.pre_reduce(x, vec, z, pos, batch)

            grad_outputs: List[Optional[torch.Tensor]] = [torch.ones_like(out)]
            dy = grad(
                [out],
                [pos],
                grad_outputs=grad_outputs,
                create_graph=True,
                retain_graph=True,
            )[0]

            if not self.pretrain_atom_only:
                x_mol = scatter(x, batch, dim=0)
                molecule_mol_embedding = x_mol
            else:
                molecule_mol_embedding = None

            molecule_atom_embedding = x if not self.pretrain_mol_only else None

            if self.atom_prop_encoder is not None:
                assert (molecule_atom_embedding is not None), "Model does not output atomic embedding!"
                molecule_atom_embedding = molecule_atom_embedding / molecule_atom_embedding.norm(dim=-1, keepdim=True)
                atom_prop_embedding = self.atom_prop_encoder(atom_properties)
                atom_prop_embedding = atom_prop_embedding / atom_prop_embedding.norm(dim=-1, keepdim=True)
            else:
                molecule_atom_embedding, atom_prop_embedding = None, None

            if self.mol_prop_encoder is not None:
                assert (molecule_mol_embedding is not None), "Model does not output molecule embedding!"
                molecule_mol_embedding = molecule_mol_embedding / molecule_mol_embedding.norm(dim=-1, keepdim=True)
                mol_prop_embedding = self.mol_prop_encoder(mol_properties)
                mol_prop_embedding = mol_prop_embedding / mol_prop_embedding.norm(dim=-1, keepdim=True)
            else:
                molecule_mol_embedding, mol_prop_embedding = None, None

            if self.CLOOB:
                
                return molecule_mol_embedding, molecule_atom_embedding, mol_prop_embedding, atom_prop_embedding, out, -dy

            else:
                logit_scale = self.logit_scale.exp()


                if atom_prop_embedding is not None:
                    logits_per_molecule_atom = logit_scale * molecule_atom_embedding @ atom_prop_embedding.t()
                    logits_per_atom_molecule = logits_per_molecule_atom.t()
                    labels_atom = torch.arange(logits_per_atom_molecule.size(0), dtype=torch.long).cuda()
                else:
                    logits_per_molecule_atom, logits_per_atom_molecule, labels_atom = None, None, None

                if mol_prop_embedding is not None:
                    logits_per_molecule_mol = logit_scale * molecule_mol_embedding @ mol_prop_embedding.t()
                    logits_per_mol_molecule = logits_per_molecule_mol.t()
                    labels_mol = torch.arange(logits_per_molecule_mol.size(0), dtype=torch.long).cuda()
                else:
                    logits_per_molecule_mol, logits_per_mol_molecule, labels_mol = None, None, None

                return logits_per_molecule_mol, logits_per_mol_molecule, \
                       logits_per_molecule_atom, logits_per_atom_molecule, \
                       labels_mol, labels_atom, out, -dy

            # attention_s = torch.einsum('bd, dk -> bk', text_features, text_features.T)
            # p_s = torch.softmax(logit_scale * attention_s, dim=-1)
            #
            # attention_i = torch.einsum('bd, dk -> bk', image_features, image_features.T)
            # p_i = torch.softmax(logit_scale * attention_i, dim=-1)

            # attention_si = torch.einsum('bd, dk -> bk', text_features, image_features.T)
            # p_si = torch.softmax(logit_scale * attention_si, dim=-1)
            #
            # attention_is = torch.einsum('bd, dk -> bk', image_features, text_features.T)
            # p_is = torch.softmax(logit_scale * attention_is, dim=-1)
            #
            # labels = torch.arange(p_is.size(0), dtype=torch.long).cuda()

            # return p_si, p_is, labels

