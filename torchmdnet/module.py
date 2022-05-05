import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn.functional import mse_loss, l1_loss
from torchmdnet.models.CLIP import ContrastiveLoss, CLOOB_Loss
from pytorch_lightning import LightningModule
from torchmdnet.models.model import create_model, load_model


class LNNP(LightningModule):
    def __init__(self, hparams, prior_model=None, mean=None, std=None):
        super(LNNP, self).__init__()

        if "charge" not in hparams:
            hparams["charge"] = False
        if "spin" not in hparams:
            hparams["spin"] = False

        self.save_hyperparameters(hparams)

        if self.hparams.load_model:
            self.model = load_model(self.hparams.load_model, args=self.hparams)
        else:
            self.model = create_model(self.hparams, prior_model, mean, std)

        # initialize exponential smoothing
        self.ema = None
        self._reset_ema_dict()

        # initialize loss collection
        self.losses = None
        self._reset_losses_dict()

    def configure_optimizers(self):
        optimizer = AdamW(
            self.model.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )

        scheduler = ReduceLROnPlateau(
            optimizer,
            "min",
            factor=self.hparams.lr_factor,
            patience=self.hparams.lr_patience,
            min_lr=self.hparams.lr_min,
        )

        lr_scheduler = {
            "scheduler": scheduler,
            "monitor": "val_loss",
            "interval": "epoch",
            "frequency": 1,
        }
        return [optimizer], [lr_scheduler]

    def forward(self, z, pos, batch=None, q=None, s=None, atom_properties=None, mol_properties=None):
        if not self.hparams.use_clip and not self.hparams.use_cloob:
            return self.model(z, pos, batch=batch, q=q, s=s)

        elif self.hparams.use_clip or self.hparams.use_cloob:
            return self.model(z, pos, batch=batch, q=q, s=s,
                              atom_properties=atom_properties,
                              mol_properties=mol_properties,
                              )

    def training_step(self, batch, batch_idx):
        return self.step(batch, mse_loss, "train")

    def validation_step(self, batch, batch_idx, *args):
        if len(args) == 0 or (len(args) > 0 and args[0] == 0):
            # validation step
            return self.step(batch, mse_loss, "val")
        # test step
        return self.step(batch, l1_loss, "test")

    def test_step(self, batch, batch_idx):
        return self.step(batch, l1_loss, "test")

    def step(self, batch, loss_fn, stage):

        if not self.hparams.use_clip and not self.hparams.use_cloob:
            with torch.set_grad_enabled(stage == "train" or self.hparams.derivative):

                # TODO: the model doesn't necessarily need to return a derivative once
                # Union typing works under TorchScript (https://github.com/pytorch/pytorch/pull/53180)
                pred, deriv = self(batch.z, batch.pos, batch=batch.batch,
                                   q=batch.q if self.hparams.charge else None,
                                   s=batch.s if self.hparams.spin else None)

            loss_y, loss_dy = 0, 0
            if self.hparams.derivative:
                if "y" not in batch:
                    # "use" both outputs of the model's forward function but discard the first
                    # to only use the derivative and avoid 'Expected to have finished reduction
                    # in the prior iteration before starting a new one.', which otherwise get's
                    # thrown because of setting 'find_unused_parameters=False' in the DDPPlugin
                    deriv = deriv + pred.sum() * 0

                # force/derivative loss
                loss_dy = loss_fn(deriv, batch.dy)

                if stage in ["train", "val"] and self.hparams.ema_alpha_dy < 1:
                    if self.ema[stage + "_dy"] is None:
                        self.ema[stage + "_dy"] = loss_dy.detach()
                    # apply exponential smoothing over batches to dy
                    loss_dy = (
                        self.hparams.ema_alpha_dy * loss_dy
                        + (1 - self.hparams.ema_alpha_dy) * self.ema[stage + "_dy"]
                    )
                    self.ema[stage + "_dy"] = loss_dy.detach()

                if self.hparams.force_weight > 0:
                    self.losses[stage + "_dy"].append(loss_dy.detach())

            if "y" in batch:
                if batch.y.ndim == 1:
                    batch.y = batch.y.unsqueeze(1)

                # energy/prediction loss
                loss_y = loss_fn(pred, batch.y)

                if stage in ["train", "val"] and self.hparams.ema_alpha_y < 1:
                    if self.ema[stage + "_y"] is None:
                        self.ema[stage + "_y"] = loss_y.detach()
                    # apply exponential smoothing over batches to y
                    loss_y = (
                        self.hparams.ema_alpha_y * loss_y
                        + (1 - self.hparams.ema_alpha_y) * self.ema[stage + "_y"]
                    )
                    self.ema[stage + "_y"] = loss_y.detach()

                if self.hparams.energy_weight > 0:
                    self.losses[stage + "_y"].append(loss_y.detach())

            # total loss
            loss = loss_y * self.hparams.energy_weight + loss_dy * self.hparams.force_weight

            self.losses[stage].append(loss.detach())

            # check unused parameters
            #if stage in ["train"]:
            #    loss.backward()
            #    name_ = []
            #    for name, param in self.model.named_parameters():
            #        if param.grad is None:
            #            name_.append(name)

            return loss

        elif self.hparams.use_clip:
            with torch.set_grad_enabled(stage == "train" or self.hparams.derivative):
                # TODO: the model doesn't necessarily need to return a derivative once
                # Union typing works under TorchScript (https://github.com/pytorch/pytorch/pull/53180)

                mol_properties = None if self.hparams.pretrain_atom_only else batch.mol_properties
                atom_properties = None if self.hparams.pretrain_mol_only else batch.atom_properties

                logits_per_molecule_mol, logits_per_mol_molecule, \
                logits_per_molecule_atom, logits_per_atom_molecule, \
                labels_mol, labels_atom \
                    = self(batch.z, batch.pos, batch=batch.batch,
                           atom_properties=atom_properties,
                           mol_properties=mol_properties,
                           )

            loss_x, loss_y = 0, 0
            loss_clip = ContrastiveLoss()
            if logits_per_molecule_atom is not None:
                assert (logits_per_atom_molecule is not None), "CLIP does not output atom property encoding!"
                loss_y = loss_clip(logits_per_molecule_atom, logits_per_atom_molecule, labels_atom)
                self.losses[stage + "_y"].append(loss_y.detach())

            if logits_per_molecule_mol is not None:
                assert (logits_per_mol_molecule is not None), "CLIP does not output atom property encoding!"
                loss_x = loss_clip(logits_per_molecule_mol, logits_per_mol_molecule, labels_mol)
                self.losses[stage + "_x"].append(loss_x.detach())

            # total loss
            loss = (loss_x + loss_y)/2

            self.losses[stage].append(loss.detach())
            return loss

        elif self.hparams.use_cloob:
            with torch.set_grad_enabled(stage == "train" or self.hparams.derivative):
                # TODO: the model doesn't necessarily need to return a derivative once
                # Union typing works under TorchScript (https://github.com/pytorch/pytorch/pull/53180)

                mol_properties = None if self.hparams.pretrain_atom_only else batch.mol_properties
                atom_properties = None if self.hparams.pretrain_mol_only else batch.atom_properties

                molecule_mol_embedding, molecule_atom_embedding, mol_prop_embedding, atom_prop_embedding \
                    = self(batch.z, batch.pos, batch=batch.batch,
                           atom_properties=atom_properties,
                           mol_properties=mol_properties,
                           )

            loss_x, loss_y = 0, 0
            loss_cloob = CLOOB_Loss()
            if molecule_atom_embedding is not None:
                assert (atom_prop_embedding is not None), "CLIP does not output atom property encoding!"
                loss_y = loss_cloob(molecule_atom_embedding, atom_prop_embedding)
                self.losses[stage + "_y"].append(loss_y.detach())

            if molecule_mol_embedding is not None:
                assert (mol_prop_embedding is not None), "CLIP does not output molecule property encoding!"
                loss_x = loss_cloob(molecule_mol_embedding, mol_prop_embedding)
                self.losses[stage + "_x"].append(loss_x.detach())

            # total loss
            loss = (loss_x + loss_y)/2

            self.losses[stage].append(loss.detach())
            return loss


    def optimizer_step(self, *args, **kwargs):
        optimizer = kwargs["optimizer"] if "optimizer" in kwargs else args[2]
        if self.trainer.global_step < self.hparams.lr_warmup_steps:
            lr_scale = min(
                1.0,
                float(self.trainer.global_step + 1)
                / float(self.hparams.lr_warmup_steps),
            )

            for pg in optimizer.param_groups:
                pg["lr"] = lr_scale * self.hparams.lr
        super().optimizer_step(*args, **kwargs)
        optimizer.zero_grad()

    def training_epoch_end(self, training_step_outputs):
        dm = self.trainer.datamodule
        if hasattr(dm, "test_dataset") and len(dm.test_dataset) > 0:
            should_reset = (
                self.current_epoch % self.hparams.test_interval == 0
                or (self.current_epoch - 1) % self.hparams.test_interval == 0
            )
            if should_reset:
                # reset validation dataloaders before and after testing epoch, which is faster
                # than skipping test validation steps by returning None
                self.trainer.reset_val_dataloader(self)

    def validation_epoch_end(self, validation_step_outputs):
        if not self.trainer.running_sanity_check:
            # construct dict of logged metrics

            if not self.hparams.use_clip and not self.hparams.use_cloob:
                result_dict = {
                    "epoch": self.current_epoch,
                    "lr": self.trainer.optimizers[0].param_groups[0]["lr"],
                    "train_loss": torch.stack(self.losses["train"]).mean(),
                    "val_loss": torch.stack(self.losses["val"]).mean(),
                }

                # add test loss if available
                if len(self.losses["test"]) > 0:
                    result_dict["test_loss"] = torch.stack(self.losses["test"]).mean()

                # if prediction and derivative are present, also log them separately
                if len(self.losses["train_y"]) > 0 and len(self.losses["train_dy"]) > 0:
                    result_dict["train_loss_y"] = torch.stack(self.losses["train_y"]).mean()
                    result_dict["train_loss_dy"] = torch.stack(
                        self.losses["train_dy"]
                    ).mean()
                    result_dict["val_loss_y"] = torch.stack(self.losses["val_y"]).mean()
                    result_dict["val_loss_dy"] = torch.stack(self.losses["val_dy"]).mean()

                    if len(self.losses["test"]) > 0:
                        result_dict["test_loss_y"] = torch.stack(
                            self.losses["test_y"]
                        ).mean()
                        result_dict["test_loss_dy"] = torch.stack(
                            self.losses["test_dy"]
                        ).mean()
            else:
                if len(self.losses["train_x"]) > 0 and len(self.losses["train_y"]) > 0:
                    result_dict = {
                        "epoch": self.current_epoch,
                        "lr": self.trainer.optimizers[0].param_groups[0]["lr"],
                        "train_loss": torch.stack(self.losses["train"]).mean(),
                        "val_loss": torch.stack(self.losses["val"]).mean(),
                        "train_loss_x": torch.stack(self.losses["train_x"]).mean(),
                        "val_loss_x": torch.stack(self.losses["val_x"]).mean(),
                        "train_loss_y": torch.stack(self.losses["train_y"]).mean(),
                        "val_loss_y": torch.stack(self.losses["val_y"]).mean(),
                    }
                elif len(self.losses["train_x"]) > 0:
                    result_dict = {
                        "epoch": self.current_epoch,
                        "lr": self.trainer.optimizers[0].param_groups[0]["lr"],
                        "train_loss": torch.stack(self.losses["train"]).mean(),
                        "val_loss": torch.stack(self.losses["val"]).mean(),
                        "train_loss_x": torch.stack(self.losses["train_x"]).mean(),
                        "val_loss_x": torch.stack(self.losses["val_x"]).mean(),
                    }

                elif len(self.losses["train_y"]) > 0:
                    result_dict = {
                        "epoch": self.current_epoch,
                        "lr": self.trainer.optimizers[0].param_groups[0]["lr"],
                        "train_loss": torch.stack(self.losses["train"]).mean(),
                        "val_loss": torch.stack(self.losses["val"]).mean(),
                        "train_loss_y": torch.stack(self.losses["train_y"]).mean(),
                        "val_loss_y": torch.stack(self.losses["val_y"]).mean(),
                    }

                if len(self.losses["test"]) > 0:

                    if len(self.losses["test_x"]) > 0:
                        result_dict["test_loss_x"] = torch.stack(
                            self.losses["test_x"]).mean()

                    if len(self.losses["test_y"]) > 0:
                        result_dict["test_loss_y"] = torch.stack(
                            self.losses["test_y"]).mean()

            self.log_dict(result_dict, sync_dist=True)
        self._reset_losses_dict()

    def _reset_losses_dict(self):
        if not self.hparams.use_clip and not self.hparams.use_cloob:
            self.losses = {
                "train": [],
                "val": [],
                "test": [],
                "train_y": [],
                "val_y": [],
                "test_y": [],
                "train_dy": [],
                "val_dy": [],
                "test_dy": [],
            }
        else:
            self.losses = {
                "train": [],
                "val": [],
                "test": [],
                "train_x": [],
                "val_x": [],
                "test_x": [],
                "train_y": [],
                "val_y": [],
                "test_y": [],
            }

    def _reset_ema_dict(self):
        self.ema = {"train_y": None, "val_y": None, "train_dy": None, "val_dy": None}
