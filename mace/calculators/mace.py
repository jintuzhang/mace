###########################################################################################
# The ASE Calculator for MACE
# Authors: Ilyes Batatia, David Kovacs
# This program is distributed under the MIT License (see MIT.md)
###########################################################################################


from glob import glob
from pathlib import Path
from typing import Union, Dict

import numpy as np
import torch
from ase.calculators.calculator import Calculator, all_changes
from ase.stress import full_3x3_to_voigt_6_stress

from mace import data
from mace.modules.utils import extract_invariant
from mace.tools import torch_geometric, torch_tools, utils


def get_model_dtype(model: torch.nn.Module) -> torch.dtype:
    """Get the dtype of the model"""
    mode_dtype = next(model.parameters()).dtype
    if mode_dtype == torch.float64:
        return "float64"
    if mode_dtype == torch.float32:
        return "float32"
    raise ValueError(f"Unknown dtype {mode_dtype}")


class MACECalculator(Calculator):
    """MACE ASE Calculator
    args:
        model_paths: str, path to model or models if a committee is produced
                to make a committee use a wild card notation like mace_*.model
        device: str, device to run on (cuda or cpu)
        energy_units_to_eV: float, conversion factor from model energy units to eV
        length_units_to_A: float, conversion factor from model length units to Angstroms
        default_dtype: str, default dtype of model
        charges_key: str, Array field of atoms object where atomic charges are stored
        model_type: str, type of model to load
                    Options: [MACE, DipoleMACE, EnergyDipoleMACE, AtomicChargesMACE, EnergyChargesMACE]
        charge_cv_expr: Callable, expression of the charge CV.

    Dipoles are returned in units of Debye
    """

    def __init__(
        self,
        model_paths: Union[list, str],
        device: str,
        energy_units_to_eV: float = 1.0,
        length_units_to_A: float = 1.0,
        default_dtype="",
        charges_key="Qs",
        model_type="MACE",
        charge_cv_expr=None,
        **kwargs,
    ):
        Calculator.__init__(self, **kwargs)
        self.results = {}

        self.model_type = model_type
        self.charge_cv_expr = charge_cv_expr

        if model_type == "MACE":
            self.implemented_properties = [
                "energy",
                "free_energy",
                "node_energy",
                "forces",
                "stress",
            ]
        elif model_type == "DipoleMACE":
            self.implemented_properties = ["dipole"]
        elif model_type == "EnergyDipoleMACE":
            self.implemented_properties = [
                "energy",
                "free_energy",
                "node_energy",
                "forces",
                "stress",
                "dipole",
            ]
        elif model_type == "AtomicChargesMACE":
            self.implemented_properties = ["charges"]
            if (self.charge_cv_expr is not None):
                self.implemented_properties.extend([
                    "charge_cv",
                    "charge_cv_gradients"
                ])
        elif model_type == "EnergyChargesMACE":
            self.implemented_properties = [
                "energy",
                "free_energy",
                "node_energy",
                "forces",
                "stress",
                "charges",
            ]
            if (self.charge_cv_expr is not None):
                self.implemented_properties.extend([
                    "charge_cv",
                    "charge_cv_gradients"
                ])
        else:
            raise ValueError(
                f"Give a valid model_type: [MACE, DipoleMACE, EnergyDipoleMACE, AtomicChargesMACE, EnergyChargesMACE], {model_type} not supported"
            )

        if "model_path" in kwargs:
            print("model_path argument deprecated, use model_paths")
            model_paths = kwargs["model_path"]

        if isinstance(model_paths, str):
            # Find all models that satisfy the wildcard (e.g. mace_model_*.pt)
            model_paths_glob = glob(model_paths)
            if len(model_paths_glob) == 0:
                raise ValueError(f"Couldn't find MACE model files: {model_paths}")
            model_paths = model_paths_glob
        elif isinstance(model_paths, Path):
            model_paths = [model_paths]
        if len(model_paths) == 0:
            raise ValueError("No mace file names supplied")
        self.num_models = len(model_paths)
        if len(model_paths) > 1:
            print(f"Running committee mace with {len(model_paths)} models")
            if model_type in ["MACE", "EnergyDipoleMACE"]:
                self.implemented_properties.extend(
                    ["energies", "energy_var", "forces_comm", "stress_var"]
                )
            elif model_type == "DipoleMACE":
                self.implemented_properties.extend(["dipole_var"])
            elif model_type == "AtomicChargesMACE":
                self.implemented_properties.extend(["charges_var"])
                if (self.charge_cv_expr is not None):
                    self.implemented_properties.extend(["charge_cv_var"])
            elif model_type == "EnergyChargesMACE":
                self.implemented_properties.extend(
                    ["energies", "energy_var", "forces_comm", "stress_var", "charges_var"]
                )
                if (self.charge_cv_expr is not None):
                    self.implemented_properties.extend(["charge_cv_var"])

        self.models = [
            torch.load(f=model_path, map_location=device) for model_path in model_paths
        ]
        for model in self.models:
            model.to(device)  # shouldn't be necessary but seems to help with GPU
        r_maxs = [model.r_max.cpu() for model in self.models]
        r_maxs = np.array(r_maxs)
        assert np.all(
            r_maxs == r_maxs[0]
        ), "committee r_max are not all the same {' '.join(r_maxs)}"
        self.r_max = float(r_maxs[0])

        self.device = torch_tools.init_device(device)
        self.energy_units_to_eV = energy_units_to_eV
        self.length_units_to_A = length_units_to_A
        self.z_table = utils.AtomicNumberTable(
            [int(z) for z in self.models[0].atomic_numbers]
        )
        self.charges_key = charges_key
        model_dtype = get_model_dtype(self.models[0])
        if default_dtype == "":
            print(
                f"No dtype selected, switching to {model_dtype} to match model dtype."
            )
            default_dtype = model_dtype
        if model_dtype != default_dtype:
            print(
                f"Default dtype {default_dtype} does not match model dtype {model_dtype}, converting models to {default_dtype}."
            )
            if default_dtype == "float64":
                self.models = [model.double() for model in self.models]
            elif default_dtype == "float32":
                self.models = [model.float() for model in self.models]
        torch_tools.set_default_dtype(default_dtype)
        for model in self.models:
            for param in model.parameters():
                param.requires_grad = False

    def _create_result_tensors(
        self, model_type: str, num_models: int, num_atoms: int
    ) -> dict:
        """
        Create tensors to store the results of the committee
        :param model_type: str, type of model to load
            Options: [MACE, DipoleMACE, EnergyDipoleMACE, AtomicChargesMACE, EnergyChargesMACE]
        :param num_models: int, number of models in the committee
        :return: tuple of torch tensors
        """
        dict_of_tensors = {}
        if model_type in ["MACE", "EnergyDipoleMACE", "EnergyChargesMACE"]:
            energies = torch.zeros(num_models, device=self.device)
            node_energy = torch.zeros(num_models, num_atoms, device=self.device)
            forces = torch.zeros(num_models, num_atoms, 3, device=self.device)
            stress = torch.zeros(num_models, 3, 3, device=self.device)
            dict_of_tensors.update(
                {
                    "energies": energies,
                    "node_energy": node_energy,
                    "forces": forces,
                    "stress": stress,
                }
            )
        if model_type in ["EnergyDipoleMACE", "DipoleMACE"]:
            dipole = torch.zeros(num_models, 3, device=self.device)
            dict_of_tensors.update({"dipole": dipole})
        if model_type in ["EnergyChargesMACE", "AtomicChargesMACE"]:
            charges = torch.zeros(num_models, num_atoms, device=self.device)
            dict_of_tensors.update({"charges": charges})
            if (self.charge_cv_expr is not None):
                charge_cvs = torch.zeros(num_models, device=self.device)
                dict_of_tensors.update({"charge_cvs": charge_cvs})
                charge_cv_gradients = torch.zeros(num_models, num_atoms, 3, device=self.device)
                dict_of_tensors.update({"charge_cv_gradients": charge_cv_gradients})
        return dict_of_tensors

    def _get_outputs(
        self,
        model_output: dict,
        batch: Dict[str, torch.Tensor],
        compute_force: bool = False,
        compute_stress: bool = False,
        compute_charge_cv: bool = False
    ):
        cell = batch["cell"]
        positions = batch["positions"]
        energy = model_output.get("energy")
        charges = model_output.get("charges")
        node_energy = model_output.get("node_energy")
        displacement = model_output.get("displacement")
        if (compute_charge_cv and charges is not None):
            charge_cv = [self.charge_cv_expr(charges)]
            charge_cv = torch.stack(charge_cv, dim=-1)
            grad_outputs = torch.ones_like(charge_cv)
            charge_cv_gradients = torch.autograd.grad(
                outputs=[charge_cv],
                inputs=[positions],
                grad_outputs=grad_outputs,
                retain_graph=True,
                create_graph=False,
                allow_unused=True
            )[0]
            if charge_cv_gradients is None:
                charge_cv_gradients = torch.zeros_like(positions)
        else:
            charge_cv = None
            charge_cv_gradients = None
        if compute_stress and displacement is not None:
            grad_outputs = torch.ones_like(energy)
            forces, virials = torch.autograd.grad(
                outputs=[energy],
                inputs=[positions, displacement],
                grad_outputs=grad_outputs,
                retain_graph=False,
                create_graph=False,
                allow_unused=True,
            )
            if forces is None:
                forces = torch.zeros_like(positions)
            if virials is None:
                virials = torch.zeros((1, 3, 3))
            stress = torch.zeros_like(displacement)
            cell = cell.view(-1, 3, 3)
            volume = torch.einsum(
                "zi,zi->z",
                cell[:, 0, :],
                torch.cross(cell[:, 1, :], cell[:, 2, :], dim=1),
            ).unsqueeze(-1)
            stress = virials / volume.view(-1, 1, 1)
            forces *= -1
        elif compute_force:
            grad_outputs = torch.ones_like(energy)
            forces = torch.autograd.grad(
                outputs=[energy],
                inputs=[positions],
                grad_outputs=grad_outputs,
                retain_graph=False,
                create_graph=False,
                allow_unused=True,
            )[0]
            if forces is None:
                forces = torch.zeros_like(positions)
            stress = None
            forces *= -1
        else:
            stress = None
            forces = None
        output = {
            "energy": energy,
            "forces": forces,
            "stress": stress,
            "charges": charges,
            "charge_cv": charge_cv,
            "node_energy": node_energy,
            "charge_cv_gradients": charge_cv_gradients,
        }
        return output

    # pylint: disable=dangerous-default-value
    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        """
        Calculate properties.
        :param atoms: ase.Atoms object
        :param properties: [str], properties to be computed, used by ASE internally
        :param system_changes: [str], system changes since last calculation, used by ASE internally
        :return:
        """
        # call to base-class to set atoms attribute
        Calculator.calculate(self, atoms)

        # prepare data
        config = data.config_from_atoms(atoms, charges_key=self.charges_key)
        data_loader = torch_geometric.dataloader.DataLoader(
            dataset=[
                data.AtomicData.from_config(
                    config, z_table=self.z_table, cutoff=self.r_max
                )
            ],
            batch_size=1,
            shuffle=False,
            drop_last=False,
        )

        if self.model_type in ["MACE", "EnergyDipoleMACE", "EnergyChargesMACE"]:
            batch = next(iter(data_loader)).to(self.device)
            node_e0 = self.models[0].atomic_energies_fn(batch["node_attrs"])
            compute_stress = True
        else:
            compute_stress = False

        batch_base = next(iter(data_loader)).to(self.device)
        ret_tensors = self._create_result_tensors(
            self.model_type, self.num_models, len(atoms)
        )
        for i, model in enumerate(self.models):
            batch = batch_base.clone()
            model_out = model(
                batch.to_dict(),
                training=False,
                compute_force=False,
                compute_virials=False,
                compute_stress=False,
                compute_displacement=compute_stress
            )
            out = self._get_outputs(
                model_out,
                batch,
                compute_force=compute_stress,
                compute_stress=compute_stress,
                compute_charge_cv=bool(self.charge_cv_expr)
            )
            if self.model_type in ["MACE", "EnergyDipoleMACE", "EnergyChargesMACE"]:
                ret_tensors["energies"][i] = out["energy"].detach()
                ret_tensors["node_energy"][i] = (out["node_energy"] - node_e0).detach()
                ret_tensors["forces"][i] = out["forces"].detach()
                if out["stress"] is not None:
                    ret_tensors["stress"][i] = out["stress"].detach()
            if self.model_type in ["DipoleMACE", "EnergyDipoleMACE"]:
                ret_tensors["dipole"][i] = out["dipole"].detach()
            if self.model_type in ["AtomicChargesMACE", "EnergyChargesMACE"]:
                ret_tensors["charges"][i] = out["charges"].detach()
                if (self.charge_cv_expr is not None):
                    ret_tensors["charge_cvs"][i] = out["charge_cv"].detach()
                    ret_tensors["charge_cv_gradients"][i] = out["charge_cv_gradients"].detach()

        self.results = {}
        if self.model_type in ["MACE", "EnergyDipoleMACE", "EnergyChargesMACE"]:
            self.results["energy"] = (
                torch.mean(ret_tensors["energies"], dim=0).cpu().item()
                * self.energy_units_to_eV
            )
            self.results["free_energy"] = self.results["energy"]
            self.results["node_energy"] = (
                torch.mean(ret_tensors["node_energy"] - node_e0, dim=0).cpu().numpy()
            )
            self.results["forces"] = (
                torch.mean(ret_tensors["forces"], dim=0).cpu().numpy()
                * self.energy_units_to_eV
                / self.length_units_to_A
            )
            if self.num_models > 1:
                self.results["energies"] = (
                    ret_tensors["energies"].cpu().numpy() * self.energy_units_to_eV
                )
                self.results["energy_var"] = (
                    torch.var(ret_tensors["energies"], dim=0, unbiased=False)
                    .cpu()
                    .item()
                    * self.energy_units_to_eV
                )
                self.results["forces_comm"] = (
                    ret_tensors["forces"].cpu().numpy()
                    * self.energy_units_to_eV
                    / self.length_units_to_A
                )
            if out["stress"] is not None:
                self.results["stress"] = full_3x3_to_voigt_6_stress(
                    torch.mean(ret_tensors["stress"], dim=0).cpu().numpy()
                    * self.energy_units_to_eV
                    / self.length_units_to_A**3
                )
                if self.num_models > 1:
                    self.results["stress_var"] = full_3x3_to_voigt_6_stress(
                        torch.var(ret_tensors["stress"], dim=0, unbiased=False)
                        .cpu()
                        .numpy()
                        * self.energy_units_to_eV
                        / self.length_units_to_A**3
                    )
        if self.model_type in ["DipoleMACE", "EnergyDipoleMACE"]:
            self.results["dipole"] = (
                torch.mean(ret_tensors["dipole"], dim=0).cpu().numpy()
            )
            if self.num_models > 1:
                self.results["dipole_var"] = (
                    torch.var(ret_tensors["dipole"], dim=0, unbiased=False)
                    .cpu()
                    .numpy()
                )
        if self.model_type in ["AtomicChargesMACE", "EnergyChargesMACE"]:
            self.results["charges"] = (
                torch.mean(ret_tensors["charges"], dim=0).cpu().numpy()
            )
            if (self.charge_cv_expr is not None):
                self.results["charge_cv"] = (
                    torch.mean(ret_tensors["charge_cvs"], dim=0).cpu().numpy()
                )
                self.results["charge_cv_gradients"] = (
                    torch.mean(ret_tensors["charge_cv_gradients"], dim=0).cpu().numpy()
                    / self.length_units_to_A
                )
            if self.num_models > 1:
                self.results["charges_var"] = (
                    torch.var(ret_tensors["charges"], dim=0, unbiased=False)
                    .cpu()
                    .numpy()
                )
                if (self.charge_cv_expr is not None):
                    self.results["charge_cv_var"] = (
                        torch.mean(ret_tensors["charge_cvs"], dim=0).cpu().numpy()
                    )

    def get_descriptors(self, atoms=None, invariants_only=True, num_layers=-1):
        """Extracts the descriptors from MACE model.
        :param atoms: ase.Atoms object
        :param invariants_only: bool, if True only the invariant descriptors are returned
        :param num_layers: int, number of layers to extract descriptors from, if -1 all layers are used
        :return: np.ndarray (num_atoms, num_interactions, invariant_features) of invariant descriptors if num_models is 1 or list[np.ndarray] otherwise
        """
        if atoms is None and self.atoms is None:
            raise ValueError("atoms not set")
        if atoms is None:
            atoms = self.atoms
        if self.model_type != "MACE":
            raise NotImplementedError("Only implemented for MACE models")
        if num_layers == -1:
            num_layers = int(self.models[0].num_interactions)
        config = data.config_from_atoms(atoms, charges_key=self.charges_key)
        data_loader = torch_geometric.dataloader.DataLoader(
            dataset=[
                data.AtomicData.from_config(
                    config, z_table=self.z_table, cutoff=self.r_max
                )
            ],
            batch_size=1,
            shuffle=False,
            drop_last=False,
        )
        batch = next(iter(data_loader)).to(self.device)
        descriptors = [model(batch.to_dict())["node_feats"] for model in self.models]
        if invariants_only:
            irreps_out = self.models[0].products[0].linear.__dict__["irreps_out"]
            l_max = irreps_out.lmax
            num_features = irreps_out.dim // (l_max + 1) ** 2
            descriptors = [
                extract_invariant(
                    descriptor,
                    num_layers=num_layers,
                    num_features=num_features,
                    l_max=l_max,
                )
                for descriptor in descriptors
            ]
        descriptors = [descriptor.detach().cpu().numpy() for descriptor in descriptors]

        if self.num_models == 1:
            return descriptors[0]
        return descriptors
