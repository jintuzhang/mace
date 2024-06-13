from typing import Dict, List, Optional

import torch
from e3nn.util.jit import compile_mode

from mace.tools.scatter import scatter_sum


@compile_mode("script")
class LAMMPS_MACE(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.register_buffer("atomic_numbers", model.atomic_numbers)
        self.register_buffer("r_max", model.r_max)
        self.register_buffer("num_interactions", model.num_interactions)
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(
        self,
        data: Dict[str, torch.Tensor],
        local_or_ghost: torch.Tensor,
        compute_virials: bool = False,
    ) -> Dict[str, Optional[torch.Tensor]]:
        num_graphs = data["ptr"].numel() - 1
        compute_displacement = False
        if compute_virials:
            compute_displacement = True
        out = self.model(
            data,
            training=False,
            compute_force=False,
            compute_virials=False,
            compute_stress=False,
            compute_displacement=compute_displacement,
        )
        node_energy = out["node_energy"]
        if node_energy is None:
            return {
                "total_energy_local": None,
                "node_energy": None,
                "forces": None,
                "virials": None,
            }
        positions = data["positions"]
        displacement = out["displacement"]
        forces: Optional[torch.Tensor] = torch.zeros_like(positions)
        virials: Optional[torch.Tensor] = torch.zeros_like(data["cell"])
        # accumulate energies of local atoms
        node_energy_local = node_energy * local_or_ghost
        total_energy_local = scatter_sum(
            src=node_energy_local, index=data["batch"], dim=-1, dim_size=num_graphs
        )
        # compute partial forces and (possibly) partial virials
        grad_outputs: List[Optional[torch.Tensor]] = [
            torch.ones_like(total_energy_local)
        ]
        if compute_virials and displacement is not None:
            forces, virials = torch.autograd.grad(
                outputs=[total_energy_local],
                inputs=[positions, displacement],
                grad_outputs=grad_outputs,
                retain_graph=False,
                create_graph=False,
                allow_unused=True,
            )
            if forces is not None:
                forces = -1 * forces
            else:
                forces = torch.zeros_like(positions)
            if virials is not None:
                virials = -1 * virials
            else:
                virials = torch.zeros_like(displacement)
        else:
            forces = torch.autograd.grad(
                outputs=[total_energy_local],
                inputs=[positions],
                grad_outputs=grad_outputs,
                retain_graph=False,
                create_graph=False,
                allow_unused=True,
            )[0]
            if forces is not None:
                forces = -1 * forces
            else:
                forces = torch.zeros_like(positions)
        return {
            "total_energy_local": total_energy_local,
            "node_energy": node_energy,
            "forces": forces,
            "virials": virials,
        }


@compile_mode("script")
class LAMMPS_MACE_CHARGE(LAMMPS_MACE):
    def __init__(self, model):
        if (model._get_name() != 'EnergyChargesMACE'):
            message = 'Only the `EnergyChargesMACE` model is supported!'
            raise RuntimeError(message)
        super().__init__(model)
        self.register_buffer("is_charge_model", torch.tensor([1]))
        # My lord, please forgive me for writing these evil codes ...
        try:
            file = __import__('charge_cv_expr')
            self.charge_cv_expr = file.function
            self.charge_cv_expr_string = file.expr
        except ImportError as e:
            print(e)
            self.has_charge_cv_expr = False
            self.charge_cv_expr = torch.tensor  # to fake the compiler
            self.charge_cv_expr_string = '(none)'
        else:
            self.has_charge_cv_expr = True

    def forward(
        self,
        data: Dict[str, torch.Tensor],
        local_or_ghost: torch.Tensor,
        compute_virials: bool = False,
        compute_total_charge_gradients: bool = False
    ) -> Dict[str, Optional[torch.Tensor]]:
        num_graphs = data["ptr"].numel() - 1
        assert num_graphs == 1
        compute_displacement = False
        if compute_virials:
            compute_displacement = True
        out = self.model(
            data,
            training=False,
            compute_force=False,
            compute_virials=False,
            compute_stress=False,
            compute_displacement=compute_displacement,
        )
        node_energy = out["node_energy"]
        if node_energy is None:
            return {
                "total_energy_local": None,
                "node_energy": None,
                "forces": None,
                "virials": None,
            }
        positions = data["positions"]
        displacement = out["displacement"]
        forces: Optional[torch.Tensor] = torch.zeros_like(positions)
        virials: Optional[torch.Tensor] = torch.zeros_like(data["cell"])
        # Get the charge CV and total charge
        charges = out["charges"]
        total_charge = out['total_charge']
        assert charges is not None
        assert total_charge is not None
        charge_cv_gradients: Optional[torch.Tensor] = torch.zeros_like(positions)
        total_charge_gradients: Optional[torch.Tensor] = torch.zeros_like(positions)
        grad_outputs: List[Optional[torch.Tensor]] = [torch.ones(1)]
        if (compute_total_charge_gradients):
            total_charge_gradients = torch.autograd.grad(
                outputs=[total_charge[0]],
                inputs=[positions],
                grad_outputs=grad_outputs,
                retain_graph=True,
                create_graph=False,
                allow_unused=True,
            )[0]
        if (self.has_charge_cv_expr):
            charge_cv = self.charge_cv_expr(charges[data["tag_r"]])
            charge_cv_gradients = torch.autograd.grad(
                outputs=[charge_cv],
                inputs=[positions],
                grad_outputs=grad_outputs,
                retain_graph=True,
                create_graph=False,
                allow_unused=True,
            )[0]
        else:
            charge_cv = torch.tensor([0.], dtype=torch.float64)
        # accumulate energies of local atoms
        node_energy_local = node_energy * local_or_ghost
        total_energy_local = scatter_sum(
            src=node_energy_local, index=data["batch"], dim=-1, dim_size=num_graphs
        )
        # compute partial forces and (possibly) partial virials
        grad_outputs: List[Optional[torch.Tensor]] = [
            torch.ones_like(total_energy_local)
        ]
        if compute_virials and displacement is not None:
            forces, virials = torch.autograd.grad(
                outputs=[total_energy_local],
                inputs=[positions, displacement],
                grad_outputs=grad_outputs,
                retain_graph=False,
                create_graph=False,
                allow_unused=True,
            )
            if forces is not None:
                forces = -1 * forces
            else:
                forces = torch.zeros_like(positions)
            if virials is not None:
                virials = -1 * virials
            else:
                virials = torch.zeros_like(displacement)
        else:
            forces = torch.autograd.grad(
                outputs=[total_energy_local],
                inputs=[positions],
                grad_outputs=grad_outputs,
                retain_graph=False,
                create_graph=False,
                allow_unused=True,
            )[0]
            if forces is not None:
                forces = -1 * forces
            else:
                forces = torch.zeros_like(positions)
        return {
            "total_energy_local": total_energy_local,
            "node_energy": node_energy,
            "forces": forces,
            "virials": virials,
            "total_charge": total_charge[0],
            "total_charge_gradients": total_charge_gradients,
            "charges": charges,
            "charge_cv": charge_cv,
            "charge_cv_gradients": charge_cv_gradients,
        }
