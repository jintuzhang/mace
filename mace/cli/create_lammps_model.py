import os
import sys
import warnings

import torch
from e3nn.util import jit

from mace.calculators import LAMMPS_MACE, LAMMPS_MACE_CHARGE


def main():
    assert len(sys.argv) in [2, 3], f"Usage: {sys.argv[0]} model_path [charge_cv_expr]"

    model_path = sys.argv[1]  # takes model name as command-line input
    model = torch.load(model_path)
    model = model.double().to("cpu")
    model_name = model._get_name()
    if (len(sys.argv) == 2):
        if (model_name not in ['EnergyChargesMACE', 'AtomicsChargesMACE']):
            lammps_model = LAMMPS_MACE(model)
            lammps_model_compiled = jit.compile(lammps_model)
            lammps_model_compiled.save(os.path.basename(model_path) + "-lammps.pt")
        else:
            message = 'Charge CV expression was not given! Will not ' + \
                      'calculate charge CV with in the model!'
            warnings.warn(message)
            lammps_model = LAMMPS_MACE_CHARGE(model)
            lammps_model_compiled = jit.compile(lammps_model)
            lammps_model_compiled.save(os.path.basename(model_path) + "-lammps_charge.pt")
    else:
        if (model_name not in ['EnergyChargesMACE', 'AtomicsChargesMACE']):
            message = '{:s} is not a charge model! Will ignore the given ' + \
                      'charge CV expression!'
            warnings.warn(message.format(sys.argv[1]))
            lammps_model = LAMMPS_MACE(model)
            lammps_model_compiled = jit.compile(lammps_model)
            lammps_model_compiled.save(os.path.basename(model_path) + "-lammps.pt")
        else:
            # This is so crap, I am so retarded ...
            with open('charge_cv_expr.py', 'w') as fp:
                function = 'import torch\n@torch.jit.script\n'
                function += 'def function(c: torch.Tensor) -> torch.Tensor:\n'
                function += '    return {:s}\n'.format(sys.argv[2])
                function += 'expr="{:s}"\n'.format(sys.argv[2])
                print(function, file=fp)
            lammps_model = LAMMPS_MACE_CHARGE(model)
            lammps_model_compiled = jit.compile(lammps_model)
            lammps_model_compiled.save(os.path.basename(model_path) + "-lammps_charge.pt")
            os.remove('charge_cv_expr.py')


if __name__ == "__main__":
    main()
