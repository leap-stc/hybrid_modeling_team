# FTorch Integration with CAM (Derecho)

Guide for integrating PyTorch-based ML emulators into the Community
Atmosphere Model (CAM) using [FTorch](https://github.com/Cambridge-ICCS/FTorch)
as the Fortran/PyTorch coupling layer, on NCAR's Derecho HPC system. This is
the coupling layer referenced in step 4 of [`Start Here.md`](../Start%20Here.md)
(`ML Model → FTorch → CAM7`).

> Source: [leap-stc/Integration_team discussion #2](https://github.com/leap-stc/Integration_team/discussions/2)
> ("FTorch example in CAM"), started by @addisug, Feb 2025.

## What is FTorch?

FTorch is an interface library that:

- Enables calling PyTorch models directly from Fortran code
- Handles data type conversions between tensors and arrays
- Manages computational resource allocation
- Executes models on CPU or GPU with minimal overhead

Reference implementation: https://github.com/alexeedm/pytorch-fortran

Integration benefits:

- Replaces expensive physics parameterizations with efficient ML emulators
- Maintains existing CAM workflows while adding ML capabilities
- Leverages PyTorch within Fortran-based climate models
- Enables hybrid modeling approaches

Key technical details:

- **Model format:** TorchScript (`.pt`) files
- **Precision:** single precision (`real32`) for PyTorch, double precision (`r8`) for CAM
- **Environment variables:** `USE_FTORCH`, `FTORCH_PREFIX_FTORCH`, `CONDA_PREFIX`
- **Compilation:** CIME scripts with the `gnu` compiler
- **Job configuration:** 20-minute wallclock time is sufficient for test runs

---

## Example 1: Constant Model

Minimal end-to-end example using a model that always returns ones, to
validate the FTorch plumbing before doing anything more complex.

### Python model (`constant_model.py`)

```python
import torch
import torch.nn as nn

class ConstantModel(nn.Module):
    def __init__(self):
        super(ConstantModel, self).__init__()

    def forward(self, x):
        return torch.ones_like(x, dtype=torch.float)

model = ConstantModel()
model_scripted = torch.jit.script(model)
model_scripted.save("constant_model.pt")

example_input = torch.tensor([0.5])
output = model(example_input)
print(f"Test input: {example_input}")
print(f"Test output: {output}")
```

### Derecho setup

```bash
module load conda
conda create --name pytorch_env
conda activate pytorch_env
conda install pytorch
python constant_model.py
```

### Fortran module (`components/cam/src/physics/cam/pytorch_test.F90`)

```fortran
module pytorch_test

  use shr_kind_mod, only: r8 => shr_kind_r8
  use ftorch, only : torch_model, torch_model_load, torch_model_forward, &
          torch_tensor, torch_tensor_from_array, torch_kCPU, torch_delete
  use iso_fortran_env

  implicit none
  save

  character(len=256) :: cb_torch_model = "/path/constant_model.pt"
  type(torch_model) :: model_pytorch

  public init_neural_net, neural_net

contains

  subroutine init_neural_net()
    implicit none
    call torch_model_load(model_pytorch, trim(cb_torch_model))
  end subroutine init_neural_net

  subroutine neural_net()
     implicit none
     integer              :: in_layout(1) = [1]
     integer              :: out_layout(1) = [1]

     type(torch_tensor), dimension(1) :: in_tensor, out_tensor
     real(real32) :: in_data(1) = 1.0
     real(real32), dimension(1) :: out_data

     call torch_tensor_from_array(in_tensor(1), in_data, in_layout, torch_kCPU)
     call torch_tensor_from_array(out_tensor(1), out_data, out_layout, torch_kCPU)
     call torch_model_forward(model_pytorch, in_tensor, out_tensor)

  end subroutine neural_net

end module pytorch_test
```

### CAM integration (`physpkg.F90`)

```fortran
use pytorch_test, only: init_neural_net, neural_net

! At end of phys_init subroutine:
call init_neural_net()
call neural_net()
```

### Compilation and execution

```bash
./cime/scripts/create_newcase --case /path/test_ftorch --mach derecho \
  --compiler gnu --compset FHIST --res f09_f09_mg17 --project <YOUR_PROJECT>

export USE_FTORCH=TRUE
export FTORCH_PREFIX_FTORCH=/root/ftorch/install/path
export CONDA_PREFIX=FALSE

./case.setup
./case.build
./xmlchange STOP_OPTION=ndays
./xmlchange STOP_N=1
./xmlchange RESUBMIT=0
./xmlchange JOB_WALLCLOCK_TIME=00:20:00
./xmlchange PROJECT=<YOUR_PROJECT>
./case.submit
```

---

## Example 2: SimpleNet Model (namelist-configurable weights)

A more complete pattern: a linear layer model, wired through a CAM namelist
option (`weights_file`) so the model path doesn't need to be hardcoded.

### Python model (`simplenet.py`)

```python
import torch
from torch import nn

class SimpleNet(nn.Module):
    def __init__(self,) -> None:
        super().__init__()
        self._fwd_seq = nn.Sequential(
            nn.Linear(5, 5, bias=False))
        with torch.no_grad():
            self._fwd_seq[0].weight = nn.Parameter(2.0 * torch.eye(5))

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        return self._fwd_seq(batch)

if __name__ == "__main__":
    model = SimpleNet()
    model.eval()

    model_scripted = torch.jit.script(model)
    model_scripted.save("simplenet_model.pt")
```

### Derecho setup

```bash
module load conda
conda create --name pytorch_env
conda activate pytorch_env
conda install pytorch
python simplenet.py
```

### Fortran module (`components/cam/src/physics/cam/cam_nn.F90`)

```fortran
module cam_nn
#ifdef USE_FTORCH
  use ftorch,         only: torch_kCPU, torch_tensor, torch_model, &
                            torch_tensor_from_array
  use ftorch,         only: torch_model_load, torch_model_forward, &
                            torch_tensor_delete
  use camsrfexch,     only: cam_in_t
  use physics_types,  only: physics_state
  use shr_kind_mod,   only: CL=>shr_kind_cl
  use cam_abortutils, only: endrun
  use cam_logfile,    only: iulog
  use iso_fortran_env, only : sp => real32
  use namelist_utils,  only: find_group_name
  use spmd_utils,     only: masterproc

  implicit none

  public :: torch_inference, torch_readnl

  type(torch_model)  :: model
  logical            :: model_initialized = .false.
  character(len=CL) :: weights_file

contains

  subroutine torch_readnl(nlfile)
    use namelist_utils, only : find_group_name
    use units,          only : getunit, freeunit
    use spmd_utils,      only: mpicom, mstrid=>masterprocid, &
                               mpi_character, masterproc
    character(len=*), intent(in) :: nlfile
    integer :: unitn
    integer :: ierr
    character(len=*), parameter :: sub="torch_readnl"

    namelist /torch_nl/ weights_file

    if(masterproc) then
       unitn = getunit()
       open( newunit=unitn, file=trim(nlfile), status='old' )
       call find_group_name(unitn, 'torch_nl', status=ierr)
       if (ierr == 0) then
          read(unitn, torch_nl, iostat=ierr)
          if (ierr /= 0) then
                call endrun(sub//': FATAL: reading namelist')
          end if
       end if
       close(unitn)
       call freeunit(unitn)
    endif

    call mpi_bcast(weights_file, len(weights_file), mpi_character, &
                   mstrid, mpicom, ierr)

  end subroutine torch_readnl

  subroutine init_torch_model(model)
    type(torch_model), intent(inout) :: model
    character(len=*), parameter :: sub="init_torch_model"

    call torch_model_load(model, weights_file, torch_kCPU)
  end subroutine init_torch_model

  subroutine torch_inference
    implicit none

    integer, parameter :: wp = sp

    type(torch_model) :: model
    type(torch_tensor), dimension(1) :: in_tensors
    type(torch_tensor), dimension(1) :: out_tensors

    real(wp), dimension(5), target :: in_data
    real(wp), dimension(5), target :: out_data

    integer      :: tensor_layout(1) = [1]

    if (.not. model_initialized) then
       call init_torch_model(model)
       model_initialized = .true.
    end if

    in_data = [0.0_wp, 1.0_wp, 2.0_wp, 3.0_wp, 4.0_wp]

    call torch_model_load(model, trim(weights_file))

    call torch_tensor_from_array(in_tensors(1), in_data, &
                                 tensor_layout, torch_kCPU)

    call torch_tensor_from_array(out_tensors(1), out_data, &
                                 tensor_layout, torch_kCPU)

    call torch_model_forward(model, in_tensors, out_tensors)

    call torch_tensor_delete(in_tensors(1))
    call torch_tensor_delete(out_tensors(1))

  end subroutine torch_inference
#endif
end module cam_nn
```

### XML configuration (`components/cam/bld/namelist_files/namelist_definition.xml`)

```xml
<entry id="weights_file" type="char*256" input_pathname="abs" 
       group="torch_nl" valid_values="">
  Path to Neural Network weights file
        Default: set by build-namelist
</entry>
```

### Runtime options (`components/cam/src/control/runtime_opts.F90`)

```fortran
use cam_nn, only: torch_readnl
```

### CAM integration (`physpkg.F90`)

```fortran
#ifdef USE_FTORCH
  use cam_nn, only: torch_inference
#endif

! At end of phys_init subroutine:
#ifdef USE_FTORCH
    call torch_inference
#endif
```

### Compilation and execution

```bash
./cime/scripts/create_newcase --case /path/test_ftorch --mach derecho \
  --compiler gnu --compset FHIST --res f09_f09_mg17 --project <YOUR_PROJECT>

# In user_nl_cam:
weights_file = '/path/simplenet_model.pt'

export USE_FTORCH=TRUE
export FTORCH_PREFIX_FTORCH=/root/ftorch/install/path
export CONDA_PREFIX=FALSE

./case.setup
./case.build
./xmlchange STOP_OPTION=ndays
./xmlchange STOP_N=1
./xmlchange RESUBMIT=0
./xmlchange JOB_WALLCLOCK_TIME=00:20:00
./xmlchange PROJECT=<YOUR_PROJECT>
./case.submit
```
