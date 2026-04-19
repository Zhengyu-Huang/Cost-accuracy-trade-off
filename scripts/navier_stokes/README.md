## Install GeophysicalFlows on wm2


https://fourierflows.github.io/GeophysicalFlowsDocumentation/stable/installation_instructions/


module load gcc/12.2.0
module load openmpi/4.1.5-gcc_12.2.0
module load cmake/3.31.9
module load OpenBLAS/0.3.17


export OMPI_MCA_btl=self,vader,tcp
unset OMPI_MCA_pml
unset OMPI_MCA_mtl




git clone https://github.com/spectralDNS/spectralDNS.git

cd spectralDNS

python setup.py build_ext --inplace

conda create --name spectralDNS -c conda-forge shenfun mpi4py-fft cython numba pythran mpich pip h5py=*=mpi*
conda activate spectralDNS


python3 ../firedrake-configure --no-package-manager --show-petsc-configure-options | xargs -L1 ./configure
make PETSC_DIR=/lustre/home/2306192137/src/petsc PETSC_ARCH=arch-firedrake-default all
make PETSC_DIR=/lustre/home/2306192137/src/petsc PETSC_ARCH=arch-firedrake-default check
cd ..

python3 -m venv venv-firedrake
. venv-firedrake/bin/activate

pip cache purge

export $(python3 firedrake-configure --no-package-manager  --show-env)
pip install --no-binary h5py 'firedrake[check]'

firedrake-check

! When any library is not supported, for example h5py
pip uninstall -y h5py
pip install --upgrade firedrake




## Use firedrake on wm2

module load anaconda3/2024.10.1 
module load gcc/12.2.0
module load openmpi/4.1.5-gcc_12.2.0
module load cmake/3.31.9
module load OpenBLAS/0.3.17


export OMPI_MCA_btl=self,vader,tcp
unset OMPI_MCA_pml
unset OMPI_MCA_mtl

. /lustre/home/2306192137/src/venv-firedrake/bin/activate