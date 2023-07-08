## Installation instructions on [NERSC](https://www.nersc.gov/)
Follow this instruction to install gRASPA-DP on the NERSC Perlmutter cluster. As of 7/7/2023, Perlmutter loads the following modules by default:
```console
  1) craype-x86-milan
  2) libfabric/1.15.2.0
  3) craype-network-ofi
  4) xpmem/2.5.2-2.4_3.48__gd0f7936.shasta
  5) PrgEnv-gnu/8.3.3
  6) cray-dsmml/0.2.2
  7) cray-libsci/23.02.1.1
  8) cray-mpich/8.1.25
  9) craype/2.7.20
 10) gcc/11.2.0
 11) perftools-base/23.03.0
 12) cpe/23.03
 13) xalt/2.10.2
 14) Nsight-Compute/2022.1.1
 15) Nsight-Systems/2022.2.1
 16) cudatoolkit/11.7
 17) craype-accel-nvidia80
 18) gpu/1.0
 19) evp-patch
 20) python/3.9-anaconda-2021.11
```
You should not need to load additional modules if you start from a pristine terminal, but make sure you have the same modules loaded as us. If everything looks good, we start by downloading TensorFlow2 C++ API to a local directory: (assuming in the HOME directory)
```shellscript
mkdir ctensorflow
cd ctensorflow
wget https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-gpu-linux-x86_64-2.11.0.tar.gz
tar -xvf libtensorflow-gpu-linux-x86_64-2.11.0.tar.gz
cd ..
vi .bashrc
```
Add the following directories to environment variables in the `.bashrc` file:
```shellscript
export LIBRARY_PATH=$LIBRARY_PATH:~/ctensorflow/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/ctensorflow/lib
```
Then, we install [CppFlow](https://github.com/serizba/cppflow):
```shellscript
git clone https://github.com/serizba/cppflow
cd cppflow
mkdir build
cd build
cmake -DCMAKE_PREFIX_PATH=~/ctensorflow/ ..
make install DESTDIR=~/ctensorflow/
```
NERSC has its own PyTorch/LibTorch module, so now we can start patching gRASPA code with ML potential functionality. For example, if we want to use Allegro model which uses LibTorch:
```shellscript
mkdir patch_Allegro
```
Modify line 64 in `patch.py` file to `patch_model=['Allegro']`. Then,
```shellscript
python patch.py
```
Once the gRASPA code gets patched, we need to modify the source code due to NERSC configuration:
```shellscript
sed -i "s/std::filesystem/std::experimental::filesystem/g" *
sed -i "s/<filesystem>/<experimental\/filesystem>/g" *
```
Then, copy `NVC_COMPILE_NERSC` to the source code folder `patch_Allegro`, and compile the code `patch_Allegro` folder as follows:
```shellscript
chmod +x NVC_COMPILE_NERSC
./NVC_COMPILE_NERSC
```
Remeber to change `cppflowDir` and `tfDir` directories in `NVC_COMPILE_NERSC` if you install CppFlow and TensorFlow API in different directories. You might see some warning messages during compilation, just ignore them. Once ready, you will see a binary excutable `nvc_main.x` in the folder.
