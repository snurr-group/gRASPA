## Installation instructions on [NERSC](https://www.nersc.gov/)
## NOTE: If you don't need the DeepPotential, download the code, `cd src_clean/`, and start from [step 6](#Step-6).
Follow this instruction to install gRASPA-DP on the NERSC Perlmutter cluster. 
# Step 1
We download TensorFlow2 C++ API to a local directory: (assuming in the HOME directory)
```shellscript
mkdir ctensorflow
cd ctensorflow
wget https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-gpu-linux-x86_64-2.11.0.tar.gz
tar -xvf libtensorflow-gpu-linux-x86_64-2.11.0.tar.gz
cd ..
vi .bashrc
```
# Step 2
Add the following directories to environment variables in the `.bashrc` file:
```shellscript
export LIBRARY_PATH=$LIBRARY_PATH:~/ctensorflow/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/ctensorflow/lib
```
# Step 3
Then, we install [CppFlow](https://github.com/serizba/cppflow):
```shellscript
git clone https://github.com/serizba/cppflow
cd cppflow
mkdir build
cd build
cmake -DCMAKE_PREFIX_PATH=~/ctensorflow/ ..
make install DESTDIR=~/ctensorflow/
```
# Step 4
NERSC has its own PyTorch/LibTorch module, so now we can start patching gRASPA code with ML potential functionality. 
For example, if we want to use Allegro model which uses LibTorch:
```shellscript
mkdir patch_Allegro
```
# Step 5
Modify line 64 in `patch.py` file to `patch_model=['Allegro']`. Then,
```shellscript
python patch.py
```
Now the patched source code is in `patch_Allegro/`. Go into the patched folder, and we have some final work to do.
```shellscript
cd patch_Allegro/
```
# Step 6
Finally, we need to modify the source code due to NERSC configuration:
```shellscript
sed -i "s/std::filesystem/std::experimental::filesystem/g" *
sed -i "s/<filesystem>/<experimental\/filesystem>/g" *
```
# Step 7
Then, copy `NVC_COMPILE_NERSC` to the source code folder, and compile the code in the folder as follows:
* NOTE: If you use the vanilla version, simply replace [`NVC_COMPILE_NERSC`](NVC_COMPILE_NERSC) with [`NVC_COMPILE_NERSC_VANILLA`](NVC_COMPILE_NERSC_VANILLA) and proceed.
```shellscript
chmod +x NVC_COMPILE_NERSC
./NVC_COMPILE_NERSC
```
Remeber to change `cppflowDir` and `tfDir` directories in [`NVC_COMPILE_NERSC`](NVC_COMPILE_NERSC) if you install CppFlow and TensorFlow API in different directories. You might see some warning messages during compilation, just ignore them. Once ready, you will see a binary excutable `nvc_main.x` in the folder.
