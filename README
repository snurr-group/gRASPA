# gRASPA-DP (Deep Potential) version 
This is an implementation of ML potential in gRASPA. Depending on ML models, it requires TensorFlow C++ API, CppFlow package, and LibTorch.

## Installation instructions on [NERSC](https://www.nersc.gov/)
Follow this instruction to install gRASPA-DP on the NERSC Perlmutter cluster. First, we download TensorFlow2 C++ API to a local directory: (assuming in the HOME directory)
```console
mkdir ctensorflow
cd ctensorflow
wget https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-gpu-linux-x86_64-2.6.0.tar.gz
tar -xvf libtensorflow-gpu-linux-x86_64-2.6.0.tar.gz
cd ..
vi .bashrc
```
Add the following directories to environment variables in the `.bashrc` file:
```
export LIBRARY_PATH=$LIBRARY_PATH:~/ctensorflow/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/ctensorflow/lib
```
Then, we install [CppFlow](https://github.com/serizba/cppflow):
```console
git clone https://github.com/serizba/cppflow
cd cppflow
mkdir build
cd build
cmake -DCMAKE_PREFIX_PATH=~/ctensorflow/ ..
make install DESTDIR=~/ctensorflow/
```
NERSC has its own PyTorch/LibTorch module, so now we can start patching gRASPA code with ML potential functionality. For example, if we want to use Allegro model which uses LibTorch:
```console
mkdir patch_Allegro
```
Modify line 64 in `patch.py` file to `patch_model=['Allegro']`. Then,
```console
python patch.py
```
Once the gRASPA code gets patched, we need to modify the source code due to NERSC configuration:
```console
sed -i "s/std::filesystem/std::experimental::filesystem/g" *
sed -i "s/<filesystem>/<experimental\/filesystem>/g" *
```
Then, copy `NVC_COMPILE_NERSC` to the source code folder `patch_Allegro`, and compile the code `patch_Allegro` folder as follows:
```console
chmod +x NVC_COMPILE_NERSC
./NVC_COMPILE_NERSC
```
Remeber to change `cppflowDir` and `tfDir` directories in `NVC_COMPILE_NERSC` if you install CppFlow and TensorFlow API in different directories. You might see some warning messages during compilation, just ignore them. Once ready, you will see a binary excutable `nvc_main.x` in the folder.


