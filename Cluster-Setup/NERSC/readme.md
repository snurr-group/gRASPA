## Installation instructions on [NERSC](https://www.nersc.gov/)
## Follow this instruction to install gRASPA-DP on the NERSC Perlmutter cluster. 
# SPECIAL NOTE: 
  * If you don't need the ML potential, download the code, `cd src_clean/`, and start from [step 6](#Step-6).
  * Installation on other clusters are similar to on Perlmutter of NERSC. Follow the instructions here, and if you encounter issues, please consult with your institution's IT. 
    * [Northwestern Quest IT](https://www.it.northwestern.edu/departments/it-services-support/research/computing/quest/)
  * Check the [nvhpc](https://developer.nvidia.com/hpc-sdk) version. 
    * on NERSC, you can do ```module load PrgEnv-nvhpc``` to use the latest version of nvhpc/cuda.
    * Also check out the [issue on this topic](https://github.com/snurr-group/gRASPA/issues/9)
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
NERSC has its own PyTorch/LibTorch module, so now we can start patching gRASPA code with ML potential functionality. If the libtorch is not pre-installed on your cluster, you can download it via:
```shellscript
wget https://download.pytorch.org/libtorch/cu117/libtorch-cxx11-abi-shared-with-deps-2.0.1%2Bcu117.zip
unzip libtorch-cxx11-abi-shared-with-deps-2.0.1+cu117.zip
```
# Step 5
Modify line 65 in `patch.py` file to `patch_model=['Allegro']`. Then,
```shellscript
python patch.py
```
Now the patched source code is in `patch_Allegro/`. Go into the patched folder, and we have some final work to do.
```shellscript
cd patch_Allegro/
```
If you are using the Lin model, modify line 64 in `patch.py` to `tf_or_torch = ['cppflow']` and line 65 to `patch_model=['LCLIN']`. Then do the similar steps to those for the Allegro model. The patched code will be in `patch_cppflow_LCLIN` folder.
# Step 6
Finally, we need to modify the source code due to NERSC configuration:
```shellscript
sed -i "s/std::filesystem/std::experimental::filesystem/g" *
sed -i "s/<filesystem>/<experimental\/filesystem>/g" *
```
* **NOTE**: Make sure there is no **sub-folder** in the source code folder. Otherwise the wildcard ```*``` will not work!
# Step 7
Then, copy `NVC_COMPILE_NERSC` to the source code folder, and compile the code in the folder as follows:
* NOTE: If you use the vanilla version, simply replace [`NVC_COMPILE_NERSC`](NVC_COMPILE_NERSC) with [`NVC_COMPILE_NERSC_VANILLA`](NVC_COMPILE_NERSC_VANILLA) and proceed.
```shellscript
chmod +x NVC_COMPILE_NERSC
./NVC_COMPILE_NERSC
```
Remeber to change `cppflowDir` and `tfDir` directories in [`NVC_COMPILE_NERSC`](NVC_COMPILE_NERSC) if you install CppFlow and TensorFlow API in different directories. You might see some warning messages during compilation, just ignore them. Once ready, you will see a binary excutable `nvc_main.x` in the folder.
# Step 8
Based on your cluster setup, you may need to do this final step to create a symbolic link:
```shellscript
ln -s libnvrtc-builtins-7237cb5d.so.11.7  libnvrtc-builtins.so.11.7
```
where `libnvrtc-builtins-7237cb5d.so.11.7` is a file in `/your_libtorch_path/libtorch/lib` and it is the source file or target of the link. This command is creating a symbolic link named `libnvrtc-builtins.so.11.7` that points to the source file `libnvrtc-builtins-7237cb5d.so.11.7`. 
<aside>
⚠️ Note that this step may not be needed for NERSC, but will be necessary for gRASPA/Allegro to work on other universitys' clusters, depending on cluster's setup
</aside>


