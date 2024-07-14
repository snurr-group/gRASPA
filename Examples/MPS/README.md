# NVIDIA-MPS
* MPS starts for **Multi-Process Service**
* Read more about this [here](https://docs.nvidia.com/deploy/mps/)

# What does it do?
* a single GCMC simulation may not fully utilize a GPU card
    * :memo: use `nvidia-smi` command to check GPU usage
* Using MPS, you can run multiple GCMC simulations (processes) on the same GPU card to **improve utility**

# How to use?
1. Start with the MPS Example (here in this folder!)
2. `chmod 777 mps_run start_as_root.sh stop_as_root.sh`
3. You can open `mps_run` and take a look:
```
#!/bin/bash
runs=3
currentdir=$(pwd)

./start_as_root.sh

for ((i = 0; i < $runs; i++)); do
  echo  $i
  mkdir $i
  cp $currentdir/*.def $i/; cp $currentdir/simulation.input $i/; cp $currentdir/*.cif $i/
  cd $currentdir/$i
  sed -i 's/xxx/'$i'/g' simulation.input
  ../../../src_clean/nvc_main.x > result &
  cd ../
done

wait

./stop_as_root.sh
```
* The process goes like the following:
    1. start the MPS daemon
    2. create `3` folders, generate input files in each of them, and run them
    3. once every simulation is done, shut down the MPS daemon
* :memo: NOTE: use `watch -n0.5 nvidia-smi` in a separate terminal to monitor GPU usage while running jobs with MPS.