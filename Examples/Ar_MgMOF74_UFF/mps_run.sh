#!/bin/bash
currentdir=$(pwd)

list=`python3 -c "import numpy as np; print(*('{:.9e}'.format(x) for x in np.logspace(-3,5,20).tolist()))"`


./start_as_root.sh

for pres in $list; do
  echo  $pres
  mkdir $pres
  cp $currentdir/*.def $pres/; cp $currentdir/simulation.input $pres/; cp $currentdir/*.cif $pres/
  #cp -r minimax.txt Random_double3_run DP_model_CH4_MgMOF74_float64_toy $pres/
  cd $currentdir/$pres
  sed -i "30s/.*/Pressure  $pres/" simulation.input
  nvc_main_mps.x > output.txt &
  cd ../
done

wait

./stop_as_root.sh
