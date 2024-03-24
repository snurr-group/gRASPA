#!/bin/bash
currentdir=$(pwd)

list=`python3 -c "import numpy as np; print(*('{:.9e}'.format(x) for x in np.logspace(-3,5,10).tolist()))"`

for pres in $list; do
  echo  $pres
  mkdir $pres
  cp *.def simulation.input *.cif ar-deployed32.pth graspa.job $pres/
  cd $currentdir/$pres
  sed -i "30s/.*/Pressure  $pres/" simulation.input
  sbatch graspa.job
  cd ../
done



