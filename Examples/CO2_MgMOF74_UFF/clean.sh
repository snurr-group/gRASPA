#! /bin/bash

# generate a list of numbers that are equally distributed in log scale
list=`python3 -c "import numpy as np; print(*('{:.9e}'.format(x) for x in np.logspace(3,6,20).tolist()))"`



for pres in $list; do
  rm -r  $pres
done
