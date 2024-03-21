#!/bin/bash

# generate a list of numbers that are equally distributed in log scale
list=`python3 -c "import numpy as np; print(*('{:.9e}'.format(x) for x in np.logspace(3,6,20).tolist()))"`

HOME=`pwd`

for pres in $list 
do

cd $pres
	

# Use grep to find the line containing "COMPONENT [1] (argon)"
component_line=$(grep -nx "COMPONENT \[1\] (CO2)" "output.txt")

# Extract the line number
line_number=$(echo "$component_line" | cut -d ':' -f 1)

# Calculate the line number for the 6th line after the component line
target_line_number=$((line_number + 6))

# Use sed to extract the 6th line after the component line
loading=`sed -n "${target_line_number}p" "output.txt"`


echo $pres $loading >> $HOME/loadings.txt

cd $HOME

done




echo "============================================">> $HOME/loadings.txt
