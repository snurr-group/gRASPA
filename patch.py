import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import glob
import os
import os.path as path
import re
import shutil

from itertools import repeat
from typing import NamedTuple


class patch:
    key = ''
    txt = []

def Read_File_into_array_string(file):
    patch_list = []
    temp = patch()
    count = -1
    with open(file) as f:
        for line in f:
            if("PATCH_" in line):
                count += 1
                patch_list.append(patch())
                patch_list[count].key = line.strip()
                patch_list[count].txt = []
                continue
            patch_list[count].txt.append(line)
    return patch_list

def CheckPATCH_STATE(fin, patchkey):
    with open(fin, 'r') as f:
            for line in f:
                if(patchkey + "_PATCHED" in line):
                    raise Exception(fin + "has already been patched, Abort!")

def Find_And_Write(fout, fin, patch):
    with open(fout, 'w') as ff:
        with open(fin, 'r') as f:
                for line in f:
                    if(patch.key in line):
                        print(line)
                        line = line.replace(patch.key, patch.key + "_PATCHED")
                        for a in patch.txt:
                            line += a
                    ff.write(line)
                    
def WritePatchTofile(fin, patch_list):
    temp = 'temp.cpp'
    shutil.copy(fin, temp)
    #shutil.copy(fin, fin)

    for patch in patch_list:
        CheckPATCH_STATE(fin, patch.key)
        shutil.copy(temp, fin)
        Find_And_Write(temp, fin, patch)
    
    shutil.move(temp, fin)

#provide patch path here
#it will read whatever after the '/' as the model name
patch_path= ['cppflow-patch/LCLIN'] # Patch either 'libtorch-patch/Allegro' or 'cppflow-patch/LCLIN'
clean_src = 'src_clean/'
#patch_keyword = 'PATCH_LCLIN_SINGLE'
for model in patch_path:
    if(len(patch_path) > 1):
        raise Exception("DO ONE PATH AT A TIME!!!")
    model_name = model.split('/')[-1]
    final_dir = 'patch_' + model_name + '/'
    os.makedirs(final_dir, exist_ok=True)
    src_files = os.listdir(clean_src)
    src_files = [f for f in src_files if os.path.isfile(clean_src + '/' + f)]
    for f in src_files:
        shutil.copy(clean_src + f, final_dir + f)
    files = os.listdir(model)
    patch_file = [f for f in files if os.path.isfile(model + '/' + f)]
    for f in patch_file:
        first_half = f.split('.txt')[0]
        srcfile    = first_half.split('PATCH_' + model_name.upper() + '_')[1]
        print("Processing " + model_name + " patch" + " for " + srcfile)
        patches = Read_File_into_array_string(model + '/' + f)
        WritePatchTofile(final_dir + srcfile, patches)
