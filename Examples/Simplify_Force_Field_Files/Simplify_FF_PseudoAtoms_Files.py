import numpy as np
import pandas as pd
import pathlib
import os
from os import path
import shutil

#################################################################
# READS CIF AND ADSORBATE DEFINITIONS AND SIMPLIFY THE FF FILES #
#################################################################

def get_frameworks_from_input():
    with open('simulation.input') as f:
        for line in f:
            if("FrameworkName" in line):
                spline = line.split()
                spline.pop(0)
                Framework_List = spline
    return Framework_List
def get_adsorbates_from_input():    
    Adsorbate_List = []
    with open('simulation.input') as f:
        for line in f:
            if("Component" in line):
                spline = line.split()
                Adsorbate_List.append(spline[3])
    return Adsorbate_List

def read_Framework_pseudoAtoms(filename):
    start = 0
    end = 0
    ncols = 0
    count = 0
    labels = []
    with open(filename) as f:
        for line in f:
            if("_atom_" in line):
                ncols += 1
            if(ncols > 0 and count > 0 and not("_atom_" in line)):
                spline = line.split()
                if not (ncols == len(spline)):
                    raise Exception("CIF FILE WRONG!")
                label = spline[0]
                # remove numbers in label
                label = ''.join(i for i in label if not i.isdigit())
                labels.append(label)
            count += 1
    labels = list(set(labels))
    f.close()
    return labels

def read_Molecule_pseudoAtoms(filename):
    start = 0
    count = 0
    NAtom = 0
    Atomcount = 0
    labels = []
    with open(filename) as f:
        for line in f:
            if(count == 5): #read number of Atoms
                spline = line.split()
                NAtom = int(spline[0])
            if("atomic positions" in line):
                start = count
            if(NAtom > 0 and start > 0 and Atomcount < NAtom and not("#" in line)):
                spline = line.split()
                labels.append(spline[1])
                print(spline[1])
                Atomcount += 1
            count += 1
        labels = list(set(labels))
        print(labels)
    f.close()
    return labels
    
def Process_ForceFieldFile(Labels):
    input_file    = "force_field_mixing_rules.def"
    original_file = "original_force_field_mixing_rules.def"
    output_file   = "output_force_field_mixing_rules.def"
    shutil.copy(input_file, original_file)
    count = 0
    newlines=len(Labels)
    oldlines=0
    with open(original_file) as fr:
        with open(output_file, 'w') as fw:
            for line in fr:
                if(count == 5):
                    spline = line.split()
                    oldlines = int(spline[0])
                    line = str(int(newlines)) + '\n'
                if(count < 7 or count >= (7 + oldlines)):
                    fw.write(line)
                else:
                    spline  = line.split()
                    element = spline[0]
                    if(element in Labels):
                        fw.write(line)
                count += 1
    fr.close()
    fw.close()
    
def Process_PseudoAtomsFile(Labels):
    input_file    = "pseudo_atoms.def"
    original_file = "original_pseudo_atoms.def"
    output_file   = "output_pseudo_atoms.def"
    shutil.copy(input_file, original_file)
    count = 0
    newlines=len(Labels)
    oldlines=0
    with open(original_file) as fr:
        with open(output_file, 'w') as fw:
            for line in fr:
                if(count == 1):
                    spline = line.split()
                    oldlines = int(spline[0])
                    line = str(int(newlines)) + '\n'
                if(count < 3):
                    fw.write(line)
                else:
                    spline  = line.split()
                    element = spline[0]
                    if(element in Labels):
                        fw.write(line)
                count += 1
    fr.close()
    fw.close()
    
Framework_List = get_frameworks_from_input()
Adsorbate_List = get_adsorbates_from_input()
Frameworklabels = []
Adsorbatelabels = []
for framework in Framework_List:
    labels = read_Framework_pseudoAtoms(framework + '.cif')
    Frameworklabels.extend(labels)
for adsorbate in Adsorbate_List:
    labels = read_Molecule_pseudoAtoms(adsorbate + '.def')
    Adsorbatelabels.extend(labels)

Labels = []
Labels.extend(Frameworklabels)
Labels.extend(Adsorbatelabels)

Process_ForceFieldFile(Labels)
Process_PseudoAtomsFile(Labels)
