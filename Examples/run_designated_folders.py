import os
homedir=os.getcwd()
repodir=os.path.dirname(homedir)
basics = ['CO2-MFI', 'Methane-TMMC', 'Bae-Mixture', 'NU2000-pX-LinkerRotations', 'Tail-Correction']
ref_calc = ['Reference_NIST_SPCE/Box-1/', 'Reference_NIST_SPCE/Box-2/', 'Reference_NIST_SPCE/Box-3/', 'Reference_NIST_SPCE/Box-4/']
sims = []
sims.extend(basics); sims.extend(ref_calc)
#basics = ['NU2000-pX-LinkerRotations', 'Tail-Correction', 'CO2-MFI', 'Methane-TMMC', 'Bae-Mixture']
for direct in sims:
  os.chdir(f"{direct}")
  os.system(f"{repodir}/src_clean/nvc_main.x > output.txt")
  os.system("rm -r AllData FirstBead Lambda Movies Restart TMMC")
  os.chdir(f"{homedir}")
  print(f"Simulation {direct} has finished.\n")
