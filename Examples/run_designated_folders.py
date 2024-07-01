import os
homedir=os.getcwd() + '/'
basics = ['CO2-MFI', 'Methane-TMMC', 'Bae-Mixture', 'NU2000-pX-LinkerRotations', 'Tail-Correction']
#basics = ['NU2000-pX-LinkerRotations', 'Tail-Correction', 'CO2-MFI', 'Methane-TMMC', 'Bae-Mixture']
for direct in basics:
  os.chdir(f"{direct}")
  os.system(f"../../src_clean/nvc_main.x > output.txt")
  os.system("rm -r AllData FirstBead Lambda Movies Restart TMMC")
  os.chdir(f"{homedir}")
  print(f"Basics Simulation {direct} has finished.\n")
