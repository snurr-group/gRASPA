Cleaning up (simplify) the force field files using python scripts. 
It reads what framework/adsorbates you are going to use, and remove every unused element from the files below. 

-- force_field_mixing_rules.def
-- pseudo_atoms.def

This can reduce the time to process the force field parameters and MAY give you a little speed up. It reduces memory required as well. 

** NOTE **
1. The example starts from force_field file using the UFF.
2. for overwritting terms (force_field.def), you need to manually add them, the python script does not do it for you. 
