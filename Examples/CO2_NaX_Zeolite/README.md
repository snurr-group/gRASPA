This folder contains example simulation input files for modeling CO2 adsorption in the
Na-exchanged faujasite zeolite NaX (Al/Si framework with 55 extra-framework Na+ cations) at
303 K and 10 kPa. The Na+ cations are treated as a mobile framework component
(`Framework_Component_1.def`), and pairwise Lennard-Jones interactions (including the strong
CO2-Na+ pairs) are supplied through `force_field.def` as explicit `# mixing rules to overwrite`
entries on top of the Lorentz-Berthelot base rules in `force_field_mixing_rules.def`.
