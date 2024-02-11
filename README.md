# gRASPA-DeepPotential version 
This is an implementation of ML potential in gRASPA. Depending on ML models, it requires TensorFlow C++ API, CppFlow package, and LibTorch.

# Documentation is available [here](https://github.com/snurr-group/gRASPA-mkdoc/tree/gh-pages)
- Click the link
- Download the code by pressing the green button called **<> Code**, then select **Download ZIP**
- Unzip the downloaded file
- Navigate into the unzipped folder, and click on **index.html**

# NOTES:
  * To install gRASPA on various clusters, check out [Cluster-Setup](Cluster-Setup/)
# Reference:
  * gRASPA paper is not published yet.
  * Part of gRASPA is documented in [Zhao Li's dissertation](https://www.proquest.com/docview/2856224877/406AD117D18F4215PQ/1?accountid=12861). Please kindly cite this if you used it in your publication. Thanks.

# Action Items
  * Fugacity Coefficient calculation using PR-EOS (Xiaoyi)
  * LAMMPS-Setup for Ewald Summation [DONE: Check commit here](https://github.com/snurr-group/gRASPA/commit/929f1e15e367a12617bcae6bbee0c06413ea2769)
  * restart-able TMMC calculations [DONE: Check commit here](https://github.com/snurr-group/gRASPA/commit/cce339a801961b2d48e97b759eabc63bc919fc27)
  * Better output files (distinguish between Initialization/Equilibration/Production phases) [DONE: Check commit here](https://github.com/snurr-group/CUDA-RASPA-DeepPotential/commit/791b796132c1429c48dd4549820a12f69ab0f353)
  * **hash-tag** in input files for commenting (Really?!)
  * **mathematical equations** for idealized pore geometries and easy inputs in the simulation.input file.
