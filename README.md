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

# TABLE of Code Capabilities
| Functionalities | gRASPA | gRASPA-fast | gRASPA-HTC |
| :---------------: | :---------------------: | :-----------------------: | :-----------------------: |
| ***Simulation Types*** |||
| Canonical Monte Carlo<br>(NVT-MC) | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| Grand Canonical Monte Carlo<br>(GCMC) | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| Transition-Matrix Monte Carlo<br>in grand canonical ensemble<br>(GC-TMMC) | :heavy_check_mark: | :heavy_check_mark: |  |
| Mixture Adsorption via GCMC | :heavy_check_mark: |
| NVT-Gibbs MC | :heavy_check_mark: |:heavy_check_mark: |
| **Interactions** |
| Lennard-Jones (12-6) | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| Short-Range Coulomb | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| Long-Range Coulomb: Ewald Summation | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| Analytical Tail Correction | :heavy_check_mark: | :heavy_check_mark: |  |
| Machine-Learning Potential<br>(via LibTorch and cppFlow) | :heavy_check_mark: |  |  |
| **Moves** |
| Translation/Rotation | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| Configurational-Bias Monte Carlo (CBMC) | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| Widom test particle insertion | :heavy_check_mark: | :heavy_check_mark: |
| Insertion/Deletion<br>(without CBMC) | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| Insertion/Deletion<br>(with CBMC) | :heavy_check_mark: | :heavy_check_mark: |  |
| NVT-Gibbs volume change move | :heavy_check_mark: | :heavy_check_mark: |  |
| Gibbs particle transfer | :heavy_check_mark: | :heavy_check_mark: |  |
| Configurational Bias/<br>Continuous Fractional Components<br>(CB/CFC) MC | :heavy_check_mark: | :heavy_check_mark: |  |
| **Functionalities** |
| Movies: LAMMPS data file | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| Automatic Determination<br>of # unit cells | | | :heavy_check_mark: |


# Action Items
  * Fugacity Coefficient calculation using PR-EOS (Xiaoyi)
  * LAMMPS-style Setup for Ewald Summation [DONE: Check commit here](https://github.com/snurr-group/gRASPA/commit/929f1e15e367a12617bcae6bbee0c06413ea2769)
  * restart-able TMMC calculations [DONE: Check commit here](https://github.com/snurr-group/gRASPA/commit/cce339a801961b2d48e97b759eabc63bc919fc27)
  * Better output files (distinguish between Initialization/Equilibration/Production phases) [DONE: Check commit here](https://github.com/snurr-group/CUDA-RASPA-DeepPotential/commit/791b796132c1429c48dd4549820a12f69ab0f353)
  * **hash-tag** in input files for commenting (Really?!)
  * **mathematical equations** for idealized pore geometries and easy inputs in the simulation.input file.
  * **Cell list (linked list)** for framework atoms (See if Zhao has time for this...)
