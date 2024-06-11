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
  * gRASPA paper is currently in progress. Please kindly cite it when it is published.

# TABLE of Code Capabilities
| Functionalities | gRASPA | gRASPA-fast | gRASPA-HTC |
| :---------------: | :---------------------: | :-----------------------: | :-----------------------: |
| ***Simulation Types*** |||
| Canonical Monte Carlo<br>(NVT-MC) | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| Grand Canonical Monte Carlo<br>(GCMC) | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| Transition-Matrix Monte Carlo<br>in grand canonical ensemble<br>(GC-TMMC) | :heavy_check_mark: | :heavy_check_mark: |  |
| Mixture Adsorption via GCMC | :heavy_check_mark: |
| NVT-Gibbs MC | :heavy_check_mark: |:heavy_check_mark: |
| ***Interactions*** |
| Lennard-Jones (12-6) | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| Short-Range Coulomb | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| Long-Range Coulomb: Ewald Summation | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| Analytical Tail Correction | :heavy_check_mark: | :heavy_check_mark: |  |
| Machine-Learning Potential<br>(via LibTorch and cppFlow) | :heavy_check_mark: |  |  |
| ***Moves*** |
| Translation/Rotation | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| Configurational-Bias Monte Carlo (CBMC) | :heavy_check_mark: | :heavy_check_mark: |
| Widom test particle insertion | :heavy_check_mark: | :heavy_check_mark: |
| Insertion/Deletion<br>(without CBMC) | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| Insertion/Deletion<br>(with CBMC) | :heavy_check_mark: | :heavy_check_mark: |
| Identity Swap | :heavy_check_mark: |
| NVT-Gibbs volume change move | :heavy_check_mark: | :heavy_check_mark: |
| Gibbs particle transfer | :heavy_check_mark: | :heavy_check_mark: |
| Configurational Bias/<br>Continuous Fractional Components<br>(CB/CFC) MC | :heavy_check_mark: | :heavy_check_mark: |
| ***Extra Functionalities*** |
| Write: LAMMPS data file | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| Read: LAMMPS data file | :heavy_check_mark: |
| Write: Restart files<br>(Compatible with RASPA-2) | :heavy_check_mark: | :heavy_check_mark: |
| Read: Restart files | :heavy_check_mark: | :heavy_check_mark: |
| Peng-Robinson Equation of State | :heavy_check_mark: |
| Automatic Determination<br>of # unit cells | | | :heavy_check_mark: |


