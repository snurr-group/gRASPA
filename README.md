# gRASPA
gRASPA (pronounced “gee raspa”) is a GPU-accelerated Monte Carlo simulation software built for molecular adsorption in nanoporous materials, such as zeolites and metal-organic frameworks (MOFs). 

## Installation
### Installation in clusters
To install gRASPA on NERSC (DOE) and QUEST (Northwestern) clusters, check out [Cluster-Setup](Cluster-Setup/)

### Installation on local machines
A detailed installation note for gRASPA on Ubuntu 24.04 (with latest CUDA/NVHPC) is documented in the manual [here](https://zhaoli2042.github.io/gRASPA-mkdoc/Install.html)

### Compatible GPUs
* For NVIDIA GPUs, gRASPA is currently compatible with NVHPC 22.5 & 22.7 & 24.5.
* gRASPA code has been tested on the following NVIDIA GPUs:
  * A40, A100, RTX 3080 Ti, RTX 3090, RTX 4090.
* gRASPA has a SYCL version (experimental) that supports other devices, available in [Releases](https://github.com/snurr-group/gRASPA/releases)
## gRASPA Manual
gRASPA manual is available online at [here](https://zhaoli2042.github.io/gRASPA-mkdoc)

## Reference
* gRASPA paper is currently in progress. Please kindly cite it when it is published.
* Part of gRASPA is available in [Zhao Li's dissertation](https://www.proquest.com/openview/900e3899582bbe385d240586668e6f90/1?pq-origsite=gscholar&cbl=18750&diss=y)

## Table of Code Capabilities
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

## Authors
* Zhao Li (Northwestern University, currently at Purdue University)
* Kaihang Shi (Northwestern University, currently at University at Buffalo)
* David Dubbeldam (University of Amsterdam)
* Mark Dewing (Argonne National Laboratory)
* Christopher Knight (Argonne National Laboratory)
* Alvaro Vazquez Mayagoitia (Argonne National Laboratory)
* Randall Q. Snurr (Northwestern University)

