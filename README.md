# gRASPA
gRASPA (pronounced “gee raspa”) is a GPU-accelerated Monte Carlo simulation software built for molecular adsorption in nanoporous materials, such as zeolites and metal-organic frameworks (MOFs). 

<div align="center">
  
<a href="">![Tests](https://github.com/snurr-group/gRASPA/actions/workflows/python-test.yml/badge.svg)</a>
<a href="">![License](https://img.shields.io/github/license/Zhaoli2042/gRASPA_fork?logo=github)</a>

<a href="#">[![Manual](https://img.shields.io/badge/User_Manual-red?logo=github)]( https://zhaoli2042.github.io/gRASPA-mkdoc)</a>
<a href="#">[![Manual](https://img.shields.io/badge/用户手册-yellow?logo=github)]( https://zhaoli2042.github.io/gRASPA-mkdoc/Chinese/)</a>
<a href="#">[![Doxygen](https://img.shields.io/badge/Doxygen_Manual-green?logo=doxygen)]( https://zhaoli2042.github.io/gRASPA/annotated.html)</a>

<a href="#"><img src="https://img.shields.io/github/repo-size/snurr-group/gRASPA?logo=github" alt="GitHub repo size in bytes" /></a>
<a href="#"> [![Lines](https://tokei.rs/b1/github/snurr-group/gRASPA?category=code)](https://github.com/XAMPPRocky/tokei)</a>
<a href="#"> ![STARS](https://img.shields.io/github/stars/snurr-group/gRASPA?logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAFAAAABLCAYAAADnAAD1AAAAAXNSR0IArs4c6QAAAARnQU1BAACxjwv8YQUAAAAJcEhZcwAADsMAAA7DAcdvqGQAAAPMSURBVHhe7ZxNbtNAFMfdjzSAyqoCia5YgQRixQV6ifYQcAjKIeAQ5BBwAVYIJFixohIfK6rSJP3yf/CLYsd2xvPeeN7Y/klR7USKpr/8Z95kxvHG2fnsOhlwRq3A+Z8PydXfz+Z4e+8g2RjfTzZ37plzTagUOP3+xghcZvTgMBnvH2VnetjM/qph+uPdijwwP5mY17ShTiBEVXGJbj37lZ3pQJXAdQmDvPnv99mZDlQJrEsfoS2FagTajm/aUqhGoE36CE0pVCGwaXWFvMtsjhgaFQKRqKbMU+kaUhhcINLnIkJLCoMLdEkfoSGFQQW6po/QkMKgAjnpI0KnMJhAbvqI0CkMJlAifUTIFAYRKJU+ImQKgwiUTB9x4eE9bWhdoFlp9tDdkMCydUTftC8w7b6+CJHCVgUiJT4H+xAp9L4nQtKupj/N2OdTIMDG09beQbJ996nZiKLnfCEikKogJCWQlT6u02PfsmyBwMWuXvqQlGstkCSZ4yxVmiS5spC7+8ScN5VbKhCiaECmLthHluVC7Fb6KLIi8N+340XSBvJA4Pjhy1wyc1UY3xAGedXATXE/JifQxzeErlF01PpEumvkBJYNkgN5qEITOYGj/SP2vKjroIgskxMIeZjFDxLLuf3oeMVN5TwQl5j1df5XBuSVDXGlRaRsvtNnquSByio8SPxPnTxQKRD0XeI6eaBWIOirRBt5oNFqzPnXV50vLAgKAmMjD6xNIIE3vvX4daeT2FQesBYIuizRRR5oJBB0UaKrPNBYIOiSRI48YF1Eyoi9sEDenWdvszM3nBJIxJxECXmAJRDEKFFKHmALBDFJlJQHRAQCNKy42KgR6TaKCQQxbEhJt1FMYEyVWLKtYgJxlUIsSKZQTOBFBN3XB3JjYExdWGMChzGQSQwVmEBbpSSKCIwpfdKICIwpfYRUm2USGNEUhpAqJDICT79kR/1j6MJM2AJjLSBot0TbRRLoE+1LZGyBvrov7pW1+3xi1u5Gnq4Yk2g7vwsLV2DIgrTlG41h0wcLttIiJSoxX6BQBV6IS2WVScJzeA0PKZESYyBrVw6cfjzMjtyADNwfsOm2Irrf7GTC6ob4ELjL++xtzbNPL7KzZkDYTjrOue7HElyREMhJc+sCpcQVgUCXq2ptr8KqgjUGNlmFRiPRWG6Dq8B7Ik14f4nx0Rb2GIgE1n3q+Gdw9T/GujbB74ZtbkaBqRIHtkD8PKzqzmtUMUNSJ1LivqxsgWC5kUZYmjptN4xFG/ELVAw72BtG5Zf4cEUE9hn134W1MwhkkSQ35F9FEWClhrQAAAAASUVORK5CYII=)</a>

</div>

## Installation
### Installation in clusters
To install gRASPA on NERSC (DOE) and QUEST (Northwestern) clusters, check out [Cluster-Setup](Cluster-Setup/)

### Installation on local machines
A detailed installation note for gRASPA on CentOS/Ubuntu 24.04 is documented in the manual [here](https://zhaoli2042.github.io/gRASPA-mkdoc/Installation.html)

### Compatible GPUs
* For NVIDIA GPUs, gRASPA code has been tested on the following NVIDIA GPUs:
  * A40, A100, RTX 3080 Ti, RTX 3090, RTX 4090.
  * 🤯: RTX 3090/4090 is faster than A40/A100 for gRASPA
* gRASPA has a SYCL version (experimental) that supports other devices, available in [Releases](https://github.com/snurr-group/gRASPA/releases)
## Quick Start
* Go to [```Examples/```](Examples/) folder and read more!

## gRASPA Manual
* gRASPA manual is available online @ https://zhaoli2042.github.io/gRASPA-mkdoc
  * also available in [Chinese](https://zhaoli2042.github.io/gRASPA-mkdoc/Chinese)
* a doxygen documentation is also available @ https://zhaoli2042.github.io/gRASPA
## Reference
* gRASPA paper is now published. Please kindly cite it if you find it useful.
  * Li, Shi, Dubbeldam, Dewing, Knight, Vázquez-Mayagoitia, Snurr, 
    "Efficient Implementation of Monte Carlo Algorithms on Graphical Processing Units for Simulation of Adsorption in Porous Materials", _J. Chem. Theory Comput_. 2024, 20, 23, 10649–10666
  * DOI: [10.1021/acs.jctc.4c01058](https://pubs.acs.org/doi/10.1021/acs.jctc.4c01058)
* Also, please give our repository a :star: if our code helps!
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
* Zhao Li (Northwestern University, currently at Purdue University/University of Notre Dame, zhaoli2023@u.northwestern.edu)
* Kaihang Shi (Northwestern University, currently at University at Buffalo, kaihangs@buffalo.edu)
* David Dubbeldam (University of Amsterdam, d.dubbeldam@uva.nl)
* Mark Dewing (Argonne National Laboratory, markdewing@yahoo.com)
* Christopher Knight (Argonne National Laboratory, knightc@anl.gov)
* Alvaro Vazquez Mayagoitia (Argonne National Laboratory, vama@alcf.anl.gov)
* Randall Q. Snurr (Northwestern University, snurr@northwestern.edu)

