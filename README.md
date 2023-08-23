# gRASPA-DeepPotential version 
This is an implementation of ML potential in gRASPA. Depending on ML models, it requires TensorFlow C++ API, CppFlow package, and LibTorch.

# For installation of gRASPA on various machines (NERSC, Quest of Northwestern, etc.), see the Cluster-Setup folder.

# Special NOTE:
1. gRASPA checks strictly for the accuracy of the input files. Make sure you run a small test case (pilot run) before running anything long!
2. The code will terminate if ANY unknown keywords are in the simulation.input file.
3. The code will check for the required keywords for a simulation and will terminate if it cannot find them.
4. (FUTURE) For testing input file correctness, we may add the functionality for a CPU-only pilot run.
