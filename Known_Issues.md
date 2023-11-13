# What? Issues? 
  * This file records the issues that I encountered.
  * Be not scared. With different combinations of keywords, you can easily reach no-man's land!
  * For me, I try to break things very often. So these issues might never appear to you, but worth keeping track of.
  * These issues go with versions of gRASPA. So what are listed here only reflects the issues with the **latest** version.
  * Here, I record things that I tried:
# Issues
  * When combining "TURN_OFF_CBMC_SWAP yes" with an empty-box (empty-box.cif) with no framework atoms, there is error about initializing ewald summation.
    * **FIXED** See commit on 11/13/2023: [**LINK**](https://github.com/snurr-group/CUDA-RASPA-DeepPotential/commit/ab7a890583f25aabc574df586e4c85c55c59a14f)
