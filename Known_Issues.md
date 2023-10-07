# What? Issues? 
  * This file records the issues that I encountered.
  * Be not scared. With different combinations of keywords, you can easily reach no-man's land!
  * For me, I try to break things very often. So these issues might never appear to you, but worth keeping track of.
  * These issues go with versions of gRASPA. So what are listed here only reflects the issues with the **latest** version.
  * Here, I record things that I tried:
# Issues
  * When combining "TURN_OFF_CBMC_SWAP yes" with an empty-box (empty-box.cif) with no framework atoms, there is error about initializing ewald summation.
    * Really? Who TF uses single swap moves together with an empty box?
    * No issue if you do non-CBMC swap + MOF, or you do CBMC swap + empty-box
