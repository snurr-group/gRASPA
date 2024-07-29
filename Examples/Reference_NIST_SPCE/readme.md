# NIST reference calculation
* Comparing gRASPA energies to NIST reference calculation
* SPC/E water in triclinic boxes
* Read more about this [here](https://www.nist.gov/mml/csd/chemical-informatics-group/spce-water-reference-calculations-non-cuboid-cell-10a-cutoff)
* Results can be seen in Box-1/result, for example.
## Energy Comparison
| Configurations | Box 1<br>NIST/gRASPA | Box 2<br>NIST/gRASPA | Box 3<br>NIST/gRASPA | Box 4<br>NIST/gRASPA |
|:--------------:|:------------------------:|:---------------------:|:-----------------------:|:---------------------:|
| $`E_{Disp}/{k_B}`$ (K)       | 111992 / 111992 |  43286 / 43286  | 14403.3 / 14403.3 | 25025.1 / 25025.1 |
| $`E_{Tail}/{k_B}`$ (K)       | -4109.19 / -4109.19 | -2105.61 / -2105.61 | -1027.3 / -1027.3 | -163.091 / -163.091 |
| $`E_{Real}/{k_B}`$ (K)       | -727219 / -727219 | -476902 / -476902 | -297129 / -297129 | -171462 / -171462 | 
| Number of Wave Vectors       | 831 / 831 | 1068 / 1068 | 838 / 838 | 1028 / 1028 |
| $`E_{Fourier}/{k_B}`$ (K)    | 44677 / 44677 | 44409.4 / 44409.7 | 28897.4 / 28897.5 | 22337.2 / 22337.8 |
| $`E_{Self}/{k_B}`$ (K)       | -11581958 / -11582033 | -8686468 / -8686525 | -5790979 / -5791017 | -2895489 / -2895508 |
| $`E_{Intra}/{k_B}`$ (K)      | 11435363 / 11435437| 8576522 / 8576578 | 5717681 / 5717719 | 2858841 / 2858859 |
| $`E_{Self+Intra}/{k_B}`$ (K) | -146595 / -146596 | -109946 / -109947 | -73298 / -73298 | -36648 / -36649 |
| $`E_{Total}/{k_B}`$ (K)      | -721254 / -721255 | -501259 / -501259 | -328153 / -328153 | -160912 / -160912 |
