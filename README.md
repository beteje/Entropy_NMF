# Entropy_NMF
Nonnegative matrix factorization using a cross fuzzy entropy similarity measure

Assuming the structure _Y=WH_ there are 2 versions of the code which have different updates of the _H_ matrix: 

- The version in the paper ICASSP_2016 which updates column-wise:
  - _FuzzyEn_Mean_HCol_ calcluates the nonnegative matrix factorization of an input matrix _Y_ using a fuzzy entropy distance measure
  - _FuzzyEn_H_Update_Col_Mean_ calculates the current gradient of _H_ column-wise
- One which updates row-wise:
  - _FuzzyEn_Mean_NMF_ calcluates the nonnegative matrix factorization of an input matrix _Y_ using a fuzzy entropy distance measure
  - _FuzzyEn_H_Update_Mean_ calculates the current gradient of _H_ row-wise
 - Both use the following functions:
   - _FuzzyEn_W_Update_Mean_ calculates the current gradient of _W_
   - _CFuzzyEn_row_ calculates the cross fuzzy entropy between the rows of _Y_ and _WH_

Requires the use of codegen to produce the mex files

## References
ICASSP_2016 [Fuzzy Entropy Based Nonnegative Matrix Factorization for Muscle Synergy Extraction](https://beteje.github.io/assets/pdf/2016_ICASSP.pdf) 
