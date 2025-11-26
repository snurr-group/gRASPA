# Probability Discrepancy Analysis: gRASPA vs RASPA2

## Summary
The discrepancy in probability display between gRASPA and RASPA2 is **NOT due to code changes** - it's a display format difference. The actual move selection logic is identical and correct.

## Comparison

### gRASPA Output (personal.log):
```
ACCUMULATED Probabilities:
Translation Probability:      0.25000
Rotation Probability:         0.50000
Special Rotation Probability: 0.50000
Widom Probability:            0.50000
Reinsertion Probability:      0.75000
Identity Swap Probability:    0.75000
CBCF Swap Probability:        0.75000
Swap Probability:             1.00000
```

### RASPA2 Output (reference):
```
Percentage of rotation moves:                      25.000000
Percentage of reinsertion moves:                   25.000000
Percentage of swap (insert/delete) moves:          25.000000
```

## Analysis

### 1. NormalizeProbabilities Function
**Status**: Identical in both current code and commit f0c86ea

The function:
1. Sums all probabilities: `TotalProb = Translation + Rotation + Reinsertion + Swap = 4.0`
2. Normalizes each: `Translation = 1.0/4.0 = 0.25`, etc.
3. Accumulates for move selection: 
   - `TranslationProb = 0.25`
   - `RotationProb = 0.25 + 0.25 = 0.50`
   - `ReinsertionProb = 0.50 + 0.25 = 0.75`
   - `SwapProb = 0.75 + 0.25 = 1.00`

### 2. Move Selection Logic (axpy.cu)
**Status**: Correct - uses accumulated probabilities

```cpp
if(RANDOMNUMBER < TranslationProb)        // 0.0 to 0.25 -> 25%
  MoveType = TRANSLATION;
else if(RANDOMNUMBER < RotationProb)      // 0.25 to 0.50 -> 25%
  MoveType = ROTATION;
else if(RANDOMNUMBER < ReinsertionProb)    // 0.50 to 0.75 -> 25%
  MoveType = REINSERTION;
else if(RANDOMNUMBER < SwapProb)           // 0.75 to 1.00 -> 25%
  MoveType = SWAP;
```

This correctly gives 25% probability to each move type.

### 3. Display Format Difference

**gRASPA**: Shows cumulative probabilities (for debugging/verification)
- Translation: 0.25 (25% of total)
- Rotation: 0.50 (cumulative: 25% + 25%)
- Reinsertion: 0.75 (cumulative: 25% + 25% + 25%)
- Swap: 1.00 (cumulative: all moves)

**RASPA2**: Shows individual percentages (user-friendly)
- Translation: 25%
- Rotation: 25%
- Reinsertion: 25%
- Swap: 25%

## Conclusion

**The discrepancy is NOT due to code changes.** Both versions:
1. Use identical `NormalizeProbabilities()` function
2. Use identical move selection logic
3. Produce identical move probabilities (25% each)

The only difference is the **display format**:
- gRASPA shows cumulative probabilities (useful for debugging)
- RASPA2 shows individual percentages (more user-friendly)

**The actual simulation behavior should be identical.** If there's a discrepancy in adsorbed molecule counts, it's likely due to:
1. Blocking logic differences (which we've been fixing)
2. Random number generation differences
3. Numerical precision differences
4. Other simulation parameters

The probability normalization is **not the source of the discrepancy**.

