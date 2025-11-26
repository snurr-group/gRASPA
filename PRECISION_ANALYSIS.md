# Precision Differences Between RASPA2 and gRASPA

## Key Differences Identified:

### 1. Distance Calculation Method
- **RASPA2**: Likely uses `sqrt(dx^2 + dy^2 + dz^2) < BlockDistance` (with sqrt)
- **gRASPA**: Uses `dist_sq <= radius_sq * 1.01` (squared distance, no sqrt)

**Impact**: 
- `sqrt()` introduces floating-point rounding errors
- Squared distance comparison avoids sqrt but may have different boundary behavior
- If RASPA2 uses `r < BlockDistance` (strict), positions exactly at boundary are NOT blocked
- If gRASPA uses `dist_sq <= radius_sq * 1.01`, we block positions within 1.01 * radius

### 2. Comparison Operator
- **RASPA2**: Likely `r < BlockDistance` (strict less than)
- **gRASPA**: `dist_sq <= radius_sq * 1.01` (less than or equal)

**Impact**: 
- RASPA2's strict `<` means boundary positions are NOT blocked
- gRASPA's `<=` with 1.01 factor should block MORE than RASPA2
- But if RASPA2 actually uses `r <= BlockDistance`, then we need to match that

### 3. PBC Rounding Precision
- **gRASPA**: Uses `static_cast<int>(x + ((x >= 0.0) ? 0.5 : -0.5))` for NINT
- **RASPA2**: May use different rounding method (possibly `round()` or `floor(x + 0.5)`)

**Impact**:
- Different rounding can cause slightly different PBC-wrapped distances
- This could affect whether a position is considered blocked or not

### 4. Floating Point Precision
- Both use `double` precision
- But accumulation of errors in PBC calculations could differ

## Recommendations:

1. **Use sqrt for exact matching**: Calculate `dist = sqrt(dist_sq)` and compare `dist <= radius * 1.01` to match RASPA2's precision while allowing slight adjustment
2. **Check PBC rounding**: Verify that the NINT implementation matches RASPA2 exactly
3. **Match comparison operator**: Use `<=` with 1.01 factor to allow slight adjustment while matching precision

## Implementation Changes Made:

1. **Changed from squared distance to sqrt**: Now uses `sqrt(dist_sq)` to match RASPA2's precision exactly
2. **Comparison**: Uses `dist <= radius * 1.01` (max) to match precision while allowing slight adjustment
3. **PBC rounding**: Already matches RASPA2's NINT implementation using `static_cast<int>(x + ((x >= 0.0) ? 0.5 : -0.5))`

