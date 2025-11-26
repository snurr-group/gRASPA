# Blocking Check Test Checklist

## Code Flow Verification

### Phase 1: SingleBody_Prepare()
- [x] BlockedByPockets initialized to false
- [x] Check only for TRANSLATION, ROTATION, SPECIAL_ROTATION, SINGLE_INSERTION
- [x] Verify component indexing (SelectedComponent >= NComponents.y for adsorbates)
- [x] Check all atoms of molecule (not just first atom)
- [x] Set BlockedByPockets = true if ANY atom is blocked
- [x] Increment statistics correctly (once per blocked move)
- [x] Handle CUDA copy failures (block move if can't verify)

### Phase 2: SingleBody_Calculation()
- [x] Energy calculation runs first (may set device_flag for overlaps)
- [x] device_flag copied to SystemComponents.flag
- [x] BlockedByPockets preserved and combined with flag[0]
- [x] flag[0] = true if BlockedByPockets = true (AFTER energy calculation)

### Phase 3: SingleBody_Acceptance()
- [x] Check flag[0] before acceptance
- [x] If flag[0] = true, skip acceptance (move rejected)
- [x] If flag[0] = false, proceed with Pacc check
- [x] Validation: Verify BlockedByPockets matches flag[0]

## Component Indexing Verification

### For Adsorbate Components:
- SelectedComponent range: [NComponents.y, NComponents.x - 1]
- Block pockets stored at: SelectedComponent (which is NComponents.y + adsorbateIndex)
- UseBlockPockets[SelectedComponent] should match
- BlockPocketCenters[SelectedComponent] should exist

### Test Cases:
1. **Single adsorbate component (index NComponents.y)**
   - SelectedComponent = NComponents.y
   - Block pockets at index NComponents.y ✓

2. **Multiple adsorbate components**
   - SelectedComponent = NComponents.y + 1
   - Block pockets at index NComponents.y + 1 ✓

## Edge Cases to Test

1. **Empty UseBlockPockets vector**
   - Should skip blocking check ✓

2. **Component index out of bounds**
   - Check: `SelectedComponent < UseBlockPockets.size()` ✓

3. **Block pockets enabled but no data**
   - Warning printed, don't block ✓

4. **CUDA copy failure**
   - Block move (conservative) ✓

5. **Polyatomic molecule (multiple atoms)**
   - All atoms checked ✓
   - If ANY atom blocked, entire molecule blocked ✓

6. **Monatomic molecule (single atom)**
   - Single atom checked ✓

## Statistics Verification

- BlockPocketTotalAttempts incremented for every move attempt
- BlockPocketBlockedCount incremented only once per blocked move
- Statistics logged every 10,000 moves

## Flag Logic Verification

- flag[0] = true means blocked/rejected ✓
- flag[0] = false means not blocked (can accept) ✓
- BlockedByPockets = true → flag[0] = true ✓
- Acceptance check: `if(!flag[0] || !CheckOverlap)` ✓

## Potential Issues to Monitor

1. **Component Index Mismatch**
   - If block pockets stored at wrong index, blocking won't work
   - Debug output will show warnings

2. **Flag Not Preserved**
   - If BlockedByPockets not preserved through calculation
   - Validation check will catch this

3. **Statistics Incorrect**
   - If counting wrong, debug output will show

4. **Distance Calculation**
   - CheckBlockedPosition uses `dist_sq < radius_sq`
   - This means positions INSIDE block pocket are blocked
   - Verify this matches RASPA2 behavior

## Debug Output to Monitor

1. **Component Indexing Warnings**
   - "WARNING: BlockPockets enabled but no block pocket data loaded!"

2. **CUDA Errors**
   - "ERROR: Failed to copy atom position"
   - "BLOCKING: Move blocked due to CUDA copy failure"

3. **Blocking Events** (every 10,000 moves)
   - "DEBUG BLOCKING: Component X, MoveType Y, Blocked at atom Z/W"
   - "DEBUG BLOCKING STATS: Component X, Total attempts: Y, Blocked: Z (P%)"

4. **Flag Validation** (every 10,000 moves)
   - "DEBUG FLAG: BlockedByPockets=true, wasOverlap=X, flag[0] now=true"
   - "DEBUG REJECT: Component X, MoveType Y, Rejected: flag[0]=X, BlockedByPockets=Y"

5. **Flag Mismatch Errors**
   - "ERROR: BlockedByPockets=true but flag[0]=false! This should not happen!"

## Expected Results

1. **Blocking should work correctly**
   - Moves to blocked positions are rejected
   - Statistics show reasonable blocking percentage

2. **No false positives**
   - Moves to non-blocked positions are not rejected
   - Flag validation passes

3. **Statistics are accurate**
   - Total attempts matches move attempts
   - Blocked count matches actual rejections

4. **Performance acceptable**
   - Debug output doesn't slow down simulation significantly
   - CUDA copies are efficient

