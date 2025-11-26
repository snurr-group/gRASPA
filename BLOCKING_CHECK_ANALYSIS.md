# Blocking Check Analysis

This document analyzes where blocking is checked in the gRASPA codebase and confirms that blocking is checked **after** we think the move is accepted, and if blocked, the move is rejected.

## Summary

The blocking check follows the correct flow:
1. Calculate acceptance probability (Pacc)
2. Check if move should be accepted (Random < Pacc) → `Accept = true`
3. **IF Accept is true, THEN check blocking**
4. **IF blocked, reject the move (set Accept = false)**
5. **IF not blocked, accept the move**

## Detailed Flow by Move Type

### 1. Single Body Moves (Translation, Rotation, Special Rotation, Single Insertion)

**Location**: `src_clean/mc_single_particle.h::SingleBody_Acceptance()` (lines 238-301)

**Flow**:
```
Line 253: Accept = false (initialize)
Line 255-260: Check if move should be accepted based on Pacc
  - If Random < Pacc → Accept = true
  
Line 262: if(Accept) {
  Line 264-275: Check if new position is blocked
    - For TRANSLATION, ROTATION, SPECIAL_ROTATION, SINGLE_INSERTION
    - Call CheckMoveBlocked()
    - If blocked:
      Line 271: Accept = false
      Line 272: SystemComponents.flag[0] = true
      Line 273: return (reject move)
  
  Line 277-294: If not blocked, proceed with acceptance
    - TRANSLATION/ROTATION/SPECIAL_ROTATION → AcceptTranslation()
    - SINGLE_INSERTION → AcceptInsertion()
}
```

**Key Function**: `CheckMoveBlocked()` (lines 204-236)
- Checks all atoms of the molecule at the new position
- Returns `true` if any atom is blocked
- Uses `BlockedPocket()` to check each atom position

### 2. CBMC Insertion Moves (INSERTION)

**Location**: `src_clean/move_struct.h::InsertionMove::Acceptance()` (lines 27-49)

**Flow**:
```
Line 33: Check if Random < Pacc → Accept = true

Line 35: if(Accept) {
  Line 38: AcceptInsertion() is called
  
  Inside AcceptInsertion() (mc_utilities.h:383-490):
    Lines 401-449: Check blocking BEFORE inserting
      - For INSERTION move type
      - Check all atoms of molecule to be inserted
      - If blocked:
        Line 447: SystemComponents.flag[0] = true
        Line 448: return (don't insert)
      - If not blocked:
        Line 452: Proceed with insertion
    
    Lines 454-484: Similar check for SINGLE_INSERTION
      - If blocked:
        Line 478: SystemComponents.flag[0] = true
        Line 479: return
  
  Back in InsertionMove::Acceptance():
    Line 41-45: Check if flag[0] is set
      - If flag[0] is true (blocked):
        Line 43: Accept = false
        Line 44: energy.zero()
}
```

**Key Function**: `AcceptInsertion()` (mc_utilities.h:383-490)
- Checks blocking for INSERTION and SINGLE_INSERTION move types
- Uses `BlockedPocket()` to check each atom position
- Sets `SystemComponents.flag[0] = true` if blocked

### 3. Single Insertion Note

For `SINGLE_INSERTION` moves, there are **two** blocking checks:
1. First check in `SingleBody_Acceptance()` (line 268) - checks before calling AcceptInsertion
2. Second check in `AcceptInsertion()` (lines 456-480) - checks again inside AcceptInsertion

This is redundant but safe - the first check will catch it and return early, so the second check won't be reached. However, the second check is still present for consistency.

## Blocking Check Functions

### `CheckMoveBlocked()` 
- **Location**: `src_clean/mc_single_particle.h` (lines 204-236)
- **Purpose**: Check if a move (translation/rotation) results in a blocked position
- **Used by**: `SingleBody_Acceptance()` for TRANSLATION, ROTATION, SPECIAL_ROTATION, SINGLE_INSERTION
- **Logic**: Checks all atoms of the molecule at the new position in `Sims.New`

### `BlockedPocket()`
- **Location**: `src_clean/read_data.cpp` (lines 3280-3348)
- **Purpose**: Check if a single position is blocked by block pockets
- **Used by**: `CheckMoveBlocked()` and `AcceptInsertion()`
- **Logic**: 
  - If `InvertBlockPockets` is true: molecule must be INSIDE block pockets
  - If `InvertBlockPockets` is false: molecule is blocked when INSIDE any block pocket

## Conclusion

✅ **Blocking is correctly checked AFTER the move is accepted** (based on Pacc)
✅ **If blocked, the move is rejected** (Accept is set to false)
✅ **If not blocked, the move is accepted** (proceeds with AcceptTranslation/AcceptInsertion)

The implementation follows the correct Monte Carlo acceptance-rejection pattern where blocking acts as an additional constraint after the energy-based acceptance criterion.

