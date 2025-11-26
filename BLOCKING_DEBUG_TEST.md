# Blocking Check Debugging and Testing

## Issues to Test

1. **Component Indexing**: Verify SelectedComponent matches UseBlockPockets indexing
2. **Flag Preservation**: Ensure BlockedByPockets persists through energy calculation
3. **All Atoms Checked**: Verify all atoms of molecule are checked
4. **Distance Calculation**: Verify CheckBlockedPosition logic
5. **Edge Cases**: Empty vectors, out of bounds, etc.

## Test Cases

### Test 1: Component Indexing
- For adsorbate components: SelectedComponent should be >= NComponents.y
- Block pockets stored at: NComponents.y + AdsorbateComponent
- Need to verify: SelectedComponent == (NComponents.y + AdsorbateComponent) for adsorbates

### Test 2: Flag Flow
- Prepare: Sets BlockedByPockets = true if blocked
- Calculation: Should preserve BlockedByPockets and set flag[0] = true
- Acceptance: Should check flag[0] and reject if true

### Test 3: All Atoms Checked
- For polyatomic molecules, all atoms must be checked
- If ANY atom is blocked, entire molecule is blocked

### Test 4: Edge Cases
- Empty UseBlockPockets vector
- Component index out of bounds
- Empty BlockPocketCenters
- CUDA memory copy failures

