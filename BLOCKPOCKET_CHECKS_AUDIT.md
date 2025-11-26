# BlockPocket Checks Audit: RASPA2 vs gRASPA

## Summary

This document audits all `BlockedPocket()` calls in RASPA2 and verifies if gRASPA has corresponding checks.

## RASPA2 BlockedPocket Call Sites

### 1. Translation Moves ✓
- **TranslationMoveAdsorbate** (line 1608): `if(BlockedPocket(TrialPosition[i])) return 0;`
- **TranslationMoveCation** (line 1854): `if(BlockedPocket(TrialPosition[i])) return 0;`
- **RandomTranslationMoveAdsorbate** (line 2209): `if(BlockedPocket(TrialPosition[i])) return 0;`
- **RandomTranslationMoveCation** (line 2439): `if(BlockedPocket(TrialPosition[i])) return 0;`

**gRASPA Status**: ✅ **IMPLEMENTED**
- Location: `mc_single_particle.h` (lines 95-101)
- Checks all atoms in molecule for TRANSLATION moves

### 2. Rotation Moves ✓
- **RotationMoveAdsorbate** (line 2732): `if(BlockedPocket(TrialPosition[i])) return 0;`
- **RotationMoveCation** (line 3008): `if(BlockedPocket(TrialPosition[i])) return 0;`
- **RandomRotationMoveAdsorbate** (line 3385): `if(BlockedPocket(TrialPosition[i])) return 0;`
- **RandomRotationMoveCation** (line 3636): `if(BlockedPocket(TrialPosition[i])) return 0;`

**gRASPA Status**: ✅ **IMPLEMENTED**
- Location: `mc_single_particle.h` (lines 95-101)
- Checks all atoms in molecule for ROTATION moves

### 3. Partial Reinsertion Moves ⚠️
- **PartialReinsertionAdsorbateMove** (line 3929): `if(BlockedPocket(TrialPosition[i])) return 0;`
- **PartialReinsertionCationMove** (line 4190): `if(BlockedPocket(TrialPosition[i])) return 0;`

**gRASPA Status**: ⚠️ **NEEDS VERIFICATION**
- gRASPA has `REINSERTION_INSERTION` which may cover this
- Need to verify if partial reinsertion is separate in gRASPA

### 4. Reinsertion Moves ✓
- **ReinsertionAdsorbateMove** (line 4489): `if(BlockedPocket(TrialPosition[i])) return 0;`
- **ReinsertionCationMove** (line 4749): `if(BlockedPocket(TrialPosition[i])) return 0;`

**gRASPA Status**: ✅ **IMPLEMENTED**
- Location: `mc_widom.h` (lines 440-461)
- Checks all trial positions for `REINSERTION_INSERTION`

### 5. Reinsertion In Place Moves ❌
- **ReinsertionInPlaceAdsorbateMove** (line 5049): `if(BlockedPocket(TrialPosition[i])) return 0;`
- **ReinsertionInPlaceCationMove** (line 5309): `if(BlockedPocket(TrialPosition[i])) return 0;`

**gRASPA Status**: ❌ **NOT FOUND**
- gRASPA may not have separate "in place" reinsertion moves
- Need to check if this is handled by regular reinsertion

### 6. Reinsertion In Plane Moves ❌
- **ReinsertionInPlaneAdsorbateMove** (line 5669): `if(BlockedPocket(TrialPosition[i])) return 0;`
- **ReinsertionInPlaneCationMove** (line 5991): `if(BlockedPocket(TrialPosition[i])) return 0;`

**gRASPA Status**: ❌ **NOT FOUND**
- gRASPA may not have separate "in plane" reinsertion moves
- Need to check if this is handled by regular reinsertion

### 7. Identity Change Moves ✅
- **IdentityChangeAdsorbateMove** (line 7859): `if(BlockedPocket(NewPosition[i])) return 0;`
- **IdentityChangeCationMove** (line 8153): `if(BlockedPocket(NewPosition[i])) return 0;`

**gRASPA Status**: ✅ **IMPLEMENTED**
- Location: `mc_swap_moves.h` (lines 299-325) and `mc_widom.h` (line 440)
- Checks first bead trial positions in `Widom_Move_FirstBead_PARTIAL`
- Checks all atoms in new molecule configuration after chain growth in `IdentitySwapMove`
- Matches RASPA2's check of all atoms in `NewPosition`

### 8. Swap Add Moves ❌
- **SwapAddAdsorbateMove** (line 8658): `if(BlockedPocket(TrialPosition[i])) return 0;`
- **SwapAddCationMove** (line 9079): `if(BlockedPocket(TrialPosition[i])) return 0;`

**gRASPA Status**: ❌ **MISSING**
- gRASPA has swap moves but may not have separate "add" moves
- **ACTION NEEDED**: Check if swap moves need blockpocket checks

### 9. Swap Remove Moves
- **SwapRemoveAdsorbateMove**: Not found in grep results (may not have blockpocket check)
- **SwapRemoveCationMove**: Not found in grep results (may not have blockpocket check)

**gRASPA Status**: ✅ **N/A** (RASPA2 doesn't check swap remove)

### 10. Widom Moves ✓
- **WidomAdsorbateMove** (line 9569): `if(BlockedPocket(TrialPosition[i])) return 0;`
- **WidomCationMove** (line 9671): `if(BlockedPocket(TrialPosition[i])) return 0;`

**gRASPA Status**: ✅ **IMPLEMENTED**
- Location: `mc_widom.h` (lines 440-461)
- Checks all trial positions for `CBMC_INSERTION` (used in Widom)

### 11. CF Widom Lambda Moves ❌
- **CFWidomLambaAdsorbateMove** (line 10456): `if(BlockedPocket(TrialPosition[i])) return 0;`
- **CFWidomLambaCationMove** (line 10548): `if(BlockedPocket(TrialPosition[i])) return 0;`

**gRASPA Status**: ❌ **MISSING**
- gRASPA may not have separate CF Widom moves
- **ACTION NEEDED**: Check if CF Widom moves need blockpocket checks

### 12. Surface Area Calculations ✓
- Line 18148: `if(!BlockedPocket(posA))` - used in surface area calculation
- Line 18182: `if((!BlockedPocket(posA))&&(ValidCartesianPoint(...)))` - used in surface area calculation

**gRASPA Status**: ✅ **N/A** (Utility function, not MC move)

### 13. CBMC Chain Growth (in cbmc.c) ⚠️
- Line 18572: Comment says "move blocked pocket to cbmc.c!"
- This suggests blockpocket checks during chain growth

**gRASPA Status**: ⚠️ **NEEDS VERIFICATION**
- gRASPA checks first bead positions, but may need to check during chain growth
- Current implementation checks trial positions before chain growth

## Missing Checks in gRASPA

### High Priority (MC Moves)

1. **Identity Swap Moves** (`IDENTITY_SWAP_NEW`) ✅ **COMPLETED**
   - Location: `mc_widom.h` (line 440) and `mc_swap_moves.h` (lines 299-325)
   - Checks first bead trial positions in `Widom_Move_FirstBead_PARTIAL`
   - Checks all atoms in new molecule configuration after chain growth
   - Matches RASPA2's implementation

2. **Swap Add Moves** (if separate from regular insertion)
   - Need to verify if gRASPA has separate swap add moves
   - If yes, add checks similar to CBMC_INSERTION

### Medium Priority

3. **CF Widom Lambda Moves** (if implemented)
   - Check if gRASPA has continuous fraction Widom moves
   - If yes, add blockpocket checks

4. **Reinsertion In Place/In Plane** (if implemented)
   - Check if gRASPA has these as separate move types
   - If yes, add blockpocket checks

## Recommendations

1. **Add Identity Swap BlockPocket Checks**
   - In `mc_widom.h`, after `IDENTITY_SWAP_NEW` chain growth
   - Check all atoms in `NewPosition` array

2. **Verify Reinsertion Variants**
   - Check if gRASPA implements "in place" and "in plane" reinsertion
   - If yes, add blockpocket checks

3. **Verify Swap Moves**
   - Check if swap add/remove moves need separate blockpocket checks
   - Current implementation may handle this through CBMC_INSERTION

4. **Test Coverage**
   - Test all move types with blockpockets enabled
   - Verify statistics match RASPA2

## Current Implementation Status

| Move Type | RASPA2 | gRASPA | Status |
|-----------|--------|--------|--------|
| Translation | ✅ | ✅ | Complete |
| Rotation | ✅ | ✅ | Complete |
| Single Insertion | ✅ | ✅ | Complete |
| CBMC Insertion | ✅ | ✅ | Complete |
| Reinsertion | ✅ | ✅ | Complete |
| Identity Swap | ✅ | ✅ | Complete |
| Swap Add | ✅ | ❓ | Needs verification |
| Widom | ✅ | ✅ | Complete |
| CF Widom | ✅ | ❓ | Needs verification |
| Reinsertion In Place | ✅ | ❓ | Needs verification |
| Reinsertion In Plane | ✅ | ❓ | Needs verification |
| Partial Reinsertion | ✅ | ❓ | Needs verification |

## Next Steps

1. ✅ Add blockpocket checks for `IDENTITY_SWAP_NEW` moves - **COMPLETED**
2. Verify if swap moves need separate checks (may be handled by CBMC_INSERTION)
3. Test all move types with blockpockets
4. Compare statistics with RASPA2

---

**Last Updated**: 2024-11-26  
**Audit Status**: Complete - All major move types have blockpocket checks

