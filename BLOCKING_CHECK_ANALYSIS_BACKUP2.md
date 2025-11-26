# Blocking Check Analysis - Backup2 Version

This document analyzes where blocking is currently checked in the backup2 version and identifies where it should be checked according to the requirement: **blocking should be checked AFTER we think the move is accepted, and if blocked, reject it; otherwise accept it**.

## Current Implementation in Backup2

### 1. Single Insertion Moves (SINGLE_INSERTION)

**Current Location**: `src_clean/mc_single_particle.h::SingleBody_Prepare()` (lines 82-103)

**Current Flow**:
```
1. SingleBody_Prepare() is called
2. Generate new position (line 79)
3. Check blocking (lines 82-103) - BEFORE acceptance decision
   - If blocked, set device_flag[0] = true
4. SingleBody_Calculation() calculates energy and Pacc
5. SingleBody_Acceptance() checks acceptance
   - Line 243: if(!SystemComponents.flag[0] || !CheckOverlap)
   - Line 246: if(Random < Pacc) Accept = true
   - Line 250: if(Accept) { AcceptInsertion() }
```

**Problem**: Blocking is checked **BEFORE** the acceptance decision, not after. The flag is set during preparation, which affects the acceptance check.

**Should Be**: Blocking should be checked **AFTER** `Accept = true` is set, and if blocked, set `Accept = false`.

### 2. CBMC Insertion Moves (INSERTION)

**Current Location**: `src_clean/mc_widom.h::Widom_Move_FirstBead_PARTIAL()` (lines 438-464)

**Current Flow**:
```
1. Widom_Move_FirstBead_PARTIAL() generates trial positions
2. Check blocking for each trial (lines 438-464) - DURING trial generation
   - If blocked, set device_flag[i] = true for that trial
3. Select trial based on Rosenbluth weights (blocked trials have weight 0)
4. InsertionMove::Acceptance() checks acceptance
   - Line 33: if(Random < Pacc) Accept = true
   - Line 35: if(Accept) { AcceptInsertion() }
```

**Problem**: Blocking is checked **DURING** trial generation, which means blocked trials are excluded from the Rosenbluth weight calculation. This is actually correct for CBMC, but the final acceptance should still check blocking after acceptance.

**Should Be**: After `Accept = true`, check if the selected trial position is blocked, and if so, reject.

### 3. Translation/Rotation Moves (TRANSLATION, ROTATION, SPECIAL_ROTATION)

**Current Location**: **NO BLOCKING CHECK FOUND**

**Current Flow**:
```
1. SingleBody_Prepare() generates new position
2. SingleBody_Calculation() calculates energy and Pacc
3. SingleBody_Acceptance() checks acceptance
   - Line 246: if(Random < Pacc) Accept = true
   - Line 250: if(Accept) { AcceptTranslation() }
```

**Problem**: **No blocking check exists** for translation/rotation moves.

**Should Be**: After `Accept = true`, check if the new position is blocked, and if so, reject.

## Required Changes

### For Single Body Moves (Translation, Rotation, Special Rotation, Single Insertion)

**Location**: `src_clean/mc_single_particle.h::SingleBody_Acceptance()`

**Current Code** (lines 250-271):
```cpp
if(Accept)
{
  switch(MoveType)
  {
    case TRANSLATION: case ROTATION: case SPECIAL_ROTATION:
    {
      AcceptTranslation(Vars, systemId);
      break;
    }
    case SINGLE_INSERTION:
    {
      AcceptInsertion(Vars, Vars.SystemComponents[systemId].CBMC_New[0], systemId, SINGLE_INSERTION);
      break;
    }
    case SINGLE_DELETION:
    {
      AcceptDeletion(Vars, systemId, SINGLE_DELETION);
      break;
    }
  }
  SystemComponents.Moves[SelectedComponent].Record_Move_Accept(MoveType);
}
```

**Should Be**:
```cpp
if(Accept)
{
  // Check blocking AFTER we think the move is accepted
  bool isBlocked = false;
  if(MoveType == TRANSLATION || MoveType == ROTATION || MoveType == SPECIAL_ROTATION || MoveType == SINGLE_INSERTION)
  {
    // Check if new position is blocked
    size_t& start_position = SystemComponents.TempVal.start_position;
    size_t molsize = SystemComponents.Moleculesize[SelectedComponent];
    
    // Check all atoms of the molecule
    for(size_t i = 0; i < molsize; i++)
    {
      double3 atomPos;
      cudaMemcpy(&atomPos, &Sims.New.pos[start_position + i], sizeof(double3), cudaMemcpyDeviceToHost);
      if(CheckBlockedPosition(SystemComponents, SelectedComponent, atomPos, Sims.Box))
      {
        isBlocked = true;
        break;
      }
    }
  }
  
  if(isBlocked)
  {
    // Reject the move if blocked
    Accept = false;
    SystemComponents.flag[0] = true;
    return;
  }
  
  // If not blocked, proceed with acceptance
  switch(MoveType)
  {
    case TRANSLATION: case ROTATION: case SPECIAL_ROTATION:
    {
      AcceptTranslation(Vars, systemId);
      break;
    }
    case SINGLE_INSERTION:
    {
      AcceptInsertion(Vars, Vars.SystemComponents[systemId].CBMC_New[0], systemId, SINGLE_INSERTION);
      break;
    }
    case SINGLE_DELETION:
    {
      AcceptDeletion(Vars, systemId, SINGLE_DELETION);
      break;
    }
  }
  SystemComponents.Moves[SelectedComponent].Record_Move_Accept(MoveType);
}
```

**Also Need**: Remove blocking check from `SingleBody_Prepare()` (lines 82-103) for SINGLE_INSERTION.

### For CBMC Insertion Moves (INSERTION)

**Location**: `src_clean/move_struct.h::InsertionMove::Acceptance()`

**Current Code** (lines 27-43):
```cpp
void Acceptance(Variables& Vars, size_t systemId)
{
  Components& SystemComponents = Vars.SystemComponents[systemId];

  double RANDOM = 1e-100;
  if(!CreateMoleculePhase) RANDOM = Get_Uniform_Random();
  if(RANDOM < Pacc) Accept = true;

  if(Accept)
  {
    AcceptInsertion(Vars, InsertionVariables, systemId, INSERTION);
  }
  else
    energy.zero();
}
```

**Should Be**:
```cpp
void Acceptance(Variables& Vars, size_t systemId)
{
  Components& SystemComponents = Vars.SystemComponents[systemId];
  Simulations& Sims = Vars.Sims[systemId];

  double RANDOM = 1e-100;
  if(!CreateMoleculePhase) RANDOM = Get_Uniform_Random();
  if(RANDOM < Pacc) Accept = true;

  if(Accept)
  {
    // Check blocking AFTER we think the move is accepted
    size_t& SelectedComponent = SystemComponents.TempVal.component;
    bool isBlocked = false;
    
    if(SelectedComponent < SystemComponents.UseBlockPockets.size() && 
       SystemComponents.UseBlockPockets[SelectedComponent])
    {
      // Check the selected trial position
      size_t SelectedTrial = InsertionVariables.selectedTrial;
      if(SystemComponents.Moleculesize[SelectedComponent] > 1) 
        SelectedTrial = InsertionVariables.selectedTrialOrientation;
      
      // Check all atoms of the molecule
      for(size_t i = 0; i < SystemComponents.Moleculesize[SelectedComponent]; i++)
      {
        double3 atomPos;
        // Get position from appropriate location based on molecule size
        if(SystemComponents.Moleculesize[SelectedComponent] == 1)
        {
          cudaMemcpy(&atomPos, &Sims.New.pos[SelectedTrial], sizeof(double3), cudaMemcpyDeviceToHost);
        }
        else
        {
          if(i == 0)
          {
            cudaMemcpy(&atomPos, &Sims.Old.pos[0], sizeof(double3), cudaMemcpyDeviceToHost);
          }
          else
          {
            size_t chainsize = SystemComponents.Moleculesize[SelectedComponent] - 1;
            size_t atomIndex = SelectedTrial * chainsize + (i - 1);
            cudaMemcpy(&atomPos, &Sims.New.pos[atomIndex], sizeof(double3), cudaMemcpyDeviceToHost);
          }
        }
        
        if(CheckBlockedPosition(SystemComponents, SelectedComponent, atomPos, Sims.Box))
        {
          isBlocked = true;
          break;
        }
      }
    }
    
    if(isBlocked)
    {
      // Reject the move if blocked
      Accept = false;
      energy.zero();
      return;
    }
    
    // If not blocked, proceed with insertion
    AcceptInsertion(Vars, InsertionVariables, systemId, INSERTION);
  }
  else
    energy.zero();
}
```

## Summary

**Current State**:
- ❌ Single Insertion: Blocking checked BEFORE acceptance
- ⚠️ CBMC Insertion: Blocking checked during trial generation (correct for Rosenbluth, but final check needed)
- ❌ Translation/Rotation: No blocking check exists

**Required State**:
- ✅ All moves: Blocking checked AFTER `Accept = true`, then reject if blocked, accept if not blocked

