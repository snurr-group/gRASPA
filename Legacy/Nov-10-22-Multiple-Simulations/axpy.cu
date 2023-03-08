#include "axpy.h"
#include "print_statistics.h"
#include "mc_translation.h"
#include "mc_insertion_deletion.h"
#include <numeric>
#include <cmath>
#include <algorithm>
#include <filesystem>
double Run_Simulation(int Cycles, Components& SystemComponents, Boxsize Box, Simulations Sims, ForceField FF, RandomNumber Random, WidomStruct Widom, Units Constants, double init_energy, bool DualPrecision, std::vector<size_t>& NumberOfCreateMolecules, bool CreateMolecules)
{
  double running_energy = 0.0;

  size_t WidomCount = 0;

  bool DEBUG = false;
  size_t transCount=0;

   // Create Molecules in the Box Before the Simulation //
  if(CreateMolecules)
  {
    size_t CreateFailCount = 0; size_t Created = 0; size_t SelectedMol = 0;
    for(size_t comp = 1; comp < SystemComponents.Total_Components; comp++)
    {
      CreateFailCount = 0;
      while(NumberOfCreateMolecules[comp] > 0)
      {
        printf("Creating %zu Molecule for Component %zu; There are %zu Molecules of that component in the System\n", Created, comp, SystemComponents.NumberOfMolecule_for_Component[comp]);
        SelectedMol = Created; if(Created > 0) SelectedMol = Created - 1; //Zhao's note: this is a little confusing, but when number of molecule for that species = 0 or 1, the chosen molecule is zero. This is creating from zero loading, need to change in the future, when we read from restart file//
        size_t OldVal = SystemComponents.NumberOfMolecule_for_Component[comp];
        running_energy += CreateMolecule(Box, SystemComponents, Sims, FF, Random, Widom, SelectedMol, comp, DualPrecision);
        if(SystemComponents.NumberOfMolecule_for_Component[comp] == OldVal)
        {CreateFailCount ++;} else {NumberOfCreateMolecules[comp] --; Created ++;}
        if(CreateFailCount > 10) throw std::runtime_error("Bad Insertions When Creating Molecules!");
      }
    }
    return running_energy;
  }
  

  printf("There are %zu Molecules, %zu Frameworks\n",SystemComponents.TotalNumberOfMolecules, SystemComponents.NumberOfFrameworks);

  for(size_t i = 0; i < Cycles; i++)
  {
    //Randomly Select an Adsorbate Molecule and determine its Component: MoleculeID --> Component
    //if((SystemComponents.TotalNumberOfMolecules - SystemComponents.NumberOfFrameworks) == 0)
    //  continue;
    size_t SelectedMolecule = (size_t) (get_random_from_zero_to_one()*(SystemComponents.TotalNumberOfMolecules-SystemComponents.NumberOfFrameworks));
    size_t comp = SystemComponents.NumberOfFrameworks; // When selecting components, skip the component 0 (because it is the framework)
    size_t SelectedMolInComponent = SelectedMolecule; size_t totalsize= 0;
    for(size_t ijk = SystemComponents.NumberOfFrameworks; ijk < SystemComponents.Total_Components; ijk++) //Assuming Framework atoms are the top in the Atoms array
    {
      if(SelectedMolInComponent == 0) break;
      totalsize += SystemComponents.NumberOfMolecule_for_Component[ijk];
      if(SelectedMolInComponent >= totalsize)
      {
        comp++;
        SelectedMolInComponent -= SystemComponents.NumberOfMolecule_for_Component[ijk];
      }
    }

    //printf("Selected Comp: %zu, SelectedMol: %zu, Num: %zu\n", comp, SelectedMolInComponent, SystemComponents.NumberOfMolecule_for_Component[comp]);
    if(SystemComponents.NumberOfMolecule_for_Component[comp] == 0){ //no molecule in the system for this species
      //printf("Doing insertion since there is no molecule for this species; SelectedMol: %zu, comp: %zu\n", SelectedMolInComponent, comp);
      running_energy += Insertion(Box, SystemComponents, Sims, FF, Random, Widom, SelectedMolInComponent, comp, DualPrecision);
      continue;
    }

    double RANDOMNUMBER = get_random_from_zero_to_one();
    if(RANDOMNUMBER < SystemComponents.Moves[comp].TranslationProb)
    {
      transCount++;
      //PERFORM TRANSLATION MOVE//
      running_energy += Translation_Move(Box, SystemComponents, Sims, Widom, FF, Random, SelectedMolInComponent, comp);
      if(DEBUG){printf("After Translation: running energy: %.10f\n", running_energy);}
    }
    else if(RANDOMNUMBER < SystemComponents.Moves[comp].RotationProb) //Rotation
    {
      //PERFORM ROTATION MOVE, a test//
      running_energy += Rotation_Move(Box, SystemComponents, Sims, Widom, FF, Random, SelectedMolInComponent, comp);
      //printf("After Translation: running energy: %.10f\n", running_energy);
    }
    else if(RANDOMNUMBER < SystemComponents.Moves[comp].WidomProb)
    {
      WidomCount ++;
      //printf("Performing Widom\n");
      size_t SelectedTrial=0; bool SuccessConstruction = false; double energy = 0.0; double StoredR = 0.0;
      double Rosenbluth=Widom_Move_FirstBead_PARTIAL(Box, SystemComponents, Sims, FF, Random, Widom, SelectedMolInComponent, comp, true, false, false, StoredR, &SelectedTrial, &SuccessConstruction, &energy, false);
      if(SystemComponents.Moleculesize[comp] > 1 && Rosenbluth > 1e-150)
      {
        size_t SelectedFirstBeadTrial = SelectedTrial; 
        Rosenbluth*=Widom_Move_Chain_PARTIAL(Box, SystemComponents, Sims, FF, Random, Widom, SelectedMolInComponent, comp, true, false, &SelectedTrial, &SuccessConstruction, &energy, SelectedFirstBeadTrial, false); //false is for using Dual-Precision. For Widom Insertion, don't use it.//
      }
      //Assume 5 blocks
      size_t BlockIDX = i/(Cycles/SystemComponents.Moves[comp].NumberOfBlocks); //printf("BlockIDX=%zu\n", BlockIDX);
      Widom.Rosenbluth[BlockIDX]+= Rosenbluth;
      Widom.RosenbluthSquared[BlockIDX]+= Rosenbluth*Rosenbluth;
      Widom.RosenbluthCount[BlockIDX]++;
    }
    else if(RANDOMNUMBER < SystemComponents.Moves[comp].ReinsertionProb)
    {
      if(DEBUG) printf("Before Reinsertion, energy: %.10f\n", running_energy);
      running_energy += Reinsertion(Box, SystemComponents, Sims, FF, Random, Widom, SelectedMolInComponent, comp, DualPrecision);
    }
    else
    {
      // DO GCMC INSERTION //
      if(get_random_from_zero_to_one() < 0.5){ 
        //printf("Doing insertion SelectedMol: %zu, comp: %zu\n", SelectedMolInComponent, comp);
        running_energy += Insertion(Box, SystemComponents, Sims, FF, Random, Widom, SelectedMolInComponent, comp, DualPrecision);}
      else{
        if(DEBUG)printf("Cycle: %zu, DOING DELETION\n", i);
        running_energy += Deletion(Box, SystemComponents, Sims, FF, Random, Widom, SelectedMolInComponent, comp, DualPrecision);}
    }
    if(i%500==0 &&(SystemComponents.Moves[comp].TranslationTotal > 0))
    {
      printf("i: %zu\n", i);
      Update_Max_Translation(SystemComponents.Moves[comp], Sims);
    }
  }
  //print statistics
  for(size_t comp = SystemComponents.NumberOfFrameworks; comp < SystemComponents.Total_Components; comp++)
  {
    Print_Translation_Statistics(SystemComponents.Moves[comp], Sims.MaxTranslation);
    Print_Rotation_Statistics(SystemComponents.Moves[comp], Sims.MaxRotation);
    Print_Widom_Statistics(Widom, SystemComponents.Moves[comp], SystemComponents.Beta, Constants.energy_to_kelvin);
    Print_Swap_Statistics(Widom, SystemComponents.Moves[comp]);
    //printf("TransCount: %zu\n", transCount);
    printf("running total: %.10f\n", running_energy + init_energy);
  }
  return running_energy;
}
