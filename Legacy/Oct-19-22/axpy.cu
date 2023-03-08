#include "axpy.h"
#include "print_statistics.h"
#include "mc_translation.h"
#include "mc_insertion_deletion.h"
#include <numeric>
#include <cmath>
#include <algorithm>
#include <filesystem>

// First Randomly Select a Component, then Randomly Determine the Molecule //
static inline void SelectMoleculeFromComponent(Components& SystemComponents, size_t SelectedComponent, size_t *SelectedMolInComponent)
{
  size_t TempMolecule = (size_t) (get_random_from_zero_to_one()*SystemComponents.NumberOfMolecule_for_Component[SelectedComponent]);
  *SelectedMolInComponent = TempMolecule;
}

std::vector<double> Multiple_Simulations(int Cycles, std::vector<Components>& SystemComponents, Boxsize Box, Simulations* Sims, Temp_Atoms* TempAtoms, ForceField FF, double* y, double* dUdlambda, RandomNumber Random, std::vector<WidomStruct>& Widom, Units Constants, std::vector<double> System_Energies, bool DualPrecision)
{
  double tot = 0.0;

  double running_energy = 0.0;

  size_t WidomCount = 0;

  bool DEBUG = false;
  printf("Simulations Begin\n");

  size_t NumberOfSimulation = System_Energies.size();

  std::vector<size_t> SelectedMolecules(NumberOfSimulation);
  
  Components TempComponent = SystemComponents[0]; //Select the first Simulation as the reference//

  bool SerialSimulations = true;  //Run these simulations in serial or in parallel//

  std::vector<double> Running_Energies(NumberOfSimulation, 0.0);

  for(size_t i = 0; i < Cycles; i++)
  {
    size_t comp = (size_t) get_random_from_zero_to_one()*(TempComponent.Total_Components-TempComponent.NumberOfFrameworks)+1; //skip framework
    for(size_t sim = 0; sim < NumberOfSimulation; sim++)
      SelectMoleculeFromComponent(SystemComponents[sim], comp, &SelectedMolecules[sim]);
    double RANDOMNUMBER = get_random_from_zero_to_one();
    if(SerialSimulations)
    {
      for(size_t j=0; j < NumberOfSimulation; j++)
      {
        Atoms* xxx = Sims[j].d_a;
        if(RANDOMNUMBER < TempComponent.Moves[comp].WidomProb)
        {
          WidomCount ++;
          size_t SelectedTrial=0; bool SuccessConstruction = false; double energy = 0.0; double StoredR = 0.0;
          double Rosenbluth=Widom_Move_FirstBead(Box, SystemComponents[j], xxx, TempAtoms[j].NewMol, FF, Random, Widom[j], 0, comp, true, false, false, StoredR, &SelectedTrial, &SuccessConstruction, &energy, false); //first false: Reinsertion? second false: Retrace? third false is for using Dual-Precision. For Widom Insertion, don't use it.//
          if(SystemComponents[j].Moleculesize[comp] > 1 && Rosenbluth > 1e-150)
          {
            size_t SelectedFirstBeadTrial = SelectedTrial; double temp_energy = energy;
            Rosenbluth*=Widom_Move_Chain(Box, SystemComponents[j], xxx, TempAtoms[j].Mol, TempAtoms[j].NewMol, FF, Random, Widom[j], 0, comp, true, false, &SelectedTrial, &SuccessConstruction, &energy, SelectedFirstBeadTrial, false); //false is for using Dual-Precision. For Widom Insertion, don't use it.//
          }
          size_t BlockIDX = i/(Cycles/SystemComponents[j].Moves[comp].NumberOfBlocks);
          Widom[j].Rosenbluth[BlockIDX]+= Rosenbluth;
          Widom[j].RosenbluthSquared[BlockIDX]+= Rosenbluth*Rosenbluth;
          Widom[j].RosenbluthCount[BlockIDX]++;
        }
      }
    }
  }
  for(size_t j=0; j < NumberOfSimulation; j++)
  {
    for(size_t comp = TempComponent.NumberOfFrameworks; comp < TempComponent.Total_Components; comp++)
    {
      Print_Translation_Statistics(SystemComponents[j].Moves[comp], FF);
      Print_Widom_Statistics(Widom[j], SystemComponents[j].Moves[comp], SystemComponents[j].Beta, Constants.energy_to_kelvin);
      Print_Swap_Statistics(SystemComponents[j].Moves[comp]);
    }
  }
  return Running_Energies;
}

double cuSoA(int Cycles, Components& SystemComponents, Boxsize Box, Atoms* d_a, Atoms Mol, Atoms NewMol, ForceField FF, double* y, double* dUdlambda, RandomNumber Random, WidomStruct Widom, Units Constants, double init_energy, bool DualPrecision)
{
  
  double tot = 0.0;

  double running_energy = 0.0;

  size_t WidomCount = 0;

  bool DEBUG = false;
  printf("Begin: There are %zu Molecules, %zu Frameworks\n",SystemComponents.TotalNumberOfMolecules, SystemComponents.NumberOfFrameworks);

  for(size_t i = 0; i < Cycles; i++)
  {
    size_t comp = (size_t) get_random_from_zero_to_one()*(SystemComponents.Total_Components-SystemComponents.NumberOfFrameworks)+1; //skip framework
    size_t SelectedMolInComponent = 0; 
    SelectMoleculeFromComponent(SystemComponents, comp, &SelectedMolInComponent);

    /*if(SystemComponents.NumberOfMolecule_for_Component[comp] == 0){ //no molecule in the system for this species
      running_energy += Insertion(Box, SystemComponents, d_a, Mol, NewMol, FF, Random, Widom, SelectedMolInComponent, comp, DualPrecision);
      continue;
    }
    */
    double RANDOMNUMBER = get_random_from_zero_to_one();
    if(RANDOMNUMBER < SystemComponents.Moves[comp].TranslationProb)
    {
      // PERFORM TRANSLATION MOVE //
      running_energy += Translation_Move(Box, SystemComponents, d_a, Mol, NewMol, FF, y, dUdlambda, Random, SelectedMolInComponent, comp);
      if(DEBUG){printf("After Translation: running energy: %.10f\n", running_energy);}
    }
    else if(RANDOMNUMBER < SystemComponents.Moves[comp].RotationProb) 
    {
      // PERFORM ROTATION MOVE //
      running_energy += Rotation_Move(Box, SystemComponents, d_a, Mol, NewMol, FF, y, dUdlambda, Random, SelectedMolInComponent, comp);
      if(DEBUG){printf("After Rotation: running energy: %.10f\n", running_energy);}
    }
    else if(RANDOMNUMBER < SystemComponents.Moves[comp].WidomProb)
    {
      WidomCount ++;
      size_t SelectedTrial=0; bool SuccessConstruction = false; double energy = 0.0; double StoredR = 0.0;
      double Rosenbluth=Widom_Move_FirstBead(Box, SystemComponents, d_a, NewMol, FF, Random, Widom, SelectedMolInComponent, comp, true, false, false, StoredR, &SelectedTrial, &SuccessConstruction, &energy, false); //first false: Reinsertion? second false: Retrace? third false is for using Dual-Precision. For Widom Insertion, don't use it.//
      if(SystemComponents.Moleculesize[comp] > 1 && Rosenbluth > 1e-150)
      {
        size_t SelectedFirstBeadTrial = SelectedTrial; double temp_energy = energy;
        Rosenbluth*=Widom_Move_Chain(Box, SystemComponents, d_a, Mol, NewMol, FF, Random, Widom, SelectedMolInComponent, comp, true, false, &SelectedTrial, &SuccessConstruction, &energy, SelectedFirstBeadTrial, false); //false is for using Dual-Precision. For Widom Insertion, don't use it.//
      }
      size_t BlockIDX = i/(Cycles/SystemComponents.Moves[comp].NumberOfBlocks);
      Widom.Rosenbluth[BlockIDX]+= Rosenbluth;
      Widom.RosenbluthSquared[BlockIDX]+= Rosenbluth*Rosenbluth;
      Widom.RosenbluthCount[BlockIDX]++;
    }
    else if(RANDOMNUMBER < SystemComponents.Moves[comp].ReinsertionProb)
    {
      if(DEBUG) printf("Before Reinsertion, energy: %.10f\n", running_energy);
      running_energy += Reinsertion(Box, SystemComponents, d_a, Mol, NewMol, FF, Random, Widom, SelectedMolInComponent, comp, DualPrecision);
    }
    else
    {
      // DO GCMC INSERTION //
      if(get_random_from_zero_to_one() < 0.5){
        running_energy += Insertion(Box, SystemComponents, d_a, Mol, NewMol, FF, Random, Widom, SelectedMolInComponent, comp, DualPrecision);}
      else{
        // DO GCMC DELETION //
        if(DEBUG){printf("Cycle: %zu, DOING DELETION\n", i);}
        running_energy += Deletion(Box, SystemComponents, d_a, Mol, NewMol, FF, Random, Widom, SelectedMolInComponent, comp, DualPrecision);}
    }
    if(i%500==0 &&(SystemComponents.Moves[comp].TranslationTotal > 0))
    {
      printf("i: %zu\n", i);
      Update_Max_Translation(FF, SystemComponents.Moves[comp]);
    }
    if(DEBUG)
    {
      printf("After %zu MOVE: Sum energies\n", i);
      double* xxx; xxx = (double*) malloc(sizeof(double)*2);
      double* device_xxx = CUDA_copy_allocate_double_array(xxx, 2);
      one_thread_GPU_test<<<1,1>>>(Box, d_a, FF, device_xxx); cudaMemcpy(xxx, device_xxx, sizeof(double), cudaMemcpyDeviceToHost);
      printf("Current Total Energy (1 thread GPU): %.10f, running total: %.10f\n", xxx[0], init_energy+running_energy);
      cudaDeviceSynchronize();
      if(abs(xxx[0] - (init_energy+running_energy)) > 0.1) //means that there is an energy drift
      {
        printf("THere is an energy drift at cycle %zu\n", i);
      }
      cudaFree(device_xxx);
    }
  }
  //print statistics
  for(size_t comp = SystemComponents.NumberOfFrameworks; comp < SystemComponents.Total_Components; comp++)
  {
    Print_Translation_Statistics(SystemComponents.Moves[comp], FF);
    Print_Widom_Statistics(Widom, SystemComponents.Moves[comp], SystemComponents.Beta, Constants.energy_to_kelvin);
    Print_Swap_Statistics(SystemComponents.Moves[comp]);
    printf("total-deltaU: %.10f\n", running_energy);
  }
  return running_energy;
}
