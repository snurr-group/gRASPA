#include <iostream>
#include <filesystem>
#include <fstream>
#include <vector>
#include <string>
#include <iomanip>

#include <execution>

#include "read_data.h"

__global__ void Initialize_DNN_Positions(Atoms* d_a, Atoms New, Atoms Old, size_t Oldsize, size_t Newsize, size_t SelectedComponent, size_t Location, size_t chainsize, int MoveType, size_t CYCLE)
{
  //Zhao's note: need to think about changing this boolean to switch//
  if(MoveType == TRANSLATION || MoveType == ROTATION || MoveType == SINGLE_INSERTION || MoveType == SINGLE_DELETION) // Translation/Rotation/single_insertion/single_deletion //
  {
    //For Translation/Rotation, the Old positions are already in the Old struct, just need to put the New positions into Old, after the Old positions//
    for(size_t i = Oldsize; i < Oldsize + Newsize; i++) //chainsize here is the total size of the molecule for translation/rotation
    {
      Old.pos[i]           = New.pos[i - Oldsize];
      Old.scale[i]         = New.scale[i - Oldsize];
      Old.charge[i]        = New.charge[i - Oldsize];
      Old.scaleCoul[i]     = New.scaleCoul[i - Oldsize];
    }
  }
  else if(MoveType == INSERTION || MoveType == CBCF_INSERTION) // Insertion & Fractional Insertion //
  {
    //Put the trial orientations in New to Old, right after the first bead position//
    if (chainsize == 0)  //If single atom molecule, first bead position is still in New, move it to old//
    {
      Old.pos[0]       = New.pos[Location];
      Old.scale[0]     = New.scale[Location];
      Old.charge[0]    = New.charge[Location];
      Old.scaleCoul[0] = New.scaleCoul[Location];
    }
    for(size_t i = 0; i < chainsize; i++)
    {
      Old.pos[i + 1]       = New.pos[Location * chainsize + i];
      Old.scale[i + 1]     = New.scale[Location * chainsize + i];
      Old.charge[i + 1]    = New.charge[Location * chainsize + i];
      Old.scaleCoul[i + 1] = New.scaleCoul[Location * chainsize + i];
    }
  }
  else if(MoveType == DELETION || MoveType == CBCF_DELETION) // Deletion //
  {
    for(size_t i = 0; i < Oldsize; i++)
    {
      // For deletion, Location = UpdateLocation, see Deletion Move //
      Old.pos[i]           = d_a[SelectedComponent].pos[Location + i];
      Old.scale[i]         = d_a[SelectedComponent].scale[Location + i];
      Old.charge[i]        = d_a[SelectedComponent].charge[Location + i];
      Old.scaleCoul[i]     = d_a[SelectedComponent].scaleCoul[Location + i];
    }
  }
  if(CYCLE == 145) 
  {
  for(size_t i = 0; i < Oldsize + Newsize; i++)
  printf("Old pos: %.5f %.5f %.5f, scale/charge/scaleCoul: %.5f %.5f %.5f\n", Old.pos[i].x, Old.pos[i].y, Old.pos[i].z, Old.scale[i], Old.charge[i], Old.scaleCoul[i]);
  }
}

void Prepare_DNN_InitialPositions(Atoms*& d_a, Atoms& New, Atoms& Old, Components& SystemComponents, size_t SelectedComponent, int MoveType, size_t Location)
{
  size_t Oldsize = 0; size_t Newsize = 0; size_t chainsize = 0;
  switch(MoveType)
  {
    case TRANSLATION: case ROTATION: // Translation/Rotation Move //
    {
      Oldsize   = SystemComponents.Moleculesize[SelectedComponent];
      Newsize   = SystemComponents.Moleculesize[SelectedComponent];
      chainsize = SystemComponents.Moleculesize[SelectedComponent];
      break;
    }
    case INSERTION: case SINGLE_INSERTION: // Insertion //
    {
      Oldsize   = 0;
      Newsize   = SystemComponents.Moleculesize[SelectedComponent];
      chainsize = SystemComponents.Moleculesize[SelectedComponent] - 1;
      break;
    }
    case DELETION:  case SINGLE_DELETION: // Deletion //
    {
      Oldsize   = SystemComponents.Moleculesize[SelectedComponent];
      Newsize   = 0;
      chainsize = SystemComponents.Moleculesize[SelectedComponent] - 1;
      break;
    }
    case REINSERTION: // Reinsertion //
    {
      throw std::runtime_error("Use the Special Function for Reinsertion");
      //break;
    }
    case IDENTITY_SWAP:
    {
      throw std::runtime_error("Use the Special Function for IDENTITY SWAP!");
    }
    case CBCF_LAMBDACHANGE: // CBCF Lambda Change //
    {
      throw std::runtime_error("Use the Special Function for CBCF Lambda Change");
      //Oldsize   = SystemComponents.Moleculesize[SelectedComponent];
      //Newsize   = SystemComponents.Moleculesize[SelectedComponent];
      //chainsize = SystemComponents.Moleculesize[SelectedComponent] - 1;
      //break;
    }
    case CBCF_INSERTION: // CBCF Lambda Insertion //
    {
      Oldsize      = 0;
      Newsize      = SystemComponents.Moleculesize[SelectedComponent];
      chainsize    = SystemComponents.Moleculesize[SelectedComponent] - 1;
      break;
    }
    case CBCF_DELETION: // CBCF Lambda Deletion //
    {
      Oldsize   = SystemComponents.Moleculesize[SelectedComponent];
      Newsize   = 0;
      chainsize = SystemComponents.Moleculesize[SelectedComponent] - 1;
      break;
    }
  }
  Initialize_DNN_Positions<<<1,1>>>(d_a, New, Old, Oldsize, Newsize, SelectedComponent, Location, chainsize, MoveType, SystemComponents.CURRENTCYCLE);
}

__global__ void Initialize_DNN_Positions_Reinsertion(double3* temp, Atoms* d_a, Atoms Old, size_t Oldsize, size_t Newsize, size_t realpos, size_t SelectedComponent)
{
  for(size_t i = 0; i < Oldsize; i++)
  {
    Old.pos[i]       = d_a[SelectedComponent].pos[realpos + i];
    Old.scale[i]     = d_a[SelectedComponent].scale[realpos + i];
    Old.charge[i]    = d_a[SelectedComponent].charge[realpos + i];
    Old.scaleCoul[i] = d_a[SelectedComponent].scaleCoul[realpos + i];
  }
  //Reinsertion New Positions stored in three arrays, other data are the same as the Old molecule information in d_a//
  for(size_t i = Oldsize; i < Oldsize + Newsize; i++) //chainsize here is the total size of the molecule for translation/rotation
  {
    Old.pos[i]       = temp[i - Oldsize];
    Old.scale[i]     = d_a[SelectedComponent].scale[realpos + i - Oldsize];
    Old.charge[i]    = d_a[SelectedComponent].charge[realpos + i - Oldsize];
    Old.scaleCoul[i] = d_a[SelectedComponent].scaleCoul[realpos + i - Oldsize];
  }
}

void Prepare_DNN_InitialPositions_Reinsertion(Atoms*& d_a, Atoms& Old, double3* temp, Components& SystemComponents, size_t SelectedComponent, size_t Location)
{
  size_t numberOfAtoms = SystemComponents.Moleculesize[SelectedComponent];
  size_t Oldsize = 0; size_t Newsize = numberOfAtoms;
  //Zhao's note: translation/rotation/reinsertion involves new + old states. Insertion/Deletion only has the new state.
  Oldsize         = SystemComponents.Moleculesize[SelectedComponent];
  numberOfAtoms  += Oldsize;
  Initialize_DNN_Positions_Reinsertion<<<1,1>>>(temp, d_a, Old, Oldsize, Newsize, Location, SelectedComponent);
}

__global__ void CalculatePairDistances(Boxsize Box, Atoms* System, Atoms Old, bool* ConsiderThisAdsorbateAtom, size_t Molsize, size_t skip, size_t* indexList, double* Distances, size_t totalThreads, bool UsedForAMove)
{
  //Parallelized over framework atoms//
  //Serialized over adsorbate atoms (only one adsorbate molecule)//
  size_t ij = blockIdx.x * blockDim.x + threadIdx.x;

  if(ij < totalThreads)
  {
    size_t comp = 0;

    const Atoms Component=System[comp];
    //const size_t typeA = Component.Type[ij];
    //const size_t MoleculeID = Component.MolID[ij];
   
    Atoms Adsorbate = Old;
    if(!UsedForAMove) //Then it is for calculating the total energy//
      Adsorbate = System[1];
    for(size_t j = 0; j < Molsize; j++)
    {
      if(!ConsiderThisAdsorbateAtom[j]) continue;
    
      size_t new_j = j + skip; 
      //size_t typeB = Adsorbate.Type[new_j];
      double3 posvec = Component.pos[ij] - Adsorbate.pos[new_j];
      PBC(posvec, Box.Cell, Box.InverseCell, Box.Cubic);
      const double rr_dot = dot(posvec, posvec);
      size_t WriteTo = indexList[ij * Molsize + j];
      
      Distances[WriteTo] = rr_dot;
      //if(ij == 0) printf("ij: %lu, j: %lu, WriteTo: %lu, TypeA(Framework): %lu, TypeB(Adsorbate): %lu, distance: %.5f\n", ij, j, WriteTo, typeA, typeB, rr_dot);
    }
  }
}

static inline size_t MatchInteractionIndex(std::vector<int3>& DNNInteractionList, size_t typeA, size_t typeB)
{
  size_t val=0;
  for(size_t i = 0; i < DNNInteractionList.size(); i++)
  {
    if(typeA == DNNInteractionList[i].x && typeB == DNNInteractionList[i].y)
    {
      val = i;
      break;
    }
  }
  return val;
}

std::vector<std::vector<double>> CalculatePairDistances_CPU(Atoms* System, Boxsize Box, bool* ConsiderThisAdsorbateAtom, std::vector<int3>& DNNInteractionList)
{
  std::vector<std::vector<double>>CPU_Distances(DNNInteractionList.size());
  size_t ads_comp = 1;
  for(size_t i = 0; i < System[0].size; i++)
    for(size_t j = 0; j < System[ads_comp].Molsize; j++)
    {
      if(!ConsiderThisAdsorbateAtom[j]) continue;
      size_t  typeAdsorbate = System[ads_comp].Type[j];
      size_t  typeFramework = System[0].Type[i];
      size_t  InteractionIndex = MatchInteractionIndex(DNNInteractionList, typeAdsorbate, typeFramework);
      double3 posvec = System[0].pos[i] - System[ads_comp].pos[j];
      if(i < 3) printf("adsorbate xyz: %.5f %.5f %.5f\n", System[ads_comp].pos[j].x, System[ads_comp].pos[j].y, System[ads_comp].pos[j].z);
      PBC(posvec, Box.Cell, Box.InverseCell, Box.Cubic);
      const double rr_dot = dot(posvec, posvec);
      CPU_Distances[InteractionIndex].push_back(rr_dot);
    }

  return CPU_Distances;
}

void PrepareOutliersFiles()
{
  std::ofstream textrestartFile{};
  std::string dirname="DNN/";
  std::string Ifname   = dirname + "/" + "Outliers_INSERTION.data";
  std::string Dfname   = dirname + "/" + "Outliers_DELETION.data";
  std::string TRname  = dirname + "/" + "Outliers_SINGLE_PARTICLE.data";
   std::filesystem::path cwd = std::filesystem::current_path();

  std::filesystem::path directoryName = cwd /dirname;
  std::filesystem::path IfileName   = cwd /Ifname;
  std::filesystem::path DfileName   = cwd /Dfname;
  std::filesystem::path TRfileName = cwd /TRname;
  std::filesystem::create_directories(directoryName);

  textrestartFile = std::ofstream(TRfileName, std::ios::out);
  textrestartFile << "THIS FILE IS RECORDING THE DELTA ENERGIES BETWEEN THE NEW AND OLD STATES (New - Old)" << "\n";
  textrestartFile << "x y z Type Move HostGuest_DNN Host_Guest_Classical_MINUS_Host_Guest_Classical" <<"\n";

  textrestartFile = std::ofstream(IfileName, std::ios::out);
  textrestartFile << "THIS FILE IS RECORDING THE ENERGIES RELATED TO EITHER THE NEW/INSERTION" << "\n";
  textrestartFile << "x y z Type Move HostGuest_DNN Host_Guest_Classical_MINUS_Host_Guest_Classical" <<"\n";

  textrestartFile = std::ofstream(DfileName, std::ios::out);
  textrestartFile << "THIS FILE IS RECORDING THE ENERGIES RELATED TO OLD/DELETION (TAKE THE OPPOSITE)" << "\n";
  textrestartFile << "x y z Type Move HostGuest_DNN Host_Guest_Classical_MINUS_Host_Guest_Classical" <<"\n";
}

static inline cppflow::model load_model(std::string& NAME)
{
  setenv("CUDA_VISIBLE_DEVICES", "", 1); //DO NOT use GPU for the tf model//
  cppflow::model DNNModel(NAME);
  return DNNModel;
}

static inline void Prepare_Model(Components& SystemComponents, std::string& NAME)
{
  cppflow::model TempModel = load_model(NAME);
  SystemComponents.DNNModel.push_back(TempModel);
}

void Read_DNN_Model(Components& SystemComponents)
{
  // Try not to let tensorflow occupy the whole GPU//
  // Serialized config options (example of 30% memory fraction)
  // Read more to see how to obtain the serialized options
  
  //Use the model on the CPU//

  size_t ModelID = 0;

  //SystemComponents.DNNModel.emplace_back(std::move(cppflow::model (SystemComponents.ModelName[ModelID])));

  Prepare_Model(SystemComponents, SystemComponents.ModelName[ModelID]);

  /*
  auto second_output = SystemComponents.DNNModel[0]({{SystemComponents.InputLayer[ModelID], real_input}},{"StatefulPartitionedCall:0"});
  //auto second_output = SystemComponents.DNNModel[0](real_input);
  double asdasd = static_cast<double>(second_output[0].get_data<double>()[0]);
  */

  //Read Min Max//
  SystemComponents.DNNMinMax = ReadMinMax();
  printf("MinMax size: %zu\n", SystemComponents.DNNMinMax.size());

  //Prepare for the outlier files//
  PrepareOutliersFiles();

}

double DNN_Evaluate(Components& SystemComponents, std::vector<double>& Feature)
{

  std::vector<double>Vector(SystemComponents.Nfeatures, 0.0);
  for(size_t i = 0; i < Vector.size(); i++) Vector[i] = static_cast<double>(Feature[i]);
  auto real_input = cppflow::tensor(Vector, {1, SystemComponents.Nfeatures});

  size_t ModelID = 0;

  double asd = 0.0;
  auto output = SystemComponents.DNNModel[ModelID]({{SystemComponents.InputLayer[ModelID], real_input}},{"StatefulPartitionedCall:0"});
  asd = static_cast<double>(output[0].get_data<double>()[0]);
  printf("REAL DNN Prediction = %.5f\n", asd);
  return asd;
}

struct Comparedouble3 {
    bool operator()(const double3& a, const double3& b) const
    {   
        return a.z < b.z;
    }   
};

static inline void NormalizeMinMax(std::vector<double>& r_features, std::vector<double2>& MinMax, size_t start)
{

  for(size_t i = 0; i < r_features.size(); i++)
  {
    r_features[i] = (r_features[i] - MinMax[start + i].x) / (MinMax[start + i].y - MinMax[start + i].x);
    //printf("r_feature: %.5f\n", r_features[i]);
  }
}

static inline std::vector<double> Calcuate_Feature_from_Distance(double dist_sq, size_t start, std::vector<double2>& MinMax)
{
  //Add Normalization in this function//
  double dist = sqrt(dist_sq);
  double one_over_dist = 1.0 / dist;
  double one_over_distsq = 1.0 / dist_sq;

  std::vector<double>r_features({std::exp(-dist), one_over_dist, std::pow(one_over_distsq, 2), std::pow(one_over_distsq, 3), std::pow(one_over_distsq, 4), std::pow(one_over_distsq, 5)});
  //printf("FeatureSize: %zu, r_featureSize: %zu, start: %zu\n", Features.size(), r_features.size(), start);
  NormalizeMinMax(r_features, MinMax, start);
  return r_features;
}

std::vector<std::vector<double>> CalculatePairDistances_GPU(Simulations& Sim, Components& SystemComponents, std::vector<double>& Features, int DNN_CalcType, size_t NMol)
{
  size_t NFrameworkAtoms = SystemComponents.Moleculesize[0] * SystemComponents.NumberOfMolecule_for_Component[0];
  size_t Nblock = 0; size_t Nthread = 0; Setup_threadblock(NFrameworkAtoms, &Nblock, &Nthread);
  size_t ads_comp = 1;

  double time = omp_get_wtime();
  switch(DNN_CalcType)
  {
    case TOTAL:
    {  
      CalculatePairDistances<<<Nblock, Nthread>>>(Sim.Box, Sim.d_a, Sim.New, SystemComponents.ConsiderThisAdsorbateAtom, SystemComponents.Moleculesize[ads_comp], NMol * SystemComponents.Moleculesize[ads_comp], SystemComponents.device_InverseIndexList, Sim.Blocksum, NFrameworkAtoms, false);
      break;
    }
    case OLD:
    {
      CalculatePairDistances<<<Nblock, Nthread>>>(Sim.Box, Sim.d_a, Sim.Old, SystemComponents.ConsiderThisAdsorbateAtom, SystemComponents.Moleculesize[ads_comp], 0, SystemComponents.device_InverseIndexList, Sim.Blocksum, NFrameworkAtoms, true);
      break;
    }
    case NEW:
    {
      CalculatePairDistances<<<Nblock, Nthread>>>(Sim.Box, Sim.d_a, Sim.New, SystemComponents.ConsiderThisAdsorbateAtom, SystemComponents.Moleculesize[ads_comp], 0, SystemComponents.device_InverseIndexList, Sim.Blocksum, NFrameworkAtoms, true);
      break;
    }
    case REINSERTION_OLD:
    {
      size_t skip = 0;
      CalculatePairDistances<<<Nblock, Nthread>>>(Sim.Box, Sim.d_a, Sim.Old, SystemComponents.ConsiderThisAdsorbateAtom, SystemComponents.Moleculesize[ads_comp], skip, SystemComponents.device_InverseIndexList, Sim.Blocksum, NFrameworkAtoms, true);
      break;
    }
    case REINSERTION_NEW:
    {
      size_t skip = SystemComponents.Moleculesize[ads_comp];
      CalculatePairDistances<<<Nblock, Nthread>>>(Sim.Box, Sim.d_a, Sim.Old, SystemComponents.ConsiderThisAdsorbateAtom, SystemComponents.Moleculesize[ads_comp], skip, SystemComponents.device_InverseIndexList, Sim.Blocksum, NFrameworkAtoms, true);
      break;
    }
    case DNN_INSERTION: case DNN_DELETION:
    {
      size_t skip = 0;
      CalculatePairDistances<<<Nblock, Nthread>>>(Sim.Box, Sim.d_a, Sim.Old, SystemComponents.ConsiderThisAdsorbateAtom, SystemComponents.Moleculesize[ads_comp], skip, SystemComponents.device_InverseIndexList, Sim.Blocksum, NFrameworkAtoms, true);
      break;
    }
  }
  SystemComponents.DNNGPUTime += omp_get_wtime() - time;

  time = omp_get_wtime();
  //Create 2D vector of distances//
  size_t start = 0; size_t end = 0; size_t top = 9; size_t n_features = 6; double stdsort_time;
  std::vector<std::vector<double>>Distances(SystemComponents.DNNInteractionList.size());
  std::vector<std::vector<double>>SORTED_Dist(SystemComponents.DNNInteractionList.size(), std::vector<double>(top));
  stdsort_time = omp_get_wtime();

  std::vector<size_t> starts; size_t start_sum = 0;
  for(size_t i = 0; i < SystemComponents.DNNInteractionList.size(); i++)
  {
    starts.push_back(start_sum);
    start_sum += SystemComponents.IndexList[i].size();
    if(SystemComponents.CURRENTCYCLE == 10) printf("number of distances [%zu]: %zu\n", i, SystemComponents.IndexList[i].size());
  }


  //#pragma omp parallel for
  for(size_t i = 0; i < SystemComponents.DNNInteractionList.size(); i++)
  {
    Distances[i].resize(SystemComponents.IndexList[i].size()); end += SystemComponents.IndexList[i].size();
    cudaMemcpyAsync(Distances[i].data(), &Sim.Blocksum[starts[i]], SystemComponents.IndexList[i].size()*sizeof(double), cudaMemcpyDeviceToHost);
    
    //Sort the vector//
    std::sort(std::execution::unseq, Distances[i].begin(), Distances[i].end());
  }
  SystemComponents.DNNstdsortTime += omp_get_wtime() - stdsort_time;

  double FeaturizationTime = omp_get_wtime();

  #pragma omp parallel for
  for(size_t ij = 0; ij < SystemComponents.DNNInteractionList.size() * top; ij++)
  {
      size_t i = ij / top;
      size_t j = ij % top;
      SORTED_Dist[i][j] = Distances[i][j];
      //Calculate, Normalize//
      start = (i * top + j) * n_features;
      std::vector<double> r_feature = Calcuate_Feature_from_Distance(SORTED_Dist[i][j], start, SystemComponents.DNNMinMax);
      for(size_t k = 0; k < n_features; k++)
        Features[(i * top + j) * n_features + k] = r_feature[k];
      //Features.insert(std::end(Features), std::begin(r_features), std::end(r_features));
  }

  SystemComponents.DNNFeaturizationTime += omp_get_wtime() - FeaturizationTime;
  SystemComponents.DNNSortTime += omp_get_wtime() - time;
  
  return Distances;
}

void Prepare_FeatureMatrix(Simulations& Sim, Components& SystemComponents, Atoms* HostSystem, Boxsize Host_Box)
{
  //Idea: Write a Kernel to get the distances, write them into pregenerated arrays, then sort using thrust//
  //First Step, count number of pairs for each type of interactions//
  size_t ads_comp = 1; //Zhao's note: Assuming the first component is the adsorbate we are interested in//
  printf("-------------- Preparing DNN Interaction Types --------------\n");
  std::vector<size_t>AdsorbateAtomTypesForDNN;
  for(size_t i = 0; i < HostSystem[ads_comp].Molsize; i++)
    if(SystemComponents.ConsiderThisAdsorbateAtom[i])
      AdsorbateAtomTypesForDNN.push_back(HostSystem[ads_comp].Type[i]);

  std::sort(AdsorbateAtomTypesForDNN.begin(), AdsorbateAtomTypesForDNN.end());
  AdsorbateAtomTypesForDNN.erase(std::unique(AdsorbateAtomTypesForDNN.begin(), AdsorbateAtomTypesForDNN.end()), AdsorbateAtomTypesForDNN.end());
  for(size_t i = 0; i < AdsorbateAtomTypesForDNN.size(); i++) printf("AdsorbateDNN Types %zu\n", AdsorbateAtomTypesForDNN[i]);

  //Hard-coded types for Framework//
  std::vector<size_t> FrameworkAtomTypesForDNN = {0, 1, 2, 3};


  //Ow-Mg (4-0), Ow-O(4-1), Ow-C(4-2), Ow-H(4-3), Hw-Mg(5-0), Hw-O(5-1), Hw-C(5-2), Hw-H(5-3)//
  //DNNInteractionList is a double3 vector, x: adsorbate-type, y: framework-type, z: Number of framework PseudoAtomsfor this type//
  for(size_t i = 0; i < AdsorbateAtomTypesForDNN.size(); i++)
  {
    size_t ads_type = AdsorbateAtomTypesForDNN[i];
    for(size_t j = 0; j < FrameworkAtomTypesForDNN.size(); j++)
    {
      size_t framework_type = FrameworkAtomTypesForDNN[j];
      if(ads_type == framework_type) 
        throw std::runtime_error("Adsorbate Type equals Framework Type for Pseudo Atoms, Weird");

      SystemComponents.DNNInteractionList.push_back({(int)ads_type, (int)framework_type, (int)SystemComponents.NumberOfPseudoAtoms[framework_type]});
      printf("TypeA-B [%zu-%zu], Number: %zu\n", ads_type, framework_type, SystemComponents.NumberOfPseudoAtoms[framework_type]);
    }
  }
  //This determines the number of features for the model// 
  //NTypes: Number of interaction types
  //Ndistances: top N distances for each type of interaction//
  //Ninteractions: for each distance, calculate 6 interaction forms//
  size_t Ntypes = SystemComponents.DNNInteractionList.size(); size_t Ndistances = 9; size_t Ninteractions = 6;
  SystemComponents.Nfeatures = Ntypes * Ndistances * Ninteractions; 
  //You need an array to determine where to store distances for each pair when calculating the distance on the GPU, this is an array that does the trick.//
  //Zhao's note: this list will be created only once at the beginning of the simulation//
  SystemComponents.IndexList.resize(SystemComponents.DNNInteractionList.size());

  size_t count = 0;

  for(size_t i = 0; i < HostSystem[ads_comp].Molsize; i++)
    printf("ConsiderThisATom? %d\n", SystemComponents.ConsiderThisAdsorbateAtom[i]);

  //Loop over framework atoms and the pseudo atoms in the adsorbate, check the Index of the Interaction//
  for(size_t i = 0; i < HostSystem[0].size; i++)
    for(size_t j = 0; j < HostSystem[ads_comp].Molsize; j++)
    {
      count++;
      if(!SystemComponents.ConsiderThisAdsorbateAtom[j]) continue;
      size_t typeAdsorbate = HostSystem[ads_comp].Type[j];
      size_t typeFramework = HostSystem[0].Type[i];
      size_t InteractionIndex = MatchInteractionIndex(SystemComponents.DNNInteractionList, typeAdsorbate, typeFramework);
      if(i < 5)
        printf("count: %zu, TypeA-B: [%zu-%zu], InteractionIndex: %zu\n", i * HostSystem[ads_comp].Molsize + j, typeAdsorbate, typeFramework, InteractionIndex);
      SystemComponents.IndexList[InteractionIndex].push_back(i * HostSystem[ads_comp].Molsize + j);
    }
  //Then, flatten the 2d vector, and reverse the values that you are recording//
  //stored value --> index, index --> stored value//
  std::vector<size_t>FlatIndexList;
  for(size_t i = 0; i < SystemComponents.IndexList.size(); i++)
  {
    printf("Interaction [%zu], Amount [%zu]\n", i, SystemComponents.IndexList[i].size());
    for(size_t j = 0; j < SystemComponents.IndexList[i].size(); j++)
      FlatIndexList.push_back(SystemComponents.IndexList[i][j]);
  }
  size_t Listsize = HostSystem[0].size * HostSystem[ads_comp].Molsize;
  std::vector<size_t>InverseIndexList(Listsize);
  for(size_t i = 0; i < FlatIndexList.size(); i++)
    InverseIndexList[FlatIndexList[i]] = i;
  //Do a test//
  //Now the elements in the flatIndexList contains the "count" we recorded//
  //the count will then be used to match the global pair-interaction ids//
  //Test: Use the value in New as the adsorbate positions//
  //size_t test_count=0;
  for(size_t i = 0; i < HostSystem[0].size; i++)
    for(size_t j = 0; j < HostSystem[ads_comp].Molsize; j++)
    {
      if(!SystemComponents.ConsiderThisAdsorbateAtom[j]) continue;
      if(i < 5)
        printf("test_count: %zu, TypeA-B: [%zu-%zu], Where it is stored: %zu\n", i * HostSystem[ads_comp].Molsize + j, HostSystem[ads_comp].Type[j], HostSystem[0].Type[i], InverseIndexList[i * HostSystem[ads_comp].Molsize + j]);
    }


  //Now the elements in the flatIndexList contains the "count" we recorded//
  //the count will then be used to match the global pair-interaction ids//
  //Test: Use the value in New as the adsorbate positions//
  cudaMalloc(&SystemComponents.device_InverseIndexList, Listsize * sizeof(size_t));
  cudaMemcpy(SystemComponents.device_InverseIndexList, InverseIndexList.data(), Listsize * sizeof(size_t), cudaMemcpyHostToDevice);
  
  //Use Pinned Memory for slightly better mem transfer//
  printf("Listsize: %zu\n", Listsize);
  //cudaMallocHost(&SystemComponents.device_Distances, Listsize * sizeof(double));
  //cudaMalloc(&SystemComponents.device_Distances, Listsize * sizeof(double));
  printf("Size of the device Distance list: %zu\n", Listsize);
}

void WriteOutliers(Components& SystemComponents, Simulations& Sim, int MoveType, MoveEnergy E, double Correction)
{
  //Write to a file for checking//
  std::ofstream textrestartFile{};
  std::string dirname="DNN/";
  std::string TRname  = dirname + "/" + "Outliers_SINGLE_PARTICLE.data";
  std::string Ifname   = dirname + "/" + "Outliers_INSERTION.data";
  std::string Dfname   = dirname + "/" + "Outliers_DELETION.data";

  
  std::filesystem::path cwd = std::filesystem::current_path();

  std::filesystem::path directoryName = cwd /dirname;
  std::filesystem::path IfileName   = cwd /Ifname;
  std::filesystem::path DfileName = cwd /Dfname;
  std::filesystem::path TRfileName = cwd /TRname;
  std::filesystem::create_directories(directoryName);

  size_t size = SystemComponents.HostSystem[1].Molsize;
  size_t ads_comp = 1;
  size_t start= 0;
  std::string Move;
  switch(MoveType)
  {
  case OLD: //Positions are stored in Sim.Old
  {
    SystemComponents.Copy_GPU_Data_To_Temp(Sim.Old, start, size);
    Move = "TRANSLATION_ROTATION_NEW_NON_CBMC_INSERTION";
    textrestartFile = std::ofstream(TRfileName, std::ios::app);
    break;
  }
  case NEW:
  {
    SystemComponents.Copy_GPU_Data_To_Temp(Sim.New, start, size);
    Move = "TRANSLATION_ROTATION_OLD_NON_CBMC_DELETION";
    textrestartFile = std::ofstream(TRfileName, std::ios::app);
    break;
  }
  case REINSERTION_OLD:
  {
    SystemComponents.Copy_GPU_Data_To_Temp(Sim.Old, start, size);
    Move = "REINSERTION_OLD";
    textrestartFile = std::ofstream(TRfileName, std::ios::app);
    break;
  }
  case REINSERTION_NEW:
  {
    start = SystemComponents.Moleculesize[ads_comp];
    SystemComponents.Copy_GPU_Data_To_Temp(Sim.Old, start, size);
    Move = "REINSERTION_NEW";
    textrestartFile = std::ofstream(TRfileName, std::ios::app);
    break;
  }
  case DNN_INSERTION:
  {
    SystemComponents.Copy_GPU_Data_To_Temp(Sim.Old, start, size);
    Move = "SWAP_INSERTION";
    textrestartFile = std::ofstream(IfileName, std::ios::app);
    break;
  }
  case DNN_DELETION:
  {
    SystemComponents.Copy_GPU_Data_To_Temp(Sim.Old, start, size);
    Move = "SWAP_DELETION";
    textrestartFile = std::ofstream(DfileName, std::ios::app);
    break;
  }
  }
  for(size_t i = 0; i < size; i++)
    textrestartFile << SystemComponents.TempSystem.pos[i].x << " " << SystemComponents.TempSystem.pos[i].y << " " << SystemComponents.TempSystem.pos[i].z << " " << SystemComponents.TempSystem.Type[i] << " " << Move << " " << E.DNN_E << " " << Correction << '\n';
}


static inline void WriteDistances(Components& SystemComponents, std::vector<std::vector<double>> Distances)
{
  //Write to a file for checking//
  std::ofstream textrestartFile{};
  std::string dirname="DNN/";
  std::string fname  = dirname + "/" + "Distances_squared.data";
  std::filesystem::path cwd = std::filesystem::current_path();

  std::filesystem::path directoryName = cwd /dirname;
  std::filesystem::path fileName = cwd /fname;
  std::filesystem::create_directories(directoryName);

  textrestartFile = std::ofstream(fileName, std::ios::out);
  textrestartFile << "TypeA(Framework) TypeB(Adsorbate) distance (GPU)" <<"\n";
  for(size_t i = 0; i < SystemComponents.DNNInteractionList.size(); i++)
  {
    //std::sort(CPU_Distances[i].begin(), CPU_Distances[i].end());
    for(size_t j = 0; j < Distances[i].size(); j++)
    {
      //printf("TypeA(Framework): %.5f TypeB(Adsorbate): %.5f distance: %.5f\n", Distances[i][j].x, Distances[i][j].y, Distances[i][j].z);
      textrestartFile << Distances[i][j] << "\n";
    }
  }
}

static inline void WriteFeatures(Components& SystemComponents, std::vector<double> Features)
{
  std::ofstream textrestartFile{};
  std::string dirname="DNN/";
  std::string fname  = dirname + "/" + "Features.data";
  std::filesystem::path cwd = std::filesystem::current_path();

  std::filesystem::path directoryName = cwd /dirname;
  std::filesystem::path fileName = cwd /fname;
  std::filesystem::create_directories(directoryName);

  textrestartFile = std::ofstream(fileName, std::ios::out);
  textrestartFile << "Features" <<"\n";
  for(size_t i = 0; i < Features.size(); i++)
    textrestartFile << Features[i] << "\n";

}

double Predict_From_FeatureMatrix_Move(Simulations& Sim, Components& SystemComponents, int DNN_CalcType)
{
  if(!SystemComponents.UseDNNforHostGuest) return 0.0;
  double time;
  //Get GPU distances//
  time = omp_get_wtime();

  std::vector<double>Features(SystemComponents.Nfeatures); //feature elements//
  std::vector<std::vector<double>> Distances = CalculatePairDistances_GPU(Sim, SystemComponents, Features, DNN_CalcType, 1);
  SystemComponents.DNNFeatureTime += omp_get_wtime() - time; 


  time = omp_get_wtime();
  size_t ModelID = 0;

  //Use the DNN Model to predict//
  auto real_input = cppflow::tensor(Features, {1, SystemComponents.Nfeatures});
  auto output = SystemComponents.DNNModel[ModelID]({{SystemComponents.InputLayer[ModelID], real_input}},{"StatefulPartitionedCall:0"});
  double prediction = static_cast<double>(output[0].get_data<double>()[0]);

  SystemComponents.DNNPredictTime += omp_get_wtime() - time;
  //printf("DNN Prediction: %.5f [kJ/mol], %.5f [internal unit]\n", prediction, prediction * 100.0);
  return prediction * 100.0;
  //Check CPU distances//
  //std::vector<std::vector<double>> CPU_Distances = CalculatePairDistances_CPU(HostSystem, Host_Box, SystemComponents.ConsiderThisAdsorbateAtom, SystemComponents.DNNInteractionList);

  
  //Write to a file for checking//
  //WriteFeatureMatrix(SystemComponents, Distances);
}

double Predict_From_FeatureMatrix_Total(Simulations& Sim, Components& SystemComponents)
{
  if(!SystemComponents.UseDNNforHostGuest) return 0.0;
  size_t ads_comp = 1;
  size_t NMol = SystemComponents.NumberOfMolecule_for_Component[ads_comp];
  //Get GPU distances//
  std::vector<double>AllFeatures;
  for(size_t iMol = 0; iMol < NMol; iMol++)
  {
    std::vector<double>Features(SystemComponents.Nfeatures); //feature elements//
    std::vector<std::vector<double>> Distances = CalculatePairDistances_GPU(Sim, SystemComponents, Features, TOTAL, iMol);
    AllFeatures.insert(std::end(AllFeatures), std::begin(Features), std::end(Features));
    if(iMol == 0)
    {
      WriteDistances(SystemComponents, Distances);
      WriteFeatures(SystemComponents, Features);
    }
  }
  size_t ModelID = 0;
 
  //Use the DNN Model to predict//
  auto real_input = cppflow::tensor(AllFeatures, {NMol, SystemComponents.Nfeatures});
  auto output = SystemComponents.DNNModel[ModelID]({{SystemComponents.InputLayer[ModelID], real_input}},{"StatefulPartitionedCall:0"});

  double tot = 0.0;
  for(size_t i = 0; i < NMol; i++)
  {
    double prediction = static_cast<double>(output[0].get_data<double>()[i]);
    tot += prediction;
    //printf("Molecule %zu, DNN Prediction: %.5f [kJ/mol], %.5f [internal unit]\n", i, prediction, prediction * 100.0);
  }
  return tot * 100.0;
}
