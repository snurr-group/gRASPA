#include <iostream>
#include <filesystem>
#include <fstream>
#include <vector>
#include <string>
#include <iomanip>

#include "read_data.h"

__global__ void CalculatePairDistances(Boxsize Box, Atoms* System, Atoms Old, bool* ConsiderThisAdsorbateAtom, size_t Molsize, size_t skip, size_t* indexList, double* Distances, size_t totalThreads, bool UsedForAMove)
{
  //Parallelized over framework atoms//
  //Serialized over adsorbate atoms (only one adsorbate molecule)//
  size_t ij = blockIdx.x * blockDim.x + threadIdx.x;

  if(ij < totalThreads)
  {
    size_t comp = 0;

    const Atoms Component=System[comp];
    const size_t typeA = Component.Type[ij];
    const size_t MoleculeID = Component.MolID[ij];
   
    Atoms Adsorbate = Old;
    if(!UsedForAMove) //Then it is for calculating the total energy//
      Adsorbate = System[1];
    for(size_t j = 0; j < Molsize; j++)
    {
      if(!ConsiderThisAdsorbateAtom[j]) continue;
    
      size_t new_j = j + skip; 
      size_t typeB = Adsorbate.Type[new_j];
      double posvec[3] = {Component.x[ij] - Adsorbate.x[new_j], Component.y[ij] - Adsorbate.y[new_j], Component.z[ij] - Adsorbate.z[new_j]};
      PBC(posvec, Box.Cell, Box.InverseCell, Box.Cubic);
      const double rr_dot = posvec[0]*posvec[0] + posvec[1]*posvec[1] + posvec[2]*posvec[2];
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
      size_t typeAdsorbate = System[ads_comp].Type[j];
      size_t typeFramework = System[0].Type[i];
      size_t InteractionIndex = MatchInteractionIndex(DNNInteractionList, typeAdsorbate, typeFramework);
      double posvec[3] = {System[0].x[i] - System[ads_comp].x[j], System[0].y[i] - System[ads_comp].y[j], System[0].z[i] - System[ads_comp].z[j]};
      if(i < 3) printf("adsorbate xyz: %.5f %.5f %.5f\n", System[ads_comp].x[j], System[ads_comp].y[j], System[ads_comp].z[j]);
      PBC_CPU(posvec, Box.Cell, Box.InverseCell, Box.Cubic);
      const double rr_dot = posvec[0]*posvec[0] + posvec[1]*posvec[1] + posvec[2]*posvec[2];
      CPU_Distances[InteractionIndex].push_back(rr_dot);
    }

  return CPU_Distances;
}

void Read_DNN_Model(Components& SystemComponents)
{
  // Try not to let tensorflow occupy the whole GPU//
  // Serialized config options (example of 30% memory fraction)
  // Read more to see how to obtain the serialized options
  
  //std::vector<uint8_t> config{0x32,0xb,0x9,0x34,0x33,0x33,0x33,0x33,0x33,0xd3,0x3f,0x20,0x1};
  //std::vector<uint8_t> config{0x32,0x9,0x9,0x34,0x33,0x33,0x33,0x33,0x33,0xd3,0x3f}; //No memory growth//
  std::vector<uint8_t> config{0x32,0x9,0x9,0x9a,0x99,0x99,0x99,0x99,0x99,0xb9,0x3f}; //10%, No mem growth//
  // Create new options with your configuration
  TFE_ContextOptions* options = TFE_NewContextOptions();
  TFE_ContextOptionsSetConfig(options, config.data(), config.size(), cppflow::context::get_status());
  // Replace the global context with your options
  cppflow::get_global_context() = cppflow::context(options);
  
  ReadDNNModelNames(SystemComponents);

  size_t ModelID = 0;

  cppflow::model DNNModel(SystemComponents.ModelName[ModelID]); //= std::make_optional(model);
  std::vector<double>Vector(432, 1.0);
  auto real_input = cppflow::tensor(Vector, {1, 432});
  //auto output = DNNModel({{"serving_default_dense_input:0", real_input}},{"StatefulPartitionedCall:0"});
  auto output = DNNModel({{SystemComponents.InputLayer[ModelID], real_input}},{"StatefulPartitionedCall:0"});


  double asd = static_cast<double>(output[0].get_data<double>()[0]);
  SystemComponents.DNNModel.push_back(DNNModel);
  auto second_output = SystemComponents.DNNModel[0]({{SystemComponents.InputLayer[ModelID], real_input}},{"StatefulPartitionedCall:0"});
  double asdasd = static_cast<double>(second_output[0].get_data<double>()[0]);

  //Read Min Max//
  SystemComponents.DNNMinMax = ReadMinMax();
  printf("MinMax size: %zu\n", SystemComponents.DNNMinMax.size());
  printf("Prediction = %.5f; %.5f\n", asd, asdasd);
}

double DNN_Evaluate(Components& SystemComponents, std::vector<double>& Feature)
{

  std::vector<double>Vector(432, 0.0);
  for(size_t i = 0; i < Vector.size(); i++) Vector[i] = static_cast<double>(Feature[i]);
  auto real_input = cppflow::tensor(Vector, {1, 432});

  size_t ModelID = 0;

  double asd = 0.0;
  auto output = SystemComponents.DNNModel[ModelID]({{SystemComponents.InputLayer[ModelID], real_input}},{"StatefulPartitionedCall:0"});
  asd = static_cast<double>(output[0].get_data<double>()[0]);
  printf("REAL DNN Prediction = %.5f\n", asd);
  return asd;
}
static inline void Setup_threadblock_DNN(size_t arraysize, size_t *Nblock, size_t *Nthread)
{
  size_t value = arraysize;
  if(value >= 128) value = 128;
  double ratio = (double)arraysize/value;
  size_t blockValue = ceil(ratio);
  if(blockValue == 0) blockValue++;
  *Nthread = value;
  *Nblock = blockValue;
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

static inline void Calcuate_Feature_from_Distance(double dist_sq, std::vector<double>& Features, std::vector<double2>& MinMax)
{
  //Add Normalization in this function//
  double dist = sqrt(dist_sq);
  double one_over_dist = 1.0 / dist;
  double one_over_distsq = 1.0 / dist_sq;
  /*
  Features.push_back(std::exp(-dist));              //e^-r
  Features.push_back(one_over_dist);                //r^-1
  Features.push_back(std::pow(one_over_distsq, 2)); //r^-4
  Features.push_back(std::pow(one_over_distsq, 3)); //r^-6
  Features.push_back(std::pow(one_over_distsq, 4)); //r^-8
  Features.push_back(std::pow(one_over_distsq, 5)); //r^-10
  */
  //std::vector<float>r_features({static_cast<float>(std::exp(-dist)), static_cast<float>(one_over_dist), static_cast<float>(std::pow(one_over_distsq, 2)), static_cast<float>(std::pow(one_over_distsq, 3)), static_cast<float>(std::pow(one_over_distsq, 4)), static_cast<float>(std::pow(one_over_distsq, 5))});
  std::vector<double>r_features({std::exp(-dist), one_over_dist, std::pow(one_over_distsq, 2), std::pow(one_over_distsq, 3), std::pow(one_over_distsq, 4), std::pow(one_over_distsq, 5)});
  size_t start = Features.size();
  //printf("FeatureSize: %zu, r_featureSize: %zu, start: %zu\n", Features.size(), r_features.size(), start);
  NormalizeMinMax(r_features, MinMax, start);
  Features.insert(std::end(Features), std::begin(r_features), std::end(r_features));
}

std::vector<std::vector<double>> CalculatePairDistances_GPU(Simulations& Sim, Components& SystemComponents, std::vector<double>& Features, int DNN_CalcType, size_t NMol)
{
  size_t NFrameworkAtoms = SystemComponents.Moleculesize[0] * SystemComponents.NumberOfMolecule_for_Component[0];
  size_t Nblock = 0; size_t Nthread = 0; Setup_threadblock_DNN(NFrameworkAtoms, &Nblock, &Nthread);
  size_t ads_comp = 1;
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
  //Create 2D vector of distances//
  std::vector<std::vector<double>>Distances(SystemComponents.DNNInteractionList.size());
  std::vector<std::vector<double>>SORTED_Dist(SystemComponents.DNNInteractionList.size());
  size_t start = 0; size_t end = 0; size_t top = 9;
  for(size_t i = 0; i < SystemComponents.DNNInteractionList.size(); i++)
  {
    Distances[i].resize(SystemComponents.IndexList[i].size()); end += SystemComponents.IndexList[i].size();
    cudaMemcpy(Distances[i].data(), &Sim.Blocksum[start], SystemComponents.IndexList[i].size()*sizeof(double), cudaMemcpyDeviceToHost);
    //Sort the vector//
    std::sort(Distances[i].begin(), Distances[i].end());
    SORTED_Dist[i].resize(top);
    for(size_t j = 0; j < top; j++)
    {
      SORTED_Dist[i][j] = Distances[i][j];
      //Calculate, Normalize//
      Calcuate_Feature_from_Distance(SORTED_Dist[i][j], Features, SystemComponents.DNNMinMax);
    }
    start += SystemComponents.IndexList[i].size();
  }
  return Distances;
}

double Prepare_FeatureMatrix(Simulations& Sim, Components& SystemComponents, Atoms* HostSystem, Boxsize Host_Box)
{
  //Idea: Write a Kernel to get the distances, write them into pregenerated arrays, then sort using thrust//
  //First Step, count number of pairs for each type of interactions//
  //Hard-coded types//
  //Ow-Mg (4-0), Ow-O(4-1), Ow-C(4-2), Ow-H(4-3), Hw-Mg(5-0), Hw-O(5-1), Hw-C(5-2), Hw-H(5-3)//
  for(size_t a=4; a < 6; a++) //Adsorbate Types//
    for(size_t b=0; b < 4; b++) //Framework Types//
    {
      SystemComponents.DNNInteractionList.push_back({(int)a, (int)b, (int)SystemComponents.NumberOfPseudoAtoms[b]});
      printf("TypeA-B [%zu-%zu], Number: %zu\n", a,b,SystemComponents.NumberOfPseudoAtoms[b]);
    }
  //You need an array to determine where to store distances for each pair when calculating the distance on the GPU, this is an array that does the trick.//
  //Zhao's note: this list will be created only once at the beginning of the simulation//
  SystemComponents.IndexList.resize(SystemComponents.DNNInteractionList.size());
  size_t ads_comp = 1; //Zhao's note: Assuming the first component is the adsorbate we are interested in//
  size_t count = 0;
  //Zhao's note: tip4p has 3 atoms considered for the feature matrix, ignore the Lw one//
  cudaMallocManaged(&SystemComponents.ConsiderThisAdsorbateAtom, HostSystem[ads_comp].Molsize * sizeof(bool));
  SystemComponents.ConsiderThisAdsorbateAtom[0] = true;
  SystemComponents.ConsiderThisAdsorbateAtom[1] = false;
  SystemComponents.ConsiderThisAdsorbateAtom[2] = true;
  SystemComponents.ConsiderThisAdsorbateAtom[3] = true;

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
  size_t test_count=0;
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
  //Get GPU distances//
  std::vector<double>Features; //feature, 1x432 elements, single precision//
  std::vector<std::vector<double>> Distances = CalculatePairDistances_GPU(Sim, SystemComponents, Features, DNN_CalcType, 1);

  size_t ModelID = 0;

  //Use the DNN Model to predict//
  auto real_input = cppflow::tensor(Features, {1, 432});
  auto output = SystemComponents.DNNModel[ModelID]({{SystemComponents.InputLayer[ModelID], real_input}},{"StatefulPartitionedCall:0"});
  double prediction = static_cast<double>(output[0].get_data<double>()[0]);
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
    std::vector<double>Features; //feature, 1x432 elements, single precision//
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
  auto real_input = cppflow::tensor(AllFeatures, {NMol, 432});
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
