#include <complex>
#include "VDW_Coulomb.cuh"
#include "maths.cuh"
#include "Ewald_Energy_Functions.h"
#include "TailCorrection_Energy_Functions.h"
#include <cuda_fp16.h>
#include <omp.h>

#include "DNN_HostGuest_Energy_Functions.h"

inline void checkCUDAErrorVDW(const char *msg)
{
    cudaError_t err = cudaGetLastError();
    if( cudaSuccess != err)
    {
        printf("CUDA Error: %s: %s.\n", msg, cudaGetErrorString(err) );
        exit(EXIT_FAILURE);
    }
}

//Zhao's note: There were a few variants of the same Setup_threadblock function, some of them are slightly different//
//This might be a point where debugging is needed//
void Setup_threadblock(size_t arraysize, size_t *Nblock, size_t *Nthread)
{
  if(arraysize == 0)  return;
  size_t value = arraysize;
  if(value >= DEFAULTTHREAD) value = DEFAULTTHREAD;
  double ratio = (double)arraysize/value;
  size_t blockValue = ceil(ratio);
  if(blockValue == 0) blockValue++;
  //Zhao's note: Default thread should always be 64, 128, 256, 512, ...
  // This is because we are using partial sums, if arraysize is smaller than defaultthread, we need to make sure that
  //while Nthread is dividing by 2, it does not generate ODD NUMBER (for example, 5/2 = 2, then element 5 will be ignored)//
  *Nthread = DEFAULTTHREAD;
  *Nblock = blockValue;
}

void VDWReal_Total_CPU(Boxsize Box, Atoms* Host_System, Atoms* System, ForceField FF, Components SystemComponents, MoveEnergy& E)
{
  printf("****** Calculating VDW + Real Energy (CPU) ******\n");
  ///////////////////////////////////////////////////////
  //All variables passed here should be device pointers//
  ///////////////////////////////////////////////////////
  //Copy Adsorbate to host//
  for(size_t ijk=1; ijk < SystemComponents.NComponents.x; ijk++) //Skip the first one(framework)
  {
    //if(Host_System[ijk].Allocate_size != System[ijk].Allocate_size)
    //{
      // if the host allocate_size is different from the device, allocate more space on the host
      Host_System[ijk].pos       = (double3*) malloc(System[ijk].Allocate_size*sizeof(double3));
      Host_System[ijk].scale     = (double*)  malloc(System[ijk].Allocate_size*sizeof(double));
      Host_System[ijk].charge    = (double*)  malloc(System[ijk].Allocate_size*sizeof(double));
      Host_System[ijk].scaleCoul = (double*)  malloc(System[ijk].Allocate_size*sizeof(double));
      Host_System[ijk].Type      = (size_t*)  malloc(System[ijk].Allocate_size*sizeof(size_t));
      Host_System[ijk].MolID     = (size_t*)  malloc(System[ijk].Allocate_size*sizeof(size_t));
      Host_System[ijk].size      = System[ijk].size; 
      Host_System[ijk].Allocate_size = System[ijk].Allocate_size;
    //}
  
    //if(Host_System[ijk].Allocate_size = System[ijk].Allocate_size) //means there is no more space allocated on the device than host, otherwise, allocate more on host
    //{
      cudaMemcpy(Host_System[ijk].pos, System[ijk].pos, sizeof(double3)*System[ijk].Allocate_size, cudaMemcpyDeviceToHost);
      cudaMemcpy(Host_System[ijk].scale, System[ijk].scale, sizeof(double)*System[ijk].Allocate_size, cudaMemcpyDeviceToHost);
      cudaMemcpy(Host_System[ijk].charge, System[ijk].charge, sizeof(double)*System[ijk].Allocate_size, cudaMemcpyDeviceToHost);
      cudaMemcpy(Host_System[ijk].scaleCoul, System[ijk].scaleCoul, sizeof(double)*System[ijk].Allocate_size, cudaMemcpyDeviceToHost);
      cudaMemcpy(Host_System[ijk].Type, System[ijk].Type, sizeof(size_t)*System[ijk].Allocate_size, cudaMemcpyDeviceToHost);
      cudaMemcpy(Host_System[ijk].MolID, System[ijk].MolID, sizeof(size_t)*System[ijk].Allocate_size, cudaMemcpyDeviceToHost);
      Host_System[ijk].size = System[ijk].size;
      //printf("CPU CHECK: comp: %zu, Host Allocate_size: %zu, Allocate_size: %zu\n", ijk, Host_System[ijk].Allocate_size, System[ijk].Allocate_size);
    //}
  }
  //Write to a file for checking//
  std::ofstream textrestartFile{};
  std::string dirname="FirstBead/";
  std::string fname  = dirname + "/" + "Energy.data";
  std::filesystem::path cwd = std::filesystem::current_path();

  std::filesystem::path directoryName = cwd /dirname;
  std::filesystem::path fileName = cwd /fname;
  std::filesystem::create_directories(directoryName);

  textrestartFile = std::ofstream(fileName, std::ios::out);
  textrestartFile << "PosA PosB TypeA TypeB E" <<"\n";

  std::vector<double>Total_VDW  = {0.0, 0.0, 0.0};
  std::vector<double>Total_Real = {0.0, 0.0, 0.0};
  size_t count = 0; size_t cutoff_count=0;
  double VDW_energy   = 0.0; double Coul_energy = 0.0;
  //FOR DEBUGGING ENERGY//
  MoveEnergy FirstBeadE; MoveEnergy ChainE; size_t FBHGPairs=0; size_t FBGGPairs = 0;
                                            size_t CHGPairs=0;  size_t CGGPairs = 0;
  size_t selectedComp= 4;
  size_t selectedMol = 44;
  //size_t selectedMol = SystemComponents.NumberOfMolecule_for_Component[selectedComp] - 1;
  std::vector<double> FBES; std::vector<double> CHAINES;
  std::vector<double2> ComponentEnergy(SystemComponents.NComponents.x * SystemComponents.NComponents.x, {0.0, 0.0});
  int InteractionType;
  for(size_t compi=0; compi < SystemComponents.NComponents.x; compi++) 
  {
    const Atoms Component=Host_System[compi];
    for(size_t i=0; i<Component.size; i++)
    {
      //printf("comp: %zu, i: %zu, x: %.10f\n", compi, i, Component.pos[i].x);
      const double scaleA = Component.scale[i];
      const double chargeA = Component.charge[i];
      const double scalingCoulombA = Component.scaleCoul[i];
      const size_t typeA = Component.Type[i];
      const size_t MoleculeID = Component.MolID[i];
      for(size_t compj=0; compj < SystemComponents.NComponents.x; compj++)
      {
        //Determine Interaction type//
        if((compi < SystemComponents.NComponents.y) || (compj < SystemComponents.NComponents.y))
        {
          if(!(compi < SystemComponents.NComponents.y) || !(compj < SystemComponents.NComponents.y))
          {
            InteractionType = HG;
          }
          else
          {
            InteractionType = HH;
          }
        }
        else
        {
            InteractionType = GG;
        }

        const Atoms Componentj=Host_System[compj];
        for(size_t j=0; j<Componentj.size; j++)
          {
            const double scaleB = Componentj.scale[j];
            const double chargeB = Componentj.charge[j];
            const double scalingCoulombB = Componentj.scaleCoul[j];
            const size_t typeB = Componentj.Type[j];
            const size_t MoleculeIDB = Componentj.MolID[j];
            if(!((MoleculeID == MoleculeIDB) && (compi == compj)))
            {
              count++;
              double3 posvec = Component.pos[i] - Componentj.pos[j];
              PBC(posvec, Box.Cell, Box.InverseCell, Box.Cubic);
              const double rr_dot = dot(posvec, posvec);
              //printf("i: %zu, j: %zu, rr_dot: %.10f\n", i,j,rr_dot);
              //if((compi > 0) && (compj > 0)) printf("CHECK_DIST: Compi: %zu Mol[%zu], compj: %zu Mol[%zu], rr_dot: %.5f\n", compi, MoleculeID, compj, MoleculeIDB, rr_dot);
              double result[2] = {0.0, 0.0};
              if(rr_dot < FF.CutOffVDW)
              {
                cutoff_count++;
                const double scaling = scaleA * scaleB;
                const size_t row = typeA*FF.size+typeB;
                const double FFarg[4] = {FF.epsilon[row], FF.sigma[row], FF.z[row], FF.shift[row]};
                VDW(FFarg, rr_dot, scaling, result);
                Total_VDW[InteractionType] += 0.5*result[0];
                //if((compi > 0) && (compj > 0)) printf("Compi: %zu Mol[%zu], compj: %zu Mol[%zu], GG_E: %.5f\n", compi, MoleculeID, compj, MoleculeIDB, result[0]);
                VDW_energy   += 0.5*result[0];
                ComponentEnergy[compi * SystemComponents.NComponents.x + compj].x += 0.5*result[0];
                //if(std::abs(result[0]) > 10000) printf("Very High Energy (VDW), comps: %zu, %zu, MolID: %zu %zu, Atom: %zu %zu, E: %.5f\n", compi, compj, MoleculeID, MoleculeIDB, i, j, result[0]);
                //DEBUG//
                if(MoleculeID == selectedMol && (compi == selectedComp)) 
                {
                  if(i%Component.Molsize == 0)//First Bead//
                  {
                    if(compj == 0){FirstBeadE.HGVDW += result[0]; FBHGPairs ++;}
                    else
                    {
                      //printf("FB_GG: posi: %zu, typeA: %zu, comp: %zu, ENERGY: %.5f\n", j, typeB, compj, result[0]);
                      FirstBeadE.GGVDW += result[0]; FBGGPairs ++;
                    }
                  }
                  else
                  { if(compj == 0){ChainE.HGVDW += result[0]; CHGPairs ++;}
                    else
                    {
                      //printf("Chain_GG: posi: %zu, typeA: %zu, comp: %zu, ENERGY: %.5f\n", j, typeB, compj, result[0]);
                      ChainE.GGVDW += result[0]; CGGPairs ++;
                    } 
                  }
                }
              }
              if (!FF.noCharges && rr_dot < FF.CutOffCoul)
              {
                const double r = sqrt(rr_dot);
                const double scalingCoul = scalingCoulombA * scalingCoulombB;
                double resultCoul[2] = {0.0, 0.0};
                CoulombReal(chargeA, chargeB, r, scalingCoul, resultCoul, Box.Prefactor, Box.Alpha);
                Total_Real[InteractionType] += 0.5*resultCoul[0];
                Coul_energy  += 0.5*resultCoul[0];
                ComponentEnergy[compi * SystemComponents.NComponents.x + compj].y += 0.5*resultCoul[0];
                //if(std::abs(result[0]) > 10000) printf("Very High Energy (Coul), comps: %zu, %zu, MolID: %zu %zu, Atom: %zu %zu, E: %.5f\n", compi, compj, MoleculeID, MoleculeIDB, i, j, resultCoul[0]);
                //DEBUG//
                if(MoleculeID == selectedMol && (compi == selectedComp))
                {
                  if(i%Component.Molsize == 0)//First Bead//
                  {
                    if(compj == 0){FirstBeadE.HGReal += resultCoul[0]; FBHGPairs ++;}
                    else
                    {
                      FirstBeadE.GGReal += resultCoul[0]; FBGGPairs ++;
                    }
                  }
                  else
                  { if(compj == 0){ChainE.HGReal += resultCoul[0]; CHGPairs ++;}
                    else{ChainE.GGReal += resultCoul[0]; CGGPairs ++;}
                  }
                }
              }
            }
          }
      }
    }  
  }
  //printf("%zu interactions, within cutoff: %zu, energy: %.10f\n", count, Total_energy, cutoff_count);
  printf("Host-Host   VDW: %.5f; Real: %.5f\n", Total_VDW[HH], Total_Real[HH]);
  printf("Host-Guest  VDW: %.5f; Real: %.5f\n", Total_VDW[HG], Total_Real[HG]);
  printf("Guest-Guest VDW: %.5f; Real: %.5f\n", Total_VDW[GG], Total_Real[GG]);

  E.HHVDW = Total_VDW[HH]; E.HHReal= Total_Real[HH];
  E.HGVDW = Total_VDW[HG]; E.HGReal= Total_Real[HG];
  E.GGVDW = Total_VDW[GG]; E.GGReal= Total_Real[GG];

  printf("********** PRINTING COMPONENT ENERGIES**********\n");
  for(size_t i = 0; i < SystemComponents.NComponents.x; i++)
    for(size_t j = i; j < SystemComponents.NComponents.x; j++)
    {
      double VDW = (i == j) ? ComponentEnergy[i * SystemComponents.NComponents.x + j].x : 2.0 * ComponentEnergy[i * SystemComponents.NComponents.x + j].x;
      double Real = (i == j) ? ComponentEnergy[i * SystemComponents.NComponents.x + j].y : 2.0 * ComponentEnergy[i * SystemComponents.NComponents.x + j].y;
      printf("Compoent [%zu-%zu], VDW: %.5f, Real: %.5f\n", i, j, VDW, Real);
      
    }
  textrestartFile.close();
}

////////////////////////////// GPU CODE //////////////////////////

__global__ void one_thread_GPU_test(Boxsize Box, Atoms* System, ForceField FF, double* xxx)
{
  bool DEBUG=false;
  //Zhao's note: added temp_xxx values for checking individual energy for each molecule//
  double temp_energy = 0.0; double temp_firstbead = 0.0; double temp_chain = 0.0; int temp_count = -1;
  double Total_energy = 0.0; size_t count = 0; size_t cutoff_count=0;
  double VDW_energy = 0.0; double Coul_energy = 0.0;
  for(size_t compi=0; compi < 2; compi++) //Zhao's note: hard coded component, change
  {
    const Atoms Component=System[compi];
    //printf("GPU CHECK: Comp: %lu, Comp size: %lu, Allocate size: %lu\n", compi, Component.size, Component.Allocate_size);
    for(size_t i=0; i<Component.size; i++)
    {
      //printf("comp: %lu, i: %lu, x: %.10f\n", compi, i, Component.pos[i].x);
      const double scaleA = Component.scale[i];
      const double chargeA = Component.charge[i];
      const double scalingCoulombA = Component.scaleCoul[i];
      const size_t typeA = Component.Type[i];
      const size_t MoleculeID = Component.MolID[i];
      if(DEBUG){if(MoleculeID == 5) temp_count++;} //For testing individual molecule energy//
      for(size_t compj=0; compj < 2; compj++) //Zhao's note: hard coded component, change
      {
        if(!((compi == 0) && (compj == 0))) //ignore fraemwrok-framework interaction
        {
          const Atoms Componentj=System[compj];
          for(size_t j=0; j<Componentj.size; j++)
          {
            const double scaleB = Componentj.scale[j];
            const double chargeB = Componentj.charge[j];
            const double scalingCoulombB = Componentj.scaleCoul[j];
            const size_t typeB = Componentj.Type[j];
            const size_t MoleculeIDB = Componentj.MolID[j];
            if(!((MoleculeID == MoleculeIDB) &&(compi == compj)))
            {
              count++;
              double3 posvec = Component.pos[i] - Componentj.pos[j];
              PBC(posvec, Box.Cell, Box.InverseCell, Box.Cubic);

              const double rr_dot = dot(posvec, posvec);
              double result[2] = {0.0, 0.0};
              if(rr_dot < FF.CutOffVDW)
              {
                cutoff_count++;
                const double scaling = scaleA * scaleB;
                const size_t row = typeA*FF.size+typeB;
                const double FFarg[4] = {FF.epsilon[row], FF.sigma[row], FF.z[row], FF.shift[row]};
                VDW(FFarg, rr_dot, scaling, result);
                Total_energy += 0.5*result[0];
                VDW_energy   += 0.5*result[0];
                if(DEBUG){if(MoleculeID == 5)
                { 
                  temp_energy += result[0];
                  if(temp_count == 0){temp_firstbead += result[0];}
                  else {temp_chain += result[0];}
                } 
              }}
              //  printf("SPECIEL CHECK: compi: %lu, i: %lu, compj: %lu, j: %lu, pos: %.5f, %.5f, %.5f, rr_dot: %.10f, energy: %.10f\n", compi,i,compj,j,Component.pos[i].x, Component.pos[i].y, Component.pos[i].z, rr_dot, result[0]);
              if (!FF.noCharges && rr_dot < FF.CutOffCoul)
              {
                const double r = sqrt(rr_dot);
                const double scalingCoul = scalingCoulombA * scalingCoulombB;
                double resultCoul[2] = {0.0, 0.0};
                CoulombReal(chargeA, chargeB, r, scalingCoul, resultCoul, Box.Prefactor, Box.Alpha);
                Total_energy += 0.5*resultCoul[0]; //prefactor merged in the CoulombReal function
                Coul_energy  += 0.5*resultCoul[0];
              }
            }
          }
        }
      }
    }
  }
  if(DEBUG) printf("For Molecule 5, energy: %.10f, firstbead: %.10f, chain: %.10f\n", temp_energy, temp_firstbead, temp_chain);
  xxx[0] = Total_energy;
  printf("GPU (one Thread) Total Energy: %.5f, VDW Energy: %.5f, Coulomb Energy: %.5f\n", Total_energy, VDW_energy, Coul_energy);
  //printf("xxx: %.10f\n", Total_energy);
}
/*
__device__ __forceinline__ 
double fast_float2double (float a)
{
    unsigned int ia = __float_as_int (a);
    return __hiloint2double ((((ia >> 3) ^ ia) & 0x07ffffff) ^ ia, ia << 29);
}
*/

double CPU_EwaldDifference(Boxsize& Box, Atoms& New, Atoms& Old, ForceField& FF, Components& SystemComponents, size_t SelectedComponent, bool Swap, size_t SelectedTrial)
{
  int kx_max = Box.kmax.x;
  int ky_max = Box.kmax.y;
  int kz_max = Box.kmax.z;
  if(FF.noCharges) return 0.0;
  double alpha = Box.Alpha; double alpha_squared = alpha * alpha;
  double prefactor = Box.Prefactor * (2.0 * M_PI / Box.Volume);

  double ewaldE = 0.0;

  Boxsize Host_Box;
  Host_Box.Cell        = (double*) malloc(9 * sizeof(double)); cudaMemcpy(Host_Box.Cell, Box.Cell, 9*sizeof(double), cudaMemcpyDeviceToHost);
  Host_Box.InverseCell = (double*) malloc(9 * sizeof(double)); cudaMemcpy(Host_Box.InverseCell, Box.InverseCell, 9*sizeof(double), cudaMemcpyDeviceToHost);
  Host_Box.Cubic       = Box.Cubic;

  double ax[3] = {Host_Box.InverseCell[0], Host_Box.InverseCell[3], Host_Box.InverseCell[6]}; //printf("ax: %.10f, %.10f, %.10f\n", Box.InverseCell[0], Box.InverseCell[3], Box.InverseCell[6]);
  double ay[3] = {Host_Box.InverseCell[1], Host_Box.InverseCell[4], Host_Box.InverseCell[7]}; //printf("ay: %.10f, %.10f, %.10f\n", Box.InverseCell[1], Box.InverseCell[4], Box.InverseCell[7]);
  double az[3] = {Host_Box.InverseCell[2], Host_Box.InverseCell[5], Host_Box.InverseCell[8]}; //printf("az: %.10f, %.10f, %.10f\n", Box.InverseCell[2], Box.InverseCell[5], Box.InverseCell[8]);

  size_t Oldsize = 0; size_t Newsize = SystemComponents.Moleculesize[SelectedComponent];

  size_t numberOfAtoms = SystemComponents.Moleculesize[SelectedComponent];
  //Zhao's note: translation/rotation/reinsertion involves new + old states. Insertion/Deletion only has the new state.
  if(!Swap)
  {
    numberOfAtoms  += SystemComponents.Moleculesize[SelectedComponent];
    Oldsize         = SystemComponents.Moleculesize[SelectedComponent];
  }
  //std::vector<std::complex<double>>eik_x(numberOfAtoms * (kx_max + 1));
  //std::vector<std::complex<double>>eik_y(numberOfAtoms * (ky_max + 1));
  //std::vector<std::complex<double>>eik_z(numberOfAtoms * (kz_max + 1));
  //std::vector<std::complex<double>>eik_xy(numberOfAtoms);
  //std::vector<std::complex<double>>tempEik(SystemComponents.AdsorbateEik.size());
  size_t numberOfWaveVectors = (kx_max + 1) * (2 * ky_max + 1) * (2 * kz_max + 1);
  SystemComponents.tempEik.resize(numberOfWaveVectors);
  //Create Temporary Struct for storing values//
  Atoms TempAtoms;
  TempAtoms.pos       = (double3*) malloc(numberOfAtoms * sizeof(double3));
  TempAtoms.scale     = (double*)  malloc(numberOfAtoms * sizeof(double));
  TempAtoms.charge    = (double*)  malloc(numberOfAtoms * sizeof(double));
  TempAtoms.scaleCoul = (double*)  malloc(numberOfAtoms * sizeof(double));
  TempAtoms.Type      = (size_t*)  malloc(numberOfAtoms * sizeof(size_t));
  TempAtoms.MolID     = (size_t*)  malloc(numberOfAtoms * sizeof(size_t));
  if(Swap)
  {
    //Copy the NEW values, first bead first, First bead stored in Old, at the first element//
    cudaMemcpy(TempAtoms.pos,       Old.pos,       sizeof(double3), cudaMemcpyDeviceToHost);
    cudaMemcpy(TempAtoms.scale,     Old.scale,     sizeof(double),  cudaMemcpyDeviceToHost);
    cudaMemcpy(TempAtoms.charge,    Old.charge,    sizeof(double),  cudaMemcpyDeviceToHost);
    cudaMemcpy(TempAtoms.scaleCoul, Old.scaleCoul, sizeof(double),  cudaMemcpyDeviceToHost);
    cudaMemcpy(TempAtoms.Type,      Old.Type,      sizeof(size_t),  cudaMemcpyDeviceToHost);
    cudaMemcpy(TempAtoms.MolID,     Old.MolID,     sizeof(size_t),  cudaMemcpyDeviceToHost);
    //Copy the NEW Orientation for the selected Trial, stored in New//
    size_t chainsize  = SystemComponents.Moleculesize[SelectedComponent] - 1;
    size_t selectsize = chainsize * SelectedTrial;
    cudaMemcpy(&TempAtoms.pos[1],       &New.pos[selectsize],       chainsize * sizeof(double3),cudaMemcpyDeviceToHost);
    cudaMemcpy(&TempAtoms.scale[1],     &New.scale[selectsize],     chainsize * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(&TempAtoms.charge[1],    &New.charge[selectsize],    chainsize * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(&TempAtoms.scaleCoul[1], &New.scaleCoul[selectsize], chainsize * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(&TempAtoms.Type[1],      &New.Type[selectsize],      chainsize * sizeof(size_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(&TempAtoms.MolID[1],     &New.MolID[selectsize],     chainsize * sizeof(size_t), cudaMemcpyDeviceToHost);
  }
  else //Translation/Rotation//
  {
    cudaMemcpy(TempAtoms.pos,       Old.pos,       Oldsize * sizeof(double3), cudaMemcpyDeviceToHost);
    cudaMemcpy(TempAtoms.scale,     Old.scale,     Oldsize * sizeof(double),  cudaMemcpyDeviceToHost);
    cudaMemcpy(TempAtoms.charge,    Old.charge,    Oldsize * sizeof(double),  cudaMemcpyDeviceToHost);
    cudaMemcpy(TempAtoms.scaleCoul, Old.scaleCoul, Oldsize * sizeof(double),  cudaMemcpyDeviceToHost);
    cudaMemcpy(TempAtoms.Type,      Old.Type,      Oldsize * sizeof(size_t),  cudaMemcpyDeviceToHost);
    cudaMemcpy(TempAtoms.MolID,     Old.MolID,     Oldsize * sizeof(size_t),  cudaMemcpyDeviceToHost);

    cudaMemcpy(&TempAtoms.pos[Oldsize],       New.pos,       Newsize * sizeof(double3), cudaMemcpyDeviceToHost);
    cudaMemcpy(&TempAtoms.scale[Oldsize],     New.scale,     Newsize * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(&TempAtoms.charge[Oldsize],    New.charge,    Newsize * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(&TempAtoms.scaleCoul[Oldsize], New.scaleCoul, Newsize * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(&TempAtoms.Type[Oldsize],      New.Type,      Newsize * sizeof(size_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(&TempAtoms.MolID[Oldsize],     New.MolID,     Newsize * sizeof(size_t), cudaMemcpyDeviceToHost);
  }
  for(size_t i=0; i < numberOfAtoms; i++) printf("TempAtoms: %.5f %.5f %.5f\n", TempAtoms.pos[i].x, TempAtoms.pos[i].y, TempAtoms.pos[i].z);
  double start = omp_get_wtime();  
  // Construct exp(ik.r) for atoms and k-vectors kx, ky, kz = 0, 1 explicitly
  size_t count=0;
  //Old//
  for(size_t posi=0; posi < Oldsize; ++posi)
  {
    //determine the component for i
    double3 pos = TempAtoms.pos[posi];
    SystemComponents.eik_x[posi + 0 * numberOfAtoms] = std::complex<double>(1.0, 0.0);
    SystemComponents.eik_y[posi + 0 * numberOfAtoms] = std::complex<double>(1.0, 0.0);
    SystemComponents.eik_z[posi + 0 * numberOfAtoms] = std::complex<double>(1.0, 0.0);
    double3 s; matrix_multiply_by_vector(Host_Box.InverseCell, pos, s); s*=2*M_PI;
    SystemComponents.eik_x[posi + 1 * numberOfAtoms] = std::complex<double>(std::cos(s.x), std::sin(s.x));
    SystemComponents.eik_y[posi + 1 * numberOfAtoms] = std::complex<double>(std::cos(s.y), std::sin(s.y));
    SystemComponents.eik_z[posi + 1 * numberOfAtoms] = std::complex<double>(std::cos(s.z), std::sin(s.z));
    count++; 
  }
  //New//
  for(size_t posi=Oldsize; posi < Oldsize + Newsize; ++posi)
  {
    //determine the component for i
    double3 pos = TempAtoms.pos[posi];
    SystemComponents.eik_x[posi + 0 * numberOfAtoms] = std::complex<double>(1.0, 0.0);
    SystemComponents.eik_y[posi + 0 * numberOfAtoms] = std::complex<double>(1.0, 0.0);
    SystemComponents.eik_z[posi + 0 * numberOfAtoms] = std::complex<double>(1.0, 0.0);
    double3 s; matrix_multiply_by_vector(Host_Box.InverseCell, pos, s); s*=2*M_PI;
    SystemComponents.eik_x[posi + 1 * numberOfAtoms] = std::complex<double>(std::cos(s.x), std::sin(s.x));
    SystemComponents.eik_y[posi + 1 * numberOfAtoms] = std::complex<double>(std::cos(s.y), std::sin(s.y));
    SystemComponents.eik_z[posi + 1 * numberOfAtoms] = std::complex<double>(std::cos(s.z), std::sin(s.z));
    count++;
  }
  // Calculate remaining positive kx, ky and kz by recurrence
  for(size_t kx = 2; kx <= kx_max; ++kx)
  {
    for(size_t i = 0; i != numberOfAtoms; ++i)
    {
      SystemComponents.eik_x[i + kx * numberOfAtoms] = SystemComponents.eik_x[i + (kx - 1) * numberOfAtoms] * SystemComponents.eik_x[i + 1 * numberOfAtoms];
    }
  }
  for(size_t ky = 2; ky <= ky_max; ++ky)
  {
    for(size_t i = 0; i != numberOfAtoms; ++i)
    {
      SystemComponents.eik_y[i + ky * numberOfAtoms] = SystemComponents.eik_y[i + (ky - 1) * numberOfAtoms] * SystemComponents.eik_y[i + 1 * numberOfAtoms];
    }
  }
  for(size_t kz = 2; kz <= kz_max; ++kz)
  {
    for(size_t i = 0; i != numberOfAtoms; ++i)
    {
      SystemComponents.eik_z[i + kz * numberOfAtoms] = SystemComponents.eik_z[i + (kz - 1) * numberOfAtoms] * SystemComponents.eik_z[i + 1 * numberOfAtoms];
    }
  }
 
  size_t nvec = 0;
  std::complex<double> cksum_old(0.0, 0.0);
  std::complex<double> cksum_new(0.0, 0.0);
  //for debugging
  for(std::make_signed_t<std::size_t> kx = 0; kx <= kx_max; ++kx)
  {
    double kvec_x[3]; for(size_t j = 0; j < 3; j++) kvec_x[j] = 2.0 * M_PI * static_cast<double>(kx) * ax[j];
    // Only positive kx are used, the negative kx are taken into account by the factor of two
    double factor = (kx == 0) ? (1.0 * prefactor) : (2.0 * prefactor);

    for(std::make_signed_t<std::size_t> ky = -ky_max; ky <= ky_max; ++ky)
    {
      double kvec_y[3]; for(size_t j = 0; j < 3; j++) kvec_y[j] = 2.0 * M_PI * static_cast<double>(ky) * ay[j];
      // Precompute and store eik_x * eik_y outside the kz-loop
      // OLD //
      for(size_t i = 0; i != numberOfAtoms; ++i)
      {
        std::complex<double> eiky_temp = SystemComponents.eik_y[i + numberOfAtoms * static_cast<size_t>(std::abs(ky))];
        eiky_temp.imag(ky>=0 ? eiky_temp.imag() : -eiky_temp.imag());
        SystemComponents.eik_xy[i] = SystemComponents.eik_x[i + numberOfAtoms * static_cast<size_t>(kx)] * eiky_temp;
      }

      for(std::make_signed_t<std::size_t> kz = -kz_max; kz <= kz_max; ++kz)
      {
        // Ommit kvec==0
        double ksqr = static_cast<double>(kx * kx + ky * ky + kz * kz);
        if(Box.UseLAMMPSEwald) //Overwrite ksqr if we use the LAMMPS Setup for Ewald//
        {
          const double lx = Box.Cell[0];
          const double ly = Box.Cell[4];
          const double lz = Box.Cell[8];
          const double xy = Box.Cell[3];
          const double xz = Box.Cell[6];
          const double yz = Box.Cell[7];
          const double ux = 2*M_PI/lx;
          const double uy = 2*M_PI*(-xy)/lx/ly;
          const double uz = 2*M_PI*(xy*yz - ly*xz)/lx/ly/lz;
          const double vy = 2*M_PI/ly;
          const double vz = 2*M_PI*(-yz)/ly/lz;
          const double wz = 2*M_PI/lz;
          const double kvecx = kx*ux;
          const double kvecy = kx*uy + ky*vy;
          const double kvecz = kx*uz + ky*vz + kz*wz;
          ksqr  = kvecx*kvecx + kvecy*kvecy + kvecz*kvecz;
        }
        if((ksqr > 1e-10) && (ksqr < Box.ReciprocalCutOff))
        {
          double kvec_z[3]; for(size_t j = 0; j < 3; j++) kvec_z[j] = 2.0 * M_PI * static_cast<double>(kz) * az[j];
          cksum_old = std::complex<double>(0.0, 0.0); cksum_new = std::complex<double>(0.0, 0.0);
          count=0;
          //OLD//
          for(size_t posi=0; posi<Oldsize; ++posi)
          {
            std::complex<double> eikz_temp = SystemComponents.eik_z[posi + numberOfAtoms * static_cast<size_t>(std::abs(kz))];
            eikz_temp.imag(kz>=0 ? eikz_temp.imag() : -eikz_temp.imag());
            double charge  = TempAtoms.charge[posi];
            double scaling = TempAtoms.scaleCoul[posi];
            cksum_old     += scaling * charge * (SystemComponents.eik_xy[posi] * eikz_temp);
            count++;
          }
          //NEW//
          for(size_t posi=Oldsize; posi<Oldsize + Newsize; ++posi)
          {
            std::complex<double> eikz_temp = SystemComponents.eik_z[posi + numberOfAtoms * static_cast<size_t>(std::abs(kz))];
            eikz_temp.imag(kz>=0 ? eikz_temp.imag() : -eikz_temp.imag());
            double charge  = TempAtoms.charge[posi];
            double scaling = TempAtoms.scaleCoul[posi];
            cksum_new     += scaling * charge * (SystemComponents.eik_xy[posi] * eikz_temp);
            count++;
          }
          //double rksq = (kvec_x + kvec_y + kvec_z).length_squared();
          double tempkvec[3] = {kvec_x[0]+kvec_y[0]+kvec_z[0], kvec_x[1]+kvec_y[1]+kvec_z[1], kvec_x[2]+kvec_y[2]+kvec_z[2]};
          double rksq = tempkvec[0]*tempkvec[0] + tempkvec[1]*tempkvec[1] + tempkvec[2]*tempkvec[2];
          double temp = factor * std::exp((-0.25 / alpha_squared) * rksq) / rksq;
          //std::complex<double> cksum_sub = SystemComponents.AdsorbateEik[nvec];
          //std::complex<double> cksum_add = SystemComponents.AdsorbateEik[nvec] + cksum_new - cksum_old;
          //double tempsum_add = temp * (cksum_add.real() * cksum_add.real() + cksum_add.imag() * cksum_add.imag());
          //double tempsum_sub = temp * (cksum_sub.real() * cksum_sub.real() + cksum_sub.imag() * cksum_sub.imag());
          //ewaldE += tempsum_add;
          //ewaldE -= tempsum_sub;
          //std::complex<double> cksum_add = SystemComponents.AdsorbateEik[nvec] + cksum_new - cksum_old;
          //std::complex<double> cksum = cksum_new - cksum_old;
          //double tempE;
          //tempE  += temp * std::norm(SystemComponents.AdsorbateEik[nvec] + cksum_new - cksum_old);
          //tempE  -= temp * std::norm(SystemComponents.AdsorbateEik[nvec]);
          ewaldE += temp * std::norm(SystemComponents.AdsorbateEik[nvec] + cksum_new - cksum_old);
          ewaldE -= temp * std::norm(SystemComponents.AdsorbateEik[nvec]);
          //ewaldE += tempE;
          SystemComponents.tempEik[nvec] = SystemComponents.AdsorbateEik[nvec] + cksum_new - cksum_old;
          //++nvec;
          //printf("GPU kx/ky/kz: %d %d %d temp: %.5f, tempE: %.5f\n", kx, ky, kz, temp, tempE);
          //printf("CPU kx/ky/kz: %d %d %d, new Vector: %.5f %.5f, cksum: %.5f %.5f, stored: %.5f %.5f\n", kx, ky, kz, cksum_add.real(), cksum_add.imag(), cksum.real(), cksum.imag(), SystemComponents.AdsorbateEik[nvec].real(), SystemComponents.AdsorbateEik[nvec].imag());
          //printf("CPU kx/ky/kz: %d %d %d, new Vector: %.5f %.5f, cknew: %.5f %.5f, ckold: %.5f %.5f\n", kx, ky, kz, cksum_add.real(), cksum_add.imag(), cksum_new.real(), cksum_new.imag(), cksum_old.real(), cksum_old.imag());
        } 
        ++nvec;
      }
    }
  }

  double end = omp_get_wtime();
  printf("CPU Fourier took: %.12f sec, Post-Fourier (CPU) energy is %.5f\n", end - start, ewaldE);

  ///////////////////////////////
  // Subtract exclusion-energy // Zhao's note: taking out the pairs of energies that belong to the same molecule
  ///////////////////////////////
  double SelfEOld = 0.0; double SelfENew = 0.0;
  count=0;
  //OLD//
  for(size_t posi=0; posi<Oldsize; posi++)
  {
    double  charge  = TempAtoms.charge[posi];
    double  scaling = TempAtoms.scaleCoul[posi];
    double  factorA = charge * scaling;
    double3 posA = TempAtoms.pos[posi];
    for(size_t posj=posi+1; posj < Oldsize; posj++)
    {
      double  charge  = TempAtoms.charge[posj];
      double  scaling = TempAtoms.scaleCoul[posj];
      double  factorB = charge * scaling;
      double3 posB = TempAtoms.pos[posj];
      double3 posvec = posA - posB;
      PBC(posvec, Host_Box.Cell, Host_Box.InverseCell, Host_Box.Cubic);
      //double rr_dot = posvec[0]*posvec[0] + posvec[1]*posvec[1] + posvec[2]*posvec[2];
      double rr_dot = dot(posvec, posvec);
      double r = std::sqrt(rr_dot);
      SelfEOld      += Box.Prefactor * factorA * factorB * std::erf(alpha * r) / r;
    }
  }
  //NEW//
  for(size_t posi=Oldsize; posi!=Oldsize + Newsize; posi++)
  {
    double  charge  = TempAtoms.charge[posi];
    double  scaling = TempAtoms.scaleCoul[posi];
    double  factorA = charge * scaling;
    double3 posA = TempAtoms.pos[posi];
    for(size_t posj=posi+1; posj != Oldsize + Newsize; posj++)
    {
      double charge  = TempAtoms.charge[posj];
      double scaling = TempAtoms.scaleCoul[posj];
      double factorB = charge * scaling;
      double3 posB = TempAtoms.pos[posj];
      double3 posvec = posA - posB;
      PBC(posvec, Host_Box.Cell, Host_Box.InverseCell, Host_Box.Cubic);
      double rr_dot = dot(posvec, posvec);
      double r = std::sqrt(rr_dot);
      SelfENew      += Box.Prefactor * factorA * factorB * std::erf(alpha * r) / r;
    }
  }
  ewaldE -= SelfENew - SelfEOld;

  //////////////////////////
  // Subtract self-energy //
  //////////////////////////
  double prefactor_self = Box.Prefactor * alpha / std::sqrt(M_PI);
  double SelfExcludeOld = 0.0; double SelfExcludeNew = 0.0;
  //OLD//
  for(size_t i = 0; i != Oldsize; ++i)
  {
    double charge   = TempAtoms.charge[i];
    double scale    = TempAtoms.scale[i];
    SelfExcludeOld += prefactor_self * charge * charge * scale * scale;
  }
  //NEW//
  size_t j_count = 0;
  for(size_t i = Oldsize; i != Oldsize + Newsize; ++i)
  {
    double charge   = TempAtoms.charge[i];
    double scale    = TempAtoms.scale[i];
    SelfExcludeNew += prefactor_self * charge * charge * scale * scale;
    j_count++;
  }
  ewaldE -= SelfExcludeNew - SelfExcludeOld;

  return ewaldE;
}

__global__ void Calculate_Single_Body_Energy_SEPARATE_HostGuest_VDWReal(Boxsize Box, Atoms* System, Atoms Old, Atoms New, ForceField FF, double* BlockEnergy, size_t ComponentID, size_t totalAtoms, size_t chainsize, bool* flag, int3 Nblocks, bool Do_New, bool Do_Old, int3 NComps)
{
  //divide species into Host-Host, Host-Guest, and Guest-Guest//
  //However, Host-Host and Guest-Guest are mutually exclusive//
  //If Host-Host, then there is no Guest-Guest//
  ///////////////////////////////////////////////////////
  //All variables passed here should be device pointers//
  ///////////////////////////////////////////////////////
  extern __shared__ double sdata[];
  int cache_id = threadIdx.x;
  size_t total_ij = blockIdx.x * blockDim.x + threadIdx.x;

  size_t HH_Nblock = static_cast<size_t>(Nblocks.x);
  size_t HG_Nblock = static_cast<size_t>(Nblocks.y);
  size_t GG_Nblock = static_cast<size_t>(Nblocks.z);
  size_t Total_Nblock = HH_Nblock + HG_Nblock + GG_Nblock; 

  //if(total_ij == 0) printf("HH: %lu, HG: %lu, GG: %lu, Total_Nblock: %lu\n", HH_Nblock, HG_Nblock, GG_Nblock, Total_Nblock);

  size_t ij_within_block = total_ij - blockIdx.x * blockDim.x;

  sdata[ij_within_block] = 0.0; sdata[ij_within_block + blockDim.x] = 0.0;
  //Initialize Blocksums//
  BlockEnergy[blockIdx.x] = 0.0; 
  BlockEnergy[blockIdx.x + Total_Nblock] = 0.0;

  //__shared__ bool Blockflag = false;

  const size_t NTotalComp = NComps.x;
  const size_t NHostComp  = NComps.y;
  //const size_t NGuestComp = NComps.z;

  size_t ij = total_ij; //relative ij (threadID) within the three types of blocks//
  
  if(blockIdx.x >= (HH_Nblock + HG_Nblock))
  {
    ij -= HG_Nblock * blockDim.x; //It doesn't belong to Host-Host/Host-Guest Interaction//
  }
  if(blockIdx.x  >= HH_Nblock)
  {
    ij -= HH_Nblock * blockDim.x; //It doesn't belong to Host-Host Interaction//
  }
  //Zhao's note: consider checking the Nblock for Guest-Guest As well here, if not good, trigger exception//
  // Manually fusing/collapsing the loop //
  size_t i = ij/chainsize;
  size_t j = ij%chainsize; // position in Old and New

  //printf("ij: %lu, i: %lu, j: %lu, trial: %lu, totalAtoms: %lu, totalthreads: %lu\n", ij,i,j,k,totalAtoms, totalthreads);
  size_t posi = i; size_t totalsize= 0;
  size_t startComp = 0; size_t endComp = 0;
  //If moved molecule = framework, then check HH/HG //
  //If moved molecule = adsorbate, then check HG/GG //
  if(blockIdx.x < HH_Nblock) //Host-Host Interaction//
  {
    startComp = 0; endComp = NHostComp;
  }
  else if(blockIdx.x >= (HH_Nblock + HG_Nblock)) //Guest-Guest Interaction//
  {
    startComp = NHostComp; endComp = NTotalComp;
  }
  else //Host-Guest Interaction//
  {
    if(ComponentID < NHostComp) //Moved molecule is a Framework species, read Adsorbate part//
    {
      startComp = NHostComp; endComp = NTotalComp;
    }
    else //Moved molecule is an Adsorbate species, read Framework part//
    {
      startComp = 0; endComp = NHostComp;
    }
  }

  size_t comp = startComp;
  for(size_t ijk = startComp; ijk < endComp; ijk++)
  {
    totalsize += System[ijk].size;
    if(posi >= System[ijk].size)
    {
      comp++;
      posi -= System[ijk].size;
    }
    else
    {break;}
  }

  //Also need to check the range of the components//
  //if host-host, then comp need to fall into the framework components//
  //if guest-guest, comp -> adsorbate components//
  //if host-guest, check the moved species//
  bool CompCheck = false;
  if(blockIdx.x < HH_Nblock) //Host-Host Interaction//
  {
    if(comp < NHostComp) CompCheck = true;
  }
  else if(blockIdx.x >= (HH_Nblock + HG_Nblock)) //Guest-Guest Interaction//
  {
    if((comp >= NHostComp) && (comp < NTotalComp)) CompCheck = true;
  }
  else //Host-Guest Interaction//
  {
    if(ComponentID < NHostComp) //Moved molecule is a Framework species, read Adsorbate part//
    {
      if((comp >= NHostComp) && (comp < NTotalComp)) CompCheck = true;
    }
    else //Moved molecule is an Adsorbate species, read Framework part//
    {
      if(comp < NHostComp) CompCheck = true;
    }
  }
  /*
  if(ij == 4740)
  {
    printf("total_ij: %lu, ij: %lu, comp: %lu, pos: %lu, CompCheck: %s\n", total_ij, ij, comp, posi, CompCheck ? "true" : "false");
  }
  */
  if(CompCheck)
  if(posi < System[comp].size)
  {
  const Atoms Component=System[comp];
  const double scaleA = Component.scale[posi];
  const double chargeA = Component.charge[posi];
  const double scalingCoulombA = Component.scaleCoul[posi];
  const size_t typeA = Component.Type[posi];
  const size_t MoleculeID = Component.MolID[posi];
  double2 tempy = {0.0, 0.0}; double tempdU = 0.0;
  if(!((MoleculeID == New.MolID[0]) &&(comp == ComponentID)) && Do_New) //ComponentID: Component ID for the molecule being translated
  {
    ///////////
    //  NEW  //
    ///////////
    double3 posvec = Component.pos[posi] - New.pos[j];
    PBC(posvec, Box.Cell, Box.InverseCell, Box.Cubic);
    double rr_dot = dot(posvec, posvec);
    double result[2] = {0.0, 0.0};
    if(rr_dot < FF.CutOffVDW)
    {
      const size_t typeB = New.Type[j];
      const double scaleB = New.scale[j];
      const double scaling = scaleA * scaleB;
      const size_t row = typeA*FF.size+typeB;
      const double FFarg[4] = {FF.epsilon[row], FF.sigma[row], FF.z[row], FF.shift[row]};
      VDW(FFarg, rr_dot, scaling, result);
      if(result[0] > FF.OverlapCriteria) { /*Blockflag = true;*/ flag[0] = true; }
      if(rr_dot < 0.01)                  { /*Blockflag = true;*/ flag[0] = true; } //DistanceCheck//
      tempy.x  += result[0];
      tempdU   += result[1];
    }

    if (!FF.noCharges && rr_dot < FF.CutOffCoul)
    {
      const double chargeB = New.charge[j];
      const double scalingCoulombB = New.scaleCoul[j];
      const double r = sqrt(rr_dot);
      const double scalingCoul = scalingCoulombA * scalingCoulombB;
      CoulombReal(chargeA, chargeB, r, scalingCoul, result, Box.Prefactor, Box.Alpha);
      tempy.y += result[0]; //prefactor merged in the CoulombReal function
    }
  }
  if(!((MoleculeID == Old.MolID[0]) &&(comp == ComponentID)) && Do_Old) //ComponentID: Component ID for the molecule being translated
  {
    ///////////
    //  OLD  //
    ///////////
    double3 posvec = Component.pos[posi] - Old.pos[j];
    PBC(posvec, Box.Cell, Box.InverseCell, Box.Cubic);
    double rr_dot = dot(posvec, posvec);
    double result[2] = {0.0, 0.0};
    if(rr_dot < FF.CutOffVDW)
    {
      const size_t typeB = Old.Type[j];
      const double scaleB = Old.scale[j];
      const double scaling = scaleA * scaleB;
      const size_t row = typeA*FF.size+typeB;
      const double FFarg[4] = {FF.epsilon[row], FF.sigma[row], FF.z[row], FF.shift[row]};
      VDW(FFarg, rr_dot, scaling, result);
      tempy.x  -= result[0];
      tempdU   -= result[1];
    }
    if (!FF.noCharges && rr_dot < FF.CutOffCoul)
    {
      const double chargeB = Old.charge[j];
      const double scalingCoulombB = Old.scaleCoul[j];
      const double r = sqrt(rr_dot);
      const double scalingCoul = scalingCoulombA * scalingCoulombB;
      CoulombReal(chargeA, chargeB, r, scalingCoul, result, Box.Prefactor, Box.Alpha);
      tempy.y -= result[0]; //prefactor merged in the CoulombReal function
    }
    //printf("BlockID: %i, threadID: %i, VDW: %.5f, Real: %.5f\n", blockIdx.x, threadIdx.x, tempy.x, tempy.y);
  }
  sdata[ij_within_block] = tempy.x; //sdata[ij_within_block].y = tempdU;
  sdata[ij_within_block + blockDim.x] = tempy.y;
  //if(total_ij == 4740) printf("ij: %lu, total_ij: %lu, VDW: %.5f, Real: %.5f\n", ij, total_ij, tempy.x, tempy.y);
  }
  __syncthreads();
  //Partial block sum//
  //if(!Blockflag)
  //{
    int shift_i=blockDim.x / 2;
    while(shift_i != 0)
    {
      if(cache_id < shift_i)
      {
        sdata[cache_id] += sdata[cache_id + shift_i];
        sdata[cache_id + blockDim.x] += sdata[cache_id + shift_i + blockDim.x];
      }
      __syncthreads();
      shift_i /= 2;
    }
    if(cache_id == 0)
    {
      //printf("BlockID: %i, VDW: %.5f, Real: %.5f\n", blockIdx.x, sdata[0], sdata[blockDim.x]);
      BlockEnergy[blockIdx.x]   = sdata[0]; 
      BlockEnergy[blockIdx.x + Total_Nblock] = sdata[blockDim.x]; //Shift it//
    }
  //}
}

__global__ void Calculate_Single_Body_Energy_SEPARATE_HostGuest_VDWReal_LambdaChange(Boxsize Box, Atoms* System, Atoms Old, Atoms New, ForceField FF, double* BlockEnergy, size_t ComponentID, size_t totalAtoms, size_t chainsize, bool* flag, int3 Nblocks, bool Do_New, bool Do_Old, int3 NComps, double2 newScale)
{
  //divide species into Host-Host, Host-Guest, and Guest-Guest//
  //However, Host-Host and Guest-Guest are mutually exclusive//
  //If Host-Host, then there is no Guest-Guest//
  ///////////////////////////////////////////////////////
  //All variables passed here should be device pointers//
  ///////////////////////////////////////////////////////
  extern __shared__ double sdata[];
  int cache_id = threadIdx.x;
  size_t total_ij = blockIdx.x * blockDim.x + threadIdx.x;

  size_t HH_Nblock = static_cast<size_t>(Nblocks.x);
  size_t HG_Nblock = static_cast<size_t>(Nblocks.y);
  size_t GG_Nblock = static_cast<size_t>(Nblocks.z);
  size_t Total_Nblock = HH_Nblock + HG_Nblock + GG_Nblock; 

  //if(total_ij == 0) printf("HH: %lu, HG: %lu, GG: %lu, Total_Nblock: %lu\n", HH_Nblock, HG_Nblock, GG_Nblock, Total_Nblock);

  size_t ij_within_block = total_ij - blockIdx.x * blockDim.x;

  sdata[ij_within_block] = 0.0; sdata[ij_within_block + blockDim.x] = 0.0;
  //Initialize Blocksums//
  BlockEnergy[blockIdx.x] = 0.0; 
  BlockEnergy[blockIdx.x + Total_Nblock] = 0.0;

  //__shared__ bool Blockflag = false;

  const size_t NTotalComp = NComps.x;
  const size_t NHostComp  = NComps.y;
  //const size_t NGuestComp = NComps.z;

  size_t ij = total_ij; //relative ij (threadID) within the three types of blocks//
  
  if(blockIdx.x >= (HH_Nblock + HG_Nblock))
  {
    ij -= HG_Nblock * blockDim.x; //It doesn't belong to Host-Host/Host-Guest Interaction//
  }
  if(blockIdx.x  >= HH_Nblock)
  {
    ij -= HH_Nblock * blockDim.x; //It doesn't belong to Host-Host Interaction//
  }
  //Zhao's note: consider checking the Nblock for Guest-Guest As well here, if not good, trigger exception//
  // Manually fusing/collapsing the loop //
  size_t i = ij/chainsize;
  size_t j = ij%chainsize; // position in Old and New

  //printf("ij: %lu, i: %lu, j: %lu, trial: %lu, totalAtoms: %lu, totalthreads: %lu\n", ij,i,j,k,totalAtoms, totalthreads);
  size_t posi = i; size_t totalsize= 0;
  size_t startComp = 0; size_t endComp = 0;
  //If moved molecule = framework, then check HH/HG //
  //If moved molecule = adsorbate, then check HG/GG //
  if(blockIdx.x < HH_Nblock) //Host-Host Interaction//
  {
    startComp = 0; endComp = NHostComp;
  }
  else if(blockIdx.x >= (HH_Nblock + HG_Nblock)) //Guest-Guest Interaction//
  {
    startComp = NHostComp; endComp = NTotalComp;
  }
  else //Host-Guest Interaction//
  {
    if(ComponentID < NHostComp) //Moved molecule is a Framework species, read Adsorbate part//
    {
      startComp = NHostComp; endComp = NTotalComp;
    }
    else //Moved molecule is an Adsorbate species, read Framework part//
    {
      startComp = 0; endComp = NHostComp;
    }
  }

  size_t comp = startComp;
  for(size_t ijk = startComp; ijk < endComp; ijk++)
  {
    totalsize += System[ijk].size;
    if(posi >= System[ijk].size)
    {
      comp++;
      posi -= System[ijk].size;
    }
    else
    {break;}
  }

  //Also need to check the range of the components//
  //if host-host, then comp need to fall into the framework components//
  //if guest-guest, comp -> adsorbate components//
  //if host-guest, check the moved species//
  bool CompCheck = false;
  if(blockIdx.x < HH_Nblock) //Host-Host Interaction//
  {
    if(comp < NHostComp) CompCheck = true;
  }
  else if(blockIdx.x >= (HH_Nblock + HG_Nblock)) //Guest-Guest Interaction//
  {
    if((comp >= NHostComp) && (comp < NTotalComp)) CompCheck = true;
  }
  else //Host-Guest Interaction//
  {
    if(ComponentID < NHostComp) //Moved molecule is a Framework species, read Adsorbate part//
    {
      if((comp >= NHostComp) && (comp < NTotalComp)) CompCheck = true;
    }
    else //Moved molecule is an Adsorbate species, read Framework part//
    {
      if(comp < NHostComp) CompCheck = true;
    }
  }
  /*
  if(ij == 4740)
  {
    printf("total_ij: %lu, ij: %lu, comp: %lu, pos: %lu, CompCheck: %s\n", total_ij, ij, comp, posi, CompCheck ? "true" : "false");
  }
  */
  if(CompCheck)
  if(posi < System[comp].size)
  {
  const Atoms Component=System[comp];
  const double scaleA = Component.scale[posi];
  const double chargeA = Component.charge[posi];
  const double scalingCoulombA = Component.scaleCoul[posi];
  const size_t typeA = Component.Type[posi];
  const size_t MoleculeID = Component.MolID[posi];
  double2 tempy = {0.0, 0.0}; double tempdU = 0.0;
  if(!((MoleculeID == Old.MolID[0]) &&(comp == ComponentID)) && Do_Old) //ComponentID: Component ID for the molecule being translated
  {
    ///////////
    //  OLD  //
    ///////////
    double3 posvec = Component.pos[posi] - Old.pos[j];
    PBC(posvec, Box.Cell, Box.InverseCell, Box.Cubic);
    double rr_dot = dot(posvec, posvec);
    double result[2] = {0.0, 0.0};
    if(rr_dot < FF.CutOffVDW)
    {
      const size_t typeB = Old.Type[j];
      const double OldscaleB = Old.scale[j];
      const double NewscaleB = newScale.x;
      const double Oldscaling = scaleA * OldscaleB;
      const double Newscaling = scaleA * NewscaleB;
      const size_t row = typeA*FF.size+typeB;
      const double FFarg[4] = {FF.epsilon[row], FF.sigma[row], FF.z[row], FF.shift[row]};
      VDW(FFarg, rr_dot, Oldscaling, result);
      tempy.x  -= result[0];
      tempdU   -= result[1];
      //NEW here for fractional molecule//
      VDW(FFarg, rr_dot, Newscaling, result);
      tempy.x  += result[0];
      tempdU   += result[1];
      if(result[0] > FF.OverlapCriteria) { /*Blockflag = true;*/ flag[0] = true; }
      if(rr_dot < 0.01)                  { /*Blockflag = true;*/ flag[0] = true; } //DistanceCheck//
    }
    if (!FF.noCharges && rr_dot < FF.CutOffCoul)
    {
      const double chargeB = Old.charge[j];
      const double OldscalingCoulombB = Old.scaleCoul[j];
      const double NewscalingCoulombB = newScale.y;
      const double r = sqrt(rr_dot);
      const double OldscalingCoul = scalingCoulombA * OldscalingCoulombB;
      const double NewscalingCoul = scalingCoulombA * NewscalingCoulombB;
      CoulombReal(chargeA, chargeB, r, OldscalingCoul, result, Box.Prefactor, Box.Alpha);
      tempy.y -= result[0]; //prefactor merged in the CoulombReal function
      CoulombReal(chargeA, chargeB, r, NewscalingCoul, result, Box.Prefactor, Box.Alpha);
      tempy.y += result[0];
    }
  }
  sdata[ij_within_block] = tempy.x; //sdata[ij_within_block].y = tempdU;
  sdata[ij_within_block + blockDim.x] = tempy.y;
  //if(total_ij == 4740) printf("ij: %lu, total_ij: %lu, VDW: %.5f, Real: %.5f\n", ij, total_ij, tempy.x, tempy.y);
  }
  __syncthreads();
  //Partial block sum//
  //if(!Blockflag)
  //{
    int shift_i=blockDim.x / 2;
    while(shift_i != 0)
    {
      if(cache_id < shift_i)
      {
        sdata[cache_id] += sdata[cache_id + shift_i];
        sdata[cache_id + blockDim.x] += sdata[cache_id + shift_i + blockDim.x];
      }
      __syncthreads();
      shift_i /= 2;
    }
    if(cache_id == 0)
    {
      //printf("BlockID: %i, VDW: %.5f, Real: %.5f\n", blockIdx.x, sdata[0], sdata[blockDim.x]);
      BlockEnergy[blockIdx.x]   = sdata[0]; 
      BlockEnergy[blockIdx.x + Total_Nblock] = sdata[blockDim.x]; //Shift it//
    }
  //}
}


__global__ void Energy_difference_LambdaChange(Boxsize Box, Atoms* System, Atoms Old, ForceField FF, double* BlockEnergy, size_t ComponentID, size_t totalAtoms, size_t chainsize, size_t HG_Nblock, size_t GG_Nblock, int3 NComps, bool* flag, double2 newScale)
{
  ///////////////////////////////////////////////////////
  //All variables passed here should be device pointers//
  ///////////////////////////////////////////////////////
  extern __shared__ double sdata[]; //shared memory for partial sum, energy + dUdlambda//
  int cache_id = threadIdx.x;
  size_t total_ij = blockIdx.x * blockDim.x + threadIdx.x;

  size_t ij_within_block = total_ij - blockIdx.x * blockDim.x;

  sdata[ij_within_block] = 0.0; sdata[ij_within_block + blockDim.x] = 0.0;
  //Initialize Blocky and BlockdUdlambda//
  BlockEnergy[blockIdx.x] = 0.0; BlockEnergy[blockIdx.x + HG_Nblock + GG_Nblock] = 0.0;
  //BlockdUdlambda[blockIdx.x] = 0.0;

  __shared__ bool Blockflag = false;

  const size_t NTotalComp = NComps.x; //Zhao's note: need to change here for multicomponent (Nguest comp > 1)
  const size_t NHostComp  = NComps.y;
  //const size_t NGuestComp = NComps.z;

  size_t ij = total_ij;
  if(blockIdx.x >= HG_Nblock)
  {
    ij -= HG_Nblock * blockDim.x; //It belongs to the Guest-Guest Interaction//
  }
  // Manually fusing/collapsing the loop //
  size_t i = ij/chainsize;
  size_t j = ij%chainsize; //+ ij/totalAtoms; // position in Old and New

  //printf("ij: %lu, i: %lu, j: %lu, trial: %lu, totalAtoms: %lu, totalthreads: %lu\n", ij,i,j,k,totalAtoms, totalthreads);
  size_t posi = i; size_t totalsize= 0;
  size_t startComp = 0; size_t endComp = 0;
  if(blockIdx.x < HG_Nblock) //Host-Guest Interaction//
  {
    startComp = 0; endComp = NHostComp;
  }
  else //Guest-Guest Interaction//
  {
    startComp = NHostComp; endComp = NTotalComp;
  }
  size_t comp = startComp;
  for(size_t ijk = startComp; ijk < endComp; ijk++)
  {
    totalsize += System[ijk].size;
    if(posi >= System[ijk].size)
    {
      comp++;
      posi -= System[ijk].size;
    }
    else
    {break;}
  }
  //Also need to check the range of the components//
  //if host-guest, then comp need to fall into the framework components//
  bool CompCheck = false;
  if(blockIdx.x < HG_Nblock)
  {
    if(comp < NHostComp) CompCheck = true;
  }
  else //Guest-Guest interaction//
  {
    if((comp >= NHostComp) && (comp < NTotalComp)) CompCheck = true;
  }


  if(posi < System[comp].size && CompCheck)
  {
    const Atoms Component=System[comp];
    const double scaleA = Component.scale[posi];
    const double chargeA = Component.charge[posi];
    const double scalingCoulombA = Component.scaleCoul[posi];
    const size_t typeA = Component.Type[posi];
    const size_t MoleculeID = Component.MolID[posi];
    double tempy = 0.0; double tempdU = 0.0;
    if(!((MoleculeID == Old.MolID[0]) &&(comp == ComponentID))) //ComponentID: Component ID for the molecule being translated
    {
      ///////////
      //  NEW  //
      ///////////
      double3 posvec = Component.pos[posi] - Old.pos[j]; 
      PBC(posvec, Box.Cell, Box.InverseCell, Box.Cubic);
      double rr_dot = dot(posvec, posvec);
      double result[2];
      if(rr_dot < FF.CutOffVDW)
      {
        const size_t typeB = Old.Type[j];
        const double OldscaleB = Old.scale[j];
        const double NewscaleB = newScale.x; //VDW part of the scale//
        const double Oldscaling = scaleA * OldscaleB;
        const double Newscaling = scaleA * NewscaleB;
        const size_t row = typeA*FF.size+typeB;
        const double FFarg[4] = {FF.epsilon[row], FF.sigma[row], FF.z[row], FF.shift[row]};
        VDW(FFarg, rr_dot, Oldscaling, result);
        tempy  -= result[0];
        tempdU -= result[1];
        VDW(FFarg, rr_dot, Newscaling, result);
        if(result[0] > FF.OverlapCriteria){ Blockflag = true; flag[0] = true;}
        tempy  += result[0];
        tempdU += result[1];
      }
      if (!FF.noCharges && rr_dot < FF.CutOffCoul)
      {
        const double chargeB = Old.charge[j];
        const double OldscalingCoulombB = Old.scaleCoul[j];
        const double NewscalingCoulombB = newScale.y; //Coulomb part of the scale//
        const double r = sqrt(rr_dot);
        const double OldscalingCoul = scalingCoulombA * OldscalingCoulombB;
        const double NewscalingCoul = scalingCoulombA * NewscalingCoulombB;
        CoulombReal(chargeA, chargeB, r, OldscalingCoul, result, Box.Prefactor, Box.Alpha);
        tempy -= result[0]; 
        CoulombReal(chargeA, chargeB, r, NewscalingCoul, result, Box.Prefactor, Box.Alpha);
        tempy += result[0];
      }
    }
    sdata[ij_within_block] = tempy; //sdata[ij_within_block].y = tempdU;
  }
  __syncthreads();
  //Partial block sum//
  if(!Blockflag)
  {
    int i=blockDim.x / 2;
    while(i != 0)
    {
      if(cache_id < i)
      {
        sdata[cache_id] += sdata[cache_id + i];
        sdata[cache_id + blockDim.x] += sdata[cache_id + i + blockDim.x];
      }
      __syncthreads();
      i /= 2;
    }
    if(cache_id == 0)
    {
      //printf("BlockID: %i, VDW: %.5f, Real: %.5f\n", blockIdx.x, sdata[0], sdata[blockDim.x]);
      BlockEnergy[blockIdx.x]   = sdata[0]; //BlockdUdlambda[blockIdx.x] = sdata[0].y;
      BlockEnergy[blockIdx.x + HG_Nblock + GG_Nblock] = sdata[blockDim.x]; //Shift it//
    }
  }
}

__global__ void Calculate_Multiple_Trial_Energy_SEPARATE_HostGuest_VDWReal(Boxsize Box, Atoms* System, Atoms NewMol, ForceField FF, double* Blocksum, size_t ComponentID, size_t totalAtoms, bool* flag, size_t totalthreads, size_t chainsize, size_t NblockForTrial, size_t HG_Nblock, int3 NComps, int2* ExcludeList)
{
  //Dividing Nblocks into Nblocks for host-guest and for guest-guest//
  //NblockForTrial = HG_Nblock + GG_Nblock;
  //Separating VDW + Real, if a trial needs 5 block (cuda blocks)
  //In the Blocksum array, it will use the first 5 doubles for VDW, the later 5 doubles for Real//
  //This is slightly confusing, don't be fooled, elements in Blocksum != cuda blocks!!!! //
  ///////////////////////////////////////////////////////
  //All variables passed here should be device pointers//
  ///////////////////////////////////////////////////////
  extern __shared__ double sdata[]; //shared memory for partial sum//
  int cache_id = threadIdx.x;
  size_t trial = blockIdx.x/NblockForTrial;
  size_t total_ij = blockIdx.x * blockDim.x + threadIdx.x;
  size_t ij = total_ij - trial * NblockForTrial * blockDim.x;

  //size_t ij_within_block = total_ij - blockIdx.x * blockDim.x;

  size_t trial_blockID = blockIdx.x - NblockForTrial * trial;

  sdata[threadIdx.x] = 0.0; sdata[threadIdx.x + blockDim.x] = 0.0;
  //Initialize Blocksum//
  size_t StoreId = blockIdx.x + trial * NblockForTrial;
  if(cache_id == 0) { Blocksum[StoreId] = 0.0; Blocksum[StoreId + NblockForTrial] = 0.0; }

  //__shared__ bool Blockflag = false;

  if(trial_blockID >= HG_Nblock)
    ij -= HG_Nblock * blockDim.x; //It belongs to the Guest-Guest Interaction//
  // Manually fusing/collapsing the loop //
  size_t i = ij/chainsize; //ij is the thread id within the trial, just divide by chainsize to get the true i (atom id)
  size_t j = trial*chainsize + ij%chainsize; // position in NewMol
  //printf("ij: %lu, i: %lu, j: %lu, trial: %lu, totalAtoms: %lu, totalthreads: %lu\n", ij,i,j,k,totalAtoms, totalthreads);
  const size_t NTotalComp = NComps.x; 
  const size_t NHostComp  = NComps.y;
  //const size_t NGuestComp = NComps.z;
 
  size_t posi = i; size_t totalsize= 0;
  size_t startComp = 0; size_t endComp = 0;

  //if posi exceeds the number of atoms in their components, stop//
  size_t NFrameworkAtoms = 0; size_t NAdsorbateAtoms = 0;
  for(size_t ijk = 0;         ijk < NHostComp;  ijk++) NFrameworkAtoms += System[ijk].size;
  for(size_t ijk = NHostComp; ijk < NTotalComp; ijk++) NAdsorbateAtoms += System[ijk].size;
  //Skip calculation if the block is for Host-Guest, and the posi is greater than or equal to N_FrameworkAtoms//
  //It is equal to 
  //if((posi >= NFrameworkAtoms) && (trial_blockID < HG_Nblock)) continue;
  if((posi < NFrameworkAtoms) || !(trial_blockID < HG_Nblock))
  {
  if(trial_blockID < HG_Nblock) //Host-Guest Interaction//
  {
    startComp = 0; endComp = NHostComp;
  }
  else //Guest-Guest Interaction//
  {
    startComp = NHostComp; endComp = NTotalComp;
  }
  size_t comp = startComp;
  //printf("%lu, %lu\n", System[0].size, System[1].size);

  for(size_t ijk = startComp; ijk < endComp; ijk++)
  {
    totalsize += System[ijk].size;
    if(posi >= System[ijk].size)
    {
      comp++;
      posi -= System[ijk].size;
    }
    else
    {break;}
  }

  //Also need to check the range of the components//
  //if host-guest, then comp need to fall into the framework components//
  bool CompCheck = false;
  if(trial_blockID < HG_Nblock)
  {
    if(comp < NHostComp) CompCheck = true;
  }
  else //Guest-Guest interaction//
  {
    if((comp >= NHostComp) && (comp < NTotalComp)) CompCheck = true;
  }

  if(CompCheck)
  if(posi < System[comp].size)
  {
    const Atoms Component=System[comp];
    const double scaleA = Component.scale[posi];
    const double chargeA = Component.charge[posi];
    const double scalingCoulombA = Component.scaleCoul[posi];
    const size_t typeA = Component.Type[posi];
    const size_t MoleculeID = System[comp].MolID[posi];

    //printf("i: %lu, posi: %lu, size1: %lu, size2: %lu\n", i, posi, System[0].size, System[1].size);

    double2 tempy = {0.0, 0.0};
    bool ConsiderThisMolecule = true;
    //Checking the first element of the ExcludeList to Ignore specific component/molecule//
    if(comp == ExcludeList[0].x && MoleculeID == ExcludeList[0].y) ConsiderThisMolecule = false;
    if((MoleculeID == NewMol.MolID[0]) &&(comp == ComponentID))    ConsiderThisMolecule = false;

    if(ConsiderThisMolecule)
    {
      double3 posvec = Component.pos[posi] - NewMol.pos[j];
      PBC(posvec, Box.Cell, Box.InverseCell, Box.Cubic);
      const double rr_dot = dot(posvec, posvec);
      if(rr_dot < FF.CutOffVDW)
      {
        double result[2] = {0.0, 0.0};
        const size_t typeB = NewMol.Type[j];
        const double scaleB = NewMol.scale[j];
        const double scaling = scaleA * scaleB;
        const size_t row = typeA*FF.size+typeB;
        //printf("typeA: %lu, typeB: %lu, FF.size: %lu, row: %lu\n", typeA, typeB, FF.size, row);
        const double FFarg[4] = {FF.epsilon[row], FF.sigma[row], FF.z[row], FF.shift[row]};
        VDW(FFarg, rr_dot, scaling, result); 
        if(result[0] > FF.OverlapCriteria){ flag[trial]=true; }
        if(rr_dot < 0.01) {flag[trial]=true; } //DistanceCheck//
        tempy.x += result[0];
        //DEBUG//
        /*
        if(CYCLE == 28981 && comp != 0 && trial == 0)
        {
          printf("GG PAIR: total_ij: %lu, ij: %lu, posi: %lu, typeA: %lu, comp: %lu, ENERGY: %.5f\n", total_ij, ij, posi, typeA, comp, result[0]);
        }
        */
      }

      //if (FF.VDWRealBias && !FF.noCharges && rr_dot < FF.CutOffCoul)
      if (!FF.noCharges && rr_dot < FF.CutOffCoul)
      {
        const double chargeB = NewMol.charge[j];
        const double scalingCoulombB = NewMol.scaleCoul[j];
        const double r = sqrt(rr_dot);
        const double scalingCoul = scalingCoulombA * scalingCoulombB;
        double resultCoul[2] = {0.0, 0.0};
        CoulombReal(chargeA, chargeB, r, scalingCoul, resultCoul, Box.Prefactor, Box.Alpha);
        tempy.y += resultCoul[0]; //prefactor merged in the CoulombReal function
      }
    }
    //if((trial_blockID >= HG_Nblock) && (tempy > 1e-10))
    //  printf("Guest-Guest, comp = %lu, trial: %lu, posi = %lu, data: %.5f\n", comp, j, posi, tempy);
    sdata[threadIdx.x] = tempy.x; sdata[threadIdx.x + blockDim.x] = tempy.y;
  }
  }
  __syncthreads();
  //Partial block sum//
  //if(!Blockflag)
  //{
    int reductionsize=blockDim.x / 2;
    while(reductionsize != 0) 
    {
      if(cache_id < reductionsize) 
      {
        sdata[cache_id] += sdata[cache_id + reductionsize];
        sdata[cache_id + blockDim.x] += sdata[cache_id + reductionsize + blockDim.x];
      }
      __syncthreads();
      reductionsize /= 2;
    }
    if(cache_id == 0) 
    {
     Blocksum[StoreId] = sdata[0];
     Blocksum[StoreId + NblockForTrial] = sdata[blockDim.x];
     //if(trial_blockID >= HG_Nblock) 
    //printf("GG, trial: %lu, BlockID: %lu, data: %.5f\n", trial, blockIdx.x, sdata[0]);
    }
  //}
}

__device__ void VDWCoulEnergy_Total(Boxsize Box, Atoms ComponentA, Atoms ComponentB, ForceField FF, bool* flag, double2& tempy, size_t posA, size_t posB, size_t compA, size_t compB, bool UseOffset)
{
  //compA may go beyond the bound, if this happens, check if compA + 1 goes beyond endcompA//
  size_t OffsetA         = 0;
  if(UseOffset) OffsetA  = ComponentA.Allocate_size / 2; //Read the positions shifted to the later half of the storage//
  const double scaleA          = ComponentA.scale[posA];
  const double chargeA         = ComponentA.charge[posA];
  const double scalingCoulombA = ComponentA.scaleCoul[posA];
  const size_t typeA           = ComponentA.Type[posA];

  const double3 PosA = ComponentA.pos[posA + OffsetA];

  size_t OffsetB         = 0;
  if(UseOffset) OffsetB  = ComponentB.Allocate_size / 2; //Read the positions shifted to the later half of the storage//
  const double scaleB          = ComponentB.scale[posB];
  const double chargeB         = ComponentB.charge[posB];
  const double scalingCoulombB = ComponentB.scaleCoul[posB];
  const size_t typeB           = ComponentB.Type[posB];
  const double3 PosB = ComponentB.pos[posB + OffsetB];

  //printf("thread: %lu, i:%lu, j:%lu, comp: %lu, posi: %lu\n", ij,i,j,comp, posi);
  double3 posvec = PosA - PosB;
  PBC(posvec, Box.Cell, Box.InverseCell, Box.Cubic);
  const double rr_dot = dot(posvec, posvec);
  if(rr_dot < FF.CutOffVDW)
  {
    double result[2] = {0.0, 0.0};
    const double scaling = scaleA * scaleB;
    const size_t row = typeA*FF.size+typeB;
    const double FFarg[4] = {FF.epsilon[row], FF.sigma[row], FF.z[row], FF.shift[row]};
    VDW(FFarg, rr_dot, scaling, result);
    if(result[0] > FF.OverlapCriteria){ flag[0]=true;}
    if(rr_dot < 0.01) { flag[0]=true; } //DistanceCheck//
    //if(result[0] > FF.OverlapCriteria || rr_dot < 0.01) printf("OVERLAP IN KERNEL!\n");
    tempy.x += result[0];
  }
  //Coulombic (REAL)//
  if (!FF.noCharges && rr_dot < FF.CutOffCoul)
  {
    const double r = sqrt(rr_dot);
    const double scalingCoul = scalingCoulombA * scalingCoulombB;
    double resultCoul[2] = {0.0, 0.0};
    CoulombReal(chargeA, chargeB, r, scalingCoul, resultCoul, Box.Prefactor, Box.Alpha);
    tempy.y += resultCoul[0]; //prefactor merged in the CoulombReal function
  }
}


__device__ void determine_comp_and_Atomindex_from_thread(Atoms* System, size_t& Atom, size_t& comp, size_t startComponent, size_t endComponent)
{
  /*
  //size_t count = Atom;
  comp = startComponent;
  for(size_t ijk = startComponent; ijk < endComponent; ijk++)
  {
    //totalsize += d_a[ijk].size;
    if(Atom >= System[ijk].size)
    {
      Atom -= System[ijk].size;
      comp ++;
    }
    else
    {
      //Atom = count;
      //comp = ijk;
      break;
    }
  }
  */
  size_t count = Atom;
  comp = startComponent;
  for(size_t ijk = startComponent; ijk < endComponent; ijk++)
  {
    //totalsize += d_a[ijk].size;
    if(count >= System[ijk].size)
    {
      count -= System[ijk].size;
      //comp ++;
    }
    else
    {
      Atom = count;
      comp = ijk;
      break;
    }
  }
  //return {(int) Atom, (int) comp};
}

__device__ void determine_comp_and_Molindex_from_thread(Atoms* System, size_t& Mol, size_t& comp, size_t startComponent, size_t endComponent)
{
  for(size_t ijk = startComponent; ijk < endComponent; ijk++)
  {
    size_t Mol_ijk = System[ijk].size / System[ijk].Molsize;
    //totalsize     += Mol_ijk;
    if(Mol >= Mol_ijk)
    {
      comp++;
      Mol -= Mol_ijk;
    }
    else 
    {
      break;
    }
  }
}

__global__ void TotalVDWCoul(Boxsize Box, Atoms* System, ForceField FF, double* Blocksum, bool* flag, size_t InteractionPerThread, bool UseOffset, int3 BLOCK, int3 NComponents, size_t NFrameworkAtoms, size_t NAdsorbateAtoms, size_t NFrameworkZero_ExtraFramework, bool ConsiderIntra)
{
  extern __shared__ double sdata[]; //shared memory for partial sum//

  size_t Nblock = BLOCK.x + BLOCK.y + BLOCK.z;
  size_t HH_Nblock = BLOCK.x;
  size_t HG_Nblock = BLOCK.y;
  //size_t GG_Nblock = BLOCK.z; //currently not used//

  int cache_id = threadIdx.x;

  size_t THREADIdx = blockIdx.x * blockDim.x + threadIdx.x; if(THREADIdx > NFrameworkZero_ExtraFramework)
 
  sdata[threadIdx.x] = 0.0;
  sdata[threadIdx.x + blockDim.x] = 0.0;

  //Initialize Blocksum//
  if(cache_id == 0)
  {Blocksum[blockIdx.x] = 0.0; Blocksum[blockIdx.x + Nblock] = 0.0; }
 
    //Aij and Bij indicate the starting positions for the objects in the pairwise interaction//
  size_t AtomA = 0; size_t AtomB = 0;
  size_t MolA  = 0; size_t MolB  = 0;
  size_t compA = 0; size_t compB = 0;
  double2 tempy = {0.0, 0.0};

  size_t HH_Threads = HH_Nblock * blockDim.x;
  size_t HG_Threads = HG_Nblock * blockDim.x;
  //size_t GG_Threads = GG_Nblock * blockDim.x; //currently not used//

    for(size_t i = 0; i != InteractionPerThread; i++)
    {
      if(blockIdx.x < HH_Nblock)
      {
        size_t InteractionIdx = THREADIdx * InteractionPerThread + i;
        if(ConsiderIntra) //All Framework atom vs. All Framework atom//
        {
          AtomA = NFrameworkAtoms - 2 - std::floor(std::sqrt(-8*InteractionIdx + 4*NFrameworkAtoms*(NFrameworkAtoms-1)-7)/2.0 - 0.5);
          AtomB = InteractionIdx + AtomA + 1 - NFrameworkAtoms*(NFrameworkAtoms-1)/2 + (NFrameworkAtoms-AtomA)*((NFrameworkAtoms-AtomA)-1)/2;
        determine_comp_and_Atomindex_from_thread(System, AtomA, compA, 0, NComponents.y);
        determine_comp_and_Atomindex_from_thread(System, AtomB, compB, 0, NComponents.y);
        }
        //If you have more than 1 framework components, then the following two conditions will always be there//
        else 
        {  
          if(InteractionIdx < NFrameworkZero_ExtraFramework) //Framework comp 0 vs. other framework atoms//
          {
            AtomA  = InteractionIdx % System[0].size; //Framework Component 0 Atom//
            AtomB  = InteractionIdx / System[0].size; //Framework Component >0 Atom//
            determine_comp_and_Atomindex_from_thread(System, AtomA, compA, 0, NComponents.y);
            determine_comp_and_Atomindex_from_thread(System, AtomB, compB, 1, NComponents.y);
          }
          else 
          {
            size_t NExtraFrameworkAtoms = NFrameworkAtoms - System[0].size;
            size_t InteractionIdx = THREADIdx * InteractionPerThread + i - NFrameworkZero_ExtraFramework;
            AtomA = NExtraFrameworkAtoms - 2 - std::floor(std::sqrt(-8*InteractionIdx + 4*NExtraFrameworkAtoms*(NExtraFrameworkAtoms-1)-7)/2.0 - 0.5);
            AtomB = InteractionIdx + AtomA + 1 - NExtraFrameworkAtoms*(NExtraFrameworkAtoms-1)/2 + (NExtraFrameworkAtoms-AtomA)*((NExtraFrameworkAtoms-AtomA)-1)/2;
            determine_comp_and_Atomindex_from_thread(System, AtomA, compA, 1, NComponents.y);
            determine_comp_and_Atomindex_from_thread(System, AtomB, compB, 1, NComponents.y);
            //DEBUG//printf("THERE IS NExtraFrameworkAtoms x NExtraFrameworkAtoms INTERACTIONS, THREADIdx: %lu, blockIdx.x: %lu, threadIdx.x: %lu\n", THREADIdx, blockIdx.x, threadIdx.x);
          }
        }
      }
      else if(blockIdx.x < (HG_Nblock + HH_Nblock)) //Host-Guest//
      {
        //This thread belongs to the Host-Guest_threads//
        size_t InteractionIdx = (THREADIdx - HH_Threads) * InteractionPerThread + i;
        AtomA  = InteractionIdx / NAdsorbateAtoms; //Framework Atom//
        AtomB  = InteractionIdx % NAdsorbateAtoms; //Adsorbate Atom//

        determine_comp_and_Atomindex_from_thread(System, AtomA, compA, 0, NComponents.y);
        determine_comp_and_Atomindex_from_thread(System, AtomB, compB, NComponents.y, NComponents.x);
      }
      else //Guest-Guest//
      {
        size_t InteractionIdx = (THREADIdx - HH_Threads - HG_Threads) * InteractionPerThread + i;
        AtomA = NAdsorbateAtoms - 2 - std::floor(std::sqrt(-8*InteractionIdx + 4*NAdsorbateAtoms*(NAdsorbateAtoms-1)-7)/2.0 - 0.5);
        AtomB = InteractionIdx + AtomA + 1 - NAdsorbateAtoms*(NAdsorbateAtoms-1)/2 + (NAdsorbateAtoms-AtomA)*((NAdsorbateAtoms-AtomA)-1)/2;

        determine_comp_and_Atomindex_from_thread(System, AtomA, compA, NComponents.y, NComponents.x);
        determine_comp_and_Atomindex_from_thread(System, AtomB, compB, NComponents.y, NComponents.x);
      }
      MolA = System[compA].MolID[AtomA];
      MolB = System[compB].MolID[AtomB];
      if(AtomA >= System[compA].size || AtomB >= System[compB].size) continue;
      if(!ConsiderIntra && (MolA == MolB) && (compA == compB)) continue;

      const Atoms ComponentA = System[compA];
      const Atoms ComponentB = System[compB];
      VDWCoulEnergy_Total(Box, ComponentA, ComponentB, FF, flag, tempy, AtomA, AtomB, compA, compB, UseOffset);
    }

  
  sdata[threadIdx.x] = tempy.x; 
  sdata[threadIdx.x + blockDim.x] = tempy.y;

  __syncthreads();
  //Partial block sum//
  int reductionsize=blockDim.x / 2;
  while(reductionsize != 0)
  {
    if(cache_id < reductionsize)
    {
      sdata[cache_id] += sdata[cache_id + reductionsize];
      sdata[cache_id + blockDim.x] += sdata[cache_id + reductionsize + blockDim.x];
    }
    __syncthreads();
    reductionsize /= 2;
  }
  if(cache_id == 0)
  {
    Blocksum[blockIdx.x] = sdata[0];
    Blocksum[blockIdx.x + Nblock] = sdata[blockDim.x];
  }
  
}

//Zhao's note: here the totMol does not consider framework atoms, ONLY Adsorbates//
MoveEnergy Total_VDW_Coulomb_Energy(Simulations& Sim, Components& SystemComponents, ForceField FF, bool UseOffset)
{
  size_t NHostAtom = 0; size_t NGuestAtom = 0;
  for(size_t i = 0; i < SystemComponents.NComponents.y; i++)
    NHostAtom += SystemComponents.Moleculesize[i] * SystemComponents.NumberOfMolecule_for_Component[i];
  for(size_t i = SystemComponents.NComponents.y; i < SystemComponents.NComponents.x; i++)
    NGuestAtom+= SystemComponents.Moleculesize[i] * SystemComponents.NumberOfMolecule_for_Component[i];

  bool ConsiderIntra = false;
  size_t HH_TotalThreads = 0; //if(ConsiderHostHost) HH_TotalThreads = NHostAtom * NHostAtom / 2;
  size_t NFrameworkZero_ExtraFramework = 0;
  if(ConsiderIntra) //Atom-Atom//
  {
    HH_TotalThreads = NHostAtom * (NHostAtom - 1) / 2;
  }
  else //if molecule-intra interactions are not considered, do component 0 x component 1-n_framework + component 1-n_framework x component 1-n_framework//
  {
    //printf("THERE IS MORE THAN 1 FRAMEWORK COMPONENTS\n");
    size_t NFrameworkComponentZeroAtoms = SystemComponents.Moleculesize[0] * SystemComponents.NumberOfMolecule_for_Component[0];
    size_t NExtraFrameworkAtoms = NHostAtom - NFrameworkComponentZeroAtoms;
    NFrameworkZero_ExtraFramework = NFrameworkComponentZeroAtoms * NExtraFrameworkAtoms;
    HH_TotalThreads = NFrameworkZero_ExtraFramework; //component 0 x component 1-n_framework
    
    HH_TotalThreads+= NExtraFrameworkAtoms * (NExtraFrameworkAtoms - 1) / 2; //component 1-n_framework x component 1-n_framework//
    //printf("Framework Comp Zero Atoms: %zu, Other Comp Atoms: %zu\n", NFrameworkComponentZeroAtoms, NExtraFrameworkAtoms);
    //printf("NFrameworkZero_ExtraFramework interactions: %zu, NExtraFrameworkAtoms * (NExtraFrameworkAtoms - 1) / 2: %zu\n", NFrameworkZero_ExtraFramework, NExtraFrameworkAtoms * (NExtraFrameworkAtoms - 1) / 2);
  }

  size_t HG_TotalThreads = NHostAtom * NGuestAtom; 
  size_t GG_TotalThreads = NGuestAtom * (NGuestAtom - 1) / 2; //GG_TotalThreads =  * NGuestAtom / 2;
  size_t InteractionPerThread = 100;

  size_t HHThreadsNeeded = HH_TotalThreads / InteractionPerThread + (HH_TotalThreads % InteractionPerThread == 0 ? 0 : 1);
  size_t HGThreadsNeeded = HG_TotalThreads / InteractionPerThread + (HG_TotalThreads % InteractionPerThread == 0 ? 0 : 1);
  size_t GGThreadsNeeded = GG_TotalThreads / InteractionPerThread + (GG_TotalThreads % InteractionPerThread == 0 ? 0 : 1);

  size_t HH_Nthread=0; size_t HH_Nblock=0; Setup_threadblock(HHThreadsNeeded, &HH_Nblock, &HH_Nthread);
  size_t HG_Nthread=0; size_t HG_Nblock=0; Setup_threadblock(HGThreadsNeeded, &HG_Nblock, &HG_Nthread);
  size_t GG_Nthread=0; size_t GG_Nblock=0; Setup_threadblock(GGThreadsNeeded, &GG_Nblock, &GG_Nthread);
  MoveEnergy E;
  
  if((HH_Nblock + HG_Nblock + GG_Nblock) == 0) return E;
  size_t Nblock = 0; size_t Nthread = 0;
  //Setup_threadblock(Host_threads + Guest_threads, &Nblock, &Nthread);
  Nthread = std::max(GG_Nthread, HG_Nthread);
  if(HH_Nthread > Nthread) Nthread = HH_Nthread;

  if(Nblock*2 > Sim.Nblocks)
  {
    printf("More blocks for block sum is needed\n");
    cudaMalloc(&Sim.Blocksum, 2*Nblock * sizeof(double));
  }

  int3 BLOCKS = {HH_Nblock, HG_Nblock, GG_Nblock};

  //Calculate the energy of the new systems//
  //Host-Guest + Guest-Guest//
  Nblock = HH_Nblock + HG_Nblock + GG_Nblock;
  //printf("Atoms: %zu %zu\n", NHostAtom, NGuestAtom);
  //printf("Interactions: %zu %zu %zu\n", HH_TotalThreads, HG_TotalThreads, GG_TotalThreads);
  //printf("Nblock %zu, blocks: %zu %zu %zu, threads needed: %zu %zu %zu, Nthread: %zu\n", Nblock, HH_Nblock, HG_Nblock, GG_Nblock, HHThreadsNeeded, HGThreadsNeeded, GGThreadsNeeded, Nthread);

  //Set Overlap Flag//
  cudaMemset(Sim.device_flag, false, sizeof(bool));
 
  TotalVDWCoul<<<Nblock, Nthread, 2 * Nthread * sizeof(double)>>>(Sim.Box, Sim.d_a, FF, Sim.Blocksum, Sim.device_flag, InteractionPerThread, UseOffset, BLOCKS, SystemComponents.NComponents, NHostAtom, NGuestAtom, NFrameworkZero_ExtraFramework, ConsiderIntra);
  checkCUDAErrorEwald("WRONG TOTAL VDW+REAL ENERGY\n");

  cudaDeviceSynchronize();  

  //printf("Total VDW + Real, Nblock = %zu, Nthread = %zu, Host: %zu, Guest: %zu, Allocated size: %zu\n", Nblock, Nthread, Host_threads, Guest_threads, Sim.Nblocks);
  //Zhao's note: consider using the flag to check for overlap here//
  //printf("Total Thread: %zu, Nblock: %zu, Nthread: %zu\n", Host_threads + Guest_threads, Nblock, Nthread);
  double BlockE[Nblock*2]; cudaMemcpy(BlockE, Sim.Blocksum, Nblock*2 * sizeof(double), cudaMemcpyDeviceToHost);

  for(size_t id = 0; id < HH_Nblock; id++) E.HHVDW += BlockE[id];
  for(size_t id = HH_Nblock; id < HH_Nblock + HG_Nblock; id++) E.HGVDW += BlockE[id];
  for(size_t id = HH_Nblock + HG_Nblock; id < Nblock; id++) E.GGVDW += BlockE[id];

  for(size_t id = Nblock; id < Nblock + HH_Nblock; id++) E.HHReal += BlockE[id];
  for(size_t id = Nblock + HH_Nblock; id < Nblock + HH_Nblock + HG_Nblock; id++) E.HGReal += BlockE[id];
  for(size_t id = Nblock + HH_Nblock + HG_Nblock; id < Nblock+Nblock; id++) E.GGReal += BlockE[id];
  //printf("GPU VDW REAL ENERGY:\n"); E.print();

  return E;
}


__global__ void REZERO_VALS(double* vals, size_t size)
{
  for(size_t i = 0; i < size; i++) vals[i] = 0.0;
}
