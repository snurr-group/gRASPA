#define MAX2(x,y) (((x)>(y))?(x):(y))                 // the maximum of two numbers
#define MAX3(x,y,z) MAX2((x),MAX2((y),(z)))

void NVTGibbsMove(std::vector<Components>& SystemComponents, Simulations*& Sims, ForceField FF);

__device__ void GetComponentMol(Atoms* System, size_t& Mol, size_t& comp, size_t Ncomp)
{
  size_t total = 0;
  for(size_t ijk = comp; ijk < Ncomp; ijk++)
  {
    size_t Mol_ijk = System[ijk].size / System[ijk].Molsize;
    total += Mol_ijk;
    if(Mol >= total)
    {
      comp++;
      Mol -= Mol_ijk;
    }
  }
}

__device__ void ScaleCopyPositions(Atoms* d_a, Boxsize Box, size_t comp, double AScale, size_t start, size_t end, double3 COM)
{
  for(size_t atom = 0; atom < d_a[comp].Molsize; atom++)
  {
    size_t from = start + atom;
    double3 xyz; 
    xyz.x = d_a[comp].x[from];
    xyz.y = d_a[comp].y[from];
    xyz.z = d_a[comp].z[from];
    double posvec[3] = {xyz.x - COM.x, xyz.y - COM.y, xyz.z - COM.z};
    PBC(posvec, Box.Cell, Box.InverseCell, Box.Cubic);
    double3 newCOM; 
    newCOM.x = COM.x * AScale;
    newCOM.y = COM.y * AScale;
    newCOM.z = COM.z * AScale;
    double3 newxyz;
    newxyz.x = newCOM.x + posvec[0]; 
    newxyz.y = newCOM.y + posvec[1];
    newxyz.z = newCOM.z + posvec[2];
    //Copy data to new location//
    //Copy to the later half of the xyz arrays in d_a//
    size_t to = end + atom;
    d_a[comp].x[to] = newxyz.x; 
    d_a[comp].y[to] = newxyz.y;
    d_a[comp].z[to] = newxyz.z;
  }
}
__global__ void ScalePositions(Atoms* d_a, Boxsize Box, double AScale, size_t NComponent, bool ScaleFramework, size_t TotalMol, bool noCharges, bool* flag)
{
  //Each thread processes an adsorbate molecule//
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i < TotalMol)
  {
    size_t Mol = i; size_t comp = 0; 
    if(!ScaleFramework) comp++;
    size_t start_comp = comp;
    //GetComponentMol(d_a, Mol, comp, NComponent);
    size_t total = 0;
    for(size_t ijk = start_comp; ijk < NComponent; ijk++)
    {
      size_t Mol_ijk = d_a[ijk].size / d_a[ijk].Molsize;
      total += Mol_ijk;
      if(Mol >= total)
      {
        comp++;
        Mol -= Mol_ijk;
      }
    }
    size_t HalfAllocateSize = d_a[comp].Allocate_size / 2;
    size_t start = Mol * d_a[comp].Molsize;
    size_t end   = start + HalfAllocateSize;
    double3 COM; //Zhao's note: one can decide whether to use the first bead position or the center of mass//
    COM.x = d_a[comp].x[start]; 
    COM.y = d_a[comp].y[start]; 
    COM.z = d_a[comp].z[start];
    ScaleCopyPositions(d_a, Box, comp, AScale, start, end, COM);
  }
  //Scale Boxsize//
  if(i == 0)
  {
    double InverseScale = 1.0 / AScale;
    for(size_t j = 0; j < 9; j++)
    { 
      Box.Cell[j] *= AScale; Box.InverseCell[j] *= InverseScale;
    }
    Box.Volume *= pow(AScale, 3);
    //Update kmax and reciprocal cutoff//
    //Only Alpha and tol1 are needed to pass here//
    //see "read_Ewald_Parameters_from_input" function in read_data.cpp//
    if(!noCharges)
    {
      double Alpha = Box.Alpha;
      double tol1  = Box.tol1;
      //Zhao's note: See InitializeEwald function in RASPA-2.0 //
      // 1.0 / Pi = 0.31830988618 //
      Box.kmax.x = std::round(0.25 + Box.Cell[0] * Alpha * tol1 * 0.31830988618);
      Box.kmax.y = std::round(0.25 + Box.Cell[4] * Alpha * tol1 * 0.31830988618);
      Box.kmax.z = std::round(0.25 + Box.Cell[8] * Alpha * tol1 * 0.31830988618);
      Box.ReciprocalCutOff = pow(1.05*static_cast<double>(MAX3(Box.kmax.x, Box.kmax.y, Box.kmax.z)), 2);
    }
    flag[0] = false;
  }
}

__global__ void CopyScaledPositions(Atoms* d_a, size_t NComponent, bool ScaleFramework, size_t TotalMol)
{
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i < TotalMol)
  {
    size_t Mol = i; size_t comp = 0;
    if(!ScaleFramework) comp++;
    size_t start_comp = comp;
    //GetComponentMol(d_a, Mol, comp, NComponent);
    size_t total = 0;
    for(size_t ijk = start_comp; ijk < NComponent; ijk++)
    {
      size_t Mol_ijk = d_a[ijk].size / d_a[ijk].Molsize;
      total += Mol_ijk;
      if(Mol >= total)
      {
        comp++;
        Mol -= Mol_ijk;
      }
    }
    size_t HalfAllocateSize = d_a[comp].Allocate_size / 2;
    size_t start = Mol * d_a[comp].Molsize;  //position of the original xyz data before the Gibbs move//
    size_t end   = start + HalfAllocateSize; //position of the scaled   xyz data after  the Gibbs move//
    //Copy the xyz data from the scaled region (later half) to the original region (first half)//
    for(size_t atom = 0; atom < d_a[comp].Molsize; atom++)
    {
      size_t from = end   + atom;  
      size_t to   = start + atom;
      d_a[comp].x[to] = d_a[comp].x[from];
      d_a[comp].y[to] = d_a[comp].y[from];
      d_a[comp].z[to] = d_a[comp].z[from];
    }
  }
}

__global__ void Revert_Boxsize(Boxsize Box, double AScale, bool noCharges)
{
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i == 0)
  {
    double InverseScale = 1.0 / AScale;
    for(size_t j = 0; j < 9; j++)
    {
      Box.Cell[j] *= InverseScale; Box.InverseCell[j] *= AScale;
    }
    Box.Volume *= pow(InverseScale, 3);
    if(!noCharges)
    {
      double Alpha = Box.Alpha;
      double tol1  = Box.tol1;
      //Zhao's note: See InitializeEwald function in RASPA-2.0 //
      // 1.0 / Pi = 0.31830988618 //
      Box.kmax.x = std::round(0.25 + Box.Cell[0] * Alpha * tol1 * 0.31830988618);
      Box.kmax.y = std::round(0.25 + Box.Cell[4] * Alpha * tol1 * 0.31830988618);
      Box.kmax.z = std::round(0.25 + Box.Cell[8] * Alpha * tol1 * 0.31830988618);
      Box.ReciprocalCutOff = pow(1.05*static_cast<double>(MAX3(Box.kmax.x, Box.kmax.y, Box.kmax.z)), 2);
    }
  }
}

//Get the total number of molecules (excluding framework and fractional molecule) in a box//
static inline double Get_TotalNumberOfMolecule_In_Box(Components& SystemComponents)
{
  double NMol = static_cast<double>(SystemComponents.TotalNumberOfMolecules - SystemComponents.NumberOfFrameworks);
  //Minus the fractional molecule//
  for(size_t comp = 0; comp < SystemComponents.Total_Components; comp++)
  {
    if(SystemComponents.hasfractionalMolecule[comp])
    {
      NMol-=1.0;
    }
  }
  return NMol;
}

void NVTGibbsMove(std::vector<Components>& SystemComponents, Simulations*& Sims, ForceField FF, std::vector<SystemEnergies>& Energy, Gibbs& GibbsStatistics)
{
  size_t NBox = SystemComponents.size();
  size_t SelectedBox = 0;
  size_t OtherBox    = 1;
  if(get_random_from_zero_to_one() > 0.5)
  {
    SelectedBox = 1;
    OtherBox    = 0;
  }
  //printf("Performing NVT Gibbs Move! on Box[%zu]\n", SelectedBox);
  GibbsStatistics.GibbsBoxStats.x += 1.0;
  double TotalV = Sims[SelectedBox].Box.Volume + Sims[OtherBox].Box.Volume;

  double OldVA = Sims[SelectedBox].Box.Volume; double OldVB = Sims[OtherBox].Box.Volume;

  double MaxGibbsVolumeChange = GibbsStatistics.MaxGibbsBoxChange;
  double expdV = std::exp(std::log(Sims[SelectedBox].Box.Volume / Sims[OtherBox].Box.Volume) + MaxGibbsVolumeChange * 2.0 * (get_random_from_zero_to_one() - 0.5));
  double newVA = expdV * TotalV / (1.0 + expdV);
  double newVB = TotalV - newVA;
  //printf("TotalV: %.5f, VA/VB: %.5f/%.5f, expdV: %.5f, newVA/newVB: %.5f/%.5f\n", TotalV, Sims[SelectedBox].Box.Volume, Sims[OtherBox].Box.Volume, expdV, newVA, newVB);
  double ScaleAB[2] = {0.0, 0.0};
  ScaleAB[SelectedBox] = std::cbrt(newVA / Sims[SelectedBox].Box.Volume);
  ScaleAB[OtherBox]    = std::cbrt(newVB / Sims[OtherBox].Box.Volume);
  bool ScaleFramework = false;
  size_t Nblock = 0; size_t Nthread = 0; size_t totMol = 0;
  //Check if Allocate_size is greater than or equal to twice of the current size//
  //Then scale the boxes, calculate the new energy//
  double VDWRealEnergy[2] = {0.0, 0.0};
  double EwaldEnergy[2]   = {0.0, 0.0};

  bool Overlap = false;

  if(!ScaleFramework)
  {
    for(size_t sim = 0; sim < NBox; sim++)
    { 
      if(Overlap) continue;
      totMol = SystemComponents[sim].TotalNumberOfMolecules - SystemComponents[sim].NumberOfFrameworks;
      for(size_t comp = 1; comp < SystemComponents[sim].Total_Components; comp++)
      {
        size_t TotSize = SystemComponents[sim].Moleculesize[comp] * SystemComponents[sim].NumberOfMolecule_for_Component[comp];
        if(TotSize * 2 > SystemComponents[sim].Allocate_size[comp]) throw std::runtime_error("Allocate More space for adsorbates on the GPU!!!");
        //printf("Box[%zu], TotalMolecule for component [%zu] is %zu\n", sim, comp, totMol);
      }
      Setup_threadblock(totMol, &Nblock, &Nthread);
      ScalePositions<<<Nblock, Nthread>>>(Sims[sim].d_a, Sims[sim].Box, ScaleAB[sim], SystemComponents[sim].Total_Components, ScaleFramework, totMol, FF.noCharges, Sims[sim].device_flag);

      //////////////////////
      // TOTAL VDW + REAL //
      ////////////////////// 
      size_t Host_threads  = 0; //In this move, there should be NO Framework Atom (Framework (component 0) should be an empty box)//
      size_t  Guest_threads = totMol * (totMol - 1) / 2;
      bool    ConsiderHostHost         = false;
      bool    UseOffset                = true;
      size_t  NFrameworkAtomsPerThread = 1;
      VDWRealEnergy[sim] = Total_VDW_Coulomb_Energy(Sims[sim], FF, totMol, Host_threads, Guest_threads, NFrameworkAtomsPerThread, ConsiderHostHost, UseOffset);
      //Check for Overlaps//
      cudaMemcpy(Sims[sim].flag, Sims[sim].device_flag, sizeof(bool), cudaMemcpyDeviceToHost); 
      if(Sims[sim].flag[0]) 
      { 
        Overlap = true;
        printf("There is OVERLAP IN GIBBS VOLUME MOVE in Box[%zu]!\n", sim);
      }
      /////////////////////////////
      // TOTAL EWALD CALCULATION //
      /////////////////////////////
      //Calculate new alpha and kmax// 
      if(!FF.noCharges)
      {
        bool UseOffSet = true; //Use the second half of the allocated space for xyz
        EwaldEnergy[sim] = Ewald_TotalEnergy(Sims[sim], SystemComponents[sim], UseOffSet);
        //printf("sim: %zu, EwaldE: %.5f\n", sim, EwaldEnergy[sim]);
      }
    }
  }
  //If the Gibbs Volume Move is Accepted, for the test, assume it is always accepted//
  bool Accept = false;
  if(!Overlap)
  {
    double NMolA= Get_TotalNumberOfMolecule_In_Box(SystemComponents[SelectedBox]);
    double NMolB= Get_TotalNumberOfMolecule_In_Box(SystemComponents[OtherBox]);
    double CurrentEA   = Energy[SelectedBox].running_energy + Energy[SelectedBox].InitialEnergy;
    double CurrentEB   = Energy[OtherBox].running_energy    + Energy[OtherBox].InitialEnergy;
    double newEA       = VDWRealEnergy[SelectedBox]         + EwaldEnergy[SelectedBox];
    double newEB       = VDWRealEnergy[OtherBox]            + EwaldEnergy[OtherBox];
    double DeltaEA     = newEA - CurrentEA;
    double DeltaEB     = newEB - CurrentEB;
    double VolumeRatioA= Sims[SelectedBox].Box.Volume / OldVA;
    double VolumeRatioB= Sims[OtherBox].Box.Volume    / OldVB;

    //This assumes that the two boxes share the same temperature, it might not be true// 
    double Pacc = std::exp(-SystemComponents[SelectedBox].Beta*((DeltaEA+DeltaEB)+((NMolA+1.0)*std::log(VolumeRatioA))+((NMolB+1.0)*std::log(VolumeRatioB))));
    //printf("Gibbs Box Move, Pacc: %.5f\n", Pacc);
    if(get_random_from_zero_to_one() < Pacc) Accept = true;
  }
 
  if(Accept)
  {
    GibbsStatistics.GibbsBoxStats.y += 1.0;
    //Update Energy and positions//
    for(size_t sim = 0; sim < NBox; sim++)
    {
      totMol = SystemComponents[sim].TotalNumberOfMolecules - SystemComponents[sim].NumberOfFrameworks;
      for(size_t comp = 1; comp < SystemComponents[sim].Total_Components; comp++)
      {
        size_t TotSize = SystemComponents[sim].Moleculesize[comp] * SystemComponents[sim].NumberOfMolecule_for_Component[comp];
        if(TotSize * 2 > SystemComponents[sim].Allocate_size[comp]) throw std::runtime_error("Allocate More space for adsorbates on the GPU!!!");
      }
      double CurrentEnergy = Energy[sim].running_energy + Energy[sim].InitialEnergy;
      //Zhao's note: here the energies are only considering VDW + Real//
      double NewVDWRealEnergy = VDWRealEnergy[sim];
      double NewEwaldEnergy   = EwaldEnergy[sim];
      double NewEnergy        = NewVDWRealEnergy + NewEwaldEnergy;
      double DeltaE           = NewEnergy - CurrentEnergy;
      //SystemComponents[sim].deltaVDWReal += DeltaE;
      Energy[sim].running_energy += DeltaE;
      //printf("OldEnergy: %.5f, NewEnergy: %.5f, DeltaE: %.5f\n", CurrentEnergy, NewEnergy, DeltaE);
      Setup_threadblock(totMol, &Nblock, &Nthread);
      //Copy xyz data from new to old, also update box lengths//
      CopyScaledPositions<<<Nblock, Nthread>>>(Sims[sim].d_a, SystemComponents[sim].Total_Components, ScaleFramework, totMol);
      //Update Eik if accepted from totalEik to StoredEik//
      Update_Ewald_Vector(Sims[sim].Box, false, SystemComponents[sim]);
    }
  }
  else
  {
    for(size_t sim = 0; sim < NBox; sim++)
    {
      //Revert the Boxsize, if Charges, update kmax and Reciprocal Cutoff//
      Revert_Boxsize<<<1,1>>>(Sims[sim].Box, ScaleAB[sim], FF.noCharges);
    }
  }
}

static inline void Update_Max_GibbsVolume(Gibbs& GibbsStatistics)
{
  if(GibbsStatistics.GibbsBoxStats.x > 0)
  {
    double ratio = static_cast<double>(GibbsStatistics.GibbsBoxStats.x) / static_cast<double>(GibbsStatistics.GibbsBoxStats.y);
    double vandr = 1.05;
    if(ratio < 0.5) vandr = 0.95;
    GibbsStatistics.MaxGibbsBoxChange*=vandr;
    if(GibbsStatistics.MaxGibbsBoxChange<0.0005)
       GibbsStatistics.MaxGibbsBoxChange=0.0005;
    if(GibbsStatistics.MaxGibbsBoxChange>0.5)
       GibbsStatistics.MaxGibbsBoxChange=0.5;
  } 
}
