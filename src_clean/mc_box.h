#define MAX2(x,y) (((x)>(y))?(x):(y))                 // the maximum of two numbers
#define MAX3(x,y,z) MAX2((x),MAX2((y),(z)))

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

__device__ void ScaleCopyPositions(Atoms* d_a, Boxsize& Box, size_t comp, double AScale, size_t start, size_t end, double3 COM)
{
  for(size_t atom = 0; atom < d_a[comp].Molsize; atom++)
  {
    size_t from = start + atom;
    double3 xyz; 
    xyz = d_a[comp].pos[from];
    double3 posvec = xyz - COM;
    PBC(posvec, Box.Cell, Box.InverseCell, Box.Cubic);
    double3 newCOM; 
    newCOM.x = COM.x * AScale;
    newCOM.y = COM.y * AScale;
    newCOM.z = COM.z * AScale;
    double3 newxyz;
    newxyz = newCOM + posvec;
    //Copy data to new location//
    //Copy to the later half of the xyz arrays in d_a//
    size_t to = end + atom;
    d_a[comp].pos[to] = newxyz; 
  }
}
__global__ void ScalePositions(Atoms* d_a, Boxsize& Box, double AScale, size_t NComponent, bool ScaleFramework, size_t TotalMol, bool noCharges, bool* flag, double newV)
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
    COM = d_a[comp].pos[start]; 
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
    Box.Volume = newV;
    //printf("NewCell[0] = %.5f, scale: %.5f, Box.Volume: %.5f\n", Box.Cell[0], AScale, Box.Volume);
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
      d_a[comp].pos[to] = d_a[comp].pos[from];
    }
  }
}

__global__ void Revert_Boxsize(Boxsize& Box, double AScale, bool noCharges, double OldV)
{
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i == 0)
  {
    double InverseScale = 1.0 / AScale;
    for(size_t j = 0; j < 9; j++)
    {
      Box.Cell[j] *= InverseScale; Box.InverseCell[j] *= AScale;
    }
    //Box.Volume *= pow(InverseScale, 3);
    Box.Volume = OldV;
    //printf("REVERT BACK: Cell[0] = %.5f, scale: %.5f, Box.Volume: %.5f\n", Box.Cell[0], AScale, Box.Volume);
    if(!noCharges)
    {
      /*
      if(Box.UseLAMMPSEwald)
      {
        double lx = Box.Cell[0];
        double ly = Box.Cell[4];
        double lz = Box.Cell[8];
        double xy = 0.0; //Box.Cell[3];
        double xz = 0.0; //Box.Cell[6];
        double yz = 0.0; //Box.Cell[7];

        double M_PI = 3.14159265358979323846;
        double ux = 2*M_PI/lx;
        double uy = 2*M_PI*(-xy)/lx/ly;
        double uz = 2*M_PI*(xy*yz - ly*xz)/lx/ly/lz;
        double vy = 2*M_PI/ly;
        double vz = 2*M_PI*(-yz)/ly/lz;
        double wz = 2*M_PI/lz;
        const double kvecx = Box.kmax.x*ux;
        const double kvecy = Box.kmax.x*uy + Box.kmax.y*vy;
        const double kvecz = Box.kmax.x*uz + Box.kmax.y*vz + Box.kmax.z*wz;
        Box.ReciprocalCutOff = MAX3(kvecx*kvecx, kvecy*kvecy, kvecz*kvecz) * 1.00001;
      }
      else
      {
      */
        double Alpha = Box.Alpha;
        double tol1  = Box.tol1;
        //Zhao's note: See InitializeEwald function in RASPA-2.0 //
        // 1.0 / Pi = 0.31830988618 //
        Box.kmax.x = std::round(0.25 + Box.Cell[0] * Alpha * tol1 * 0.31830988618);
        Box.kmax.y = std::round(0.25 + Box.Cell[4] * Alpha * tol1 * 0.31830988618);
        Box.kmax.z = std::round(0.25 + Box.Cell[8] * Alpha * tol1 * 0.31830988618);
        Box.ReciprocalCutOff = pow(1.05*static_cast<double>(MAX3(Box.kmax.x, Box.kmax.y, Box.kmax.z)), 2);
      //}
    }
  }
}

//Get the total number of molecules (excluding framework and fractional molecule) in a box//
static inline double Get_TotalNumberOfMolecule_In_Box(Components& SystemComponents)
{
  double NMol = static_cast<double>(SystemComponents.TotalNumberOfMolecules - SystemComponents.NumberOfFrameworks);
  //Minus the fractional molecule//
  for(size_t comp = 0; comp < SystemComponents.NComponents.x; comp++)
  {
    if(SystemComponents.hasfractionalMolecule[comp])
    {
      NMol-=1.0;
    }
  }
  return NMol;
}

void VolumeMove(Components& SystemComponents, Simulations& Sim, ForceField FF)
{
  SystemComponents.VolumeMoveAttempts += 1;
  double OldV = Sim.Box.Volume;
  double MaxVolumeChange = SystemComponents.VolumeMoveMaxChange;
  double newV = std::exp(std::log(Sim.Box.Volume) + MaxVolumeChange * 2.0 * (Get_Uniform_Random() - 0.5));
  double Scale = std::cbrt(newV / OldV);
  bool ScaleFirstComponentFramework = false;
  size_t Nblock = 0; size_t Nthread = 0; size_t totMol = 0;
  //Check if Allocate_size is greater than or equal to twice of the current size//
  //Then scale the boxes, calculate the new energy//

  double LengthSQ = pow(std::cbrt(newV), 2);
  if(LengthSQ < 4.0*FF.CutOffVDW || LengthSQ < 4.0*FF.CutOffCoul)
  {
    printf("Cycle: %zu, Box LengthSQ %.5f (%.5f) < Cutoff\n", SystemComponents.CURRENTCYCLE, LengthSQ, std::sqrt(LengthSQ));
  }

  MoveEnergy CurrentE;
  MoveEnergy NewE;
  MoveEnergy DeltaE;

  bool Overlap = false;
  totMol = SystemComponents.TotalNumberOfMolecules - SystemComponents.NumberOfFrameworks;
  size_t startComponent = 0; if(!ScaleFirstComponentFramework) startComponent = 1;
  for(size_t comp = startComponent; comp < SystemComponents.NComponents.x; comp++)
  {
    size_t TotSize = SystemComponents.Moleculesize[comp] * SystemComponents.NumberOfMolecule_for_Component[comp];
    if(TotSize * 2 > SystemComponents.Allocate_size[comp]) throw std::runtime_error("Allocate More space for adsorbates on the GPU!!!");
  }
  Sim.Box.Volume = newV;
  Setup_threadblock(totMol, &Nblock, &Nthread);
  ScalePositions<<<Nblock, Nthread>>>(Sim.d_a, Sim.Box, Scale, SystemComponents.NComponents.x, ScaleFirstComponentFramework, totMol, FF.noCharges, Sim.device_flag, newV);
  checkCUDAError("VolumeMove: Error in ScalePositions\n");

  //////////////////////
  // TOTAL VDW + REAL //
  /////////////////////
  bool    UseOffset = true;
  NewE = Total_VDW_Coulomb_Energy(Sim, SystemComponents, FF, UseOffset);
  //Check for Overlaps//
  SystemComponents.flag[0] = false;
  cudaMemcpy(SystemComponents.flag, Sim.device_flag, sizeof(bool), cudaMemcpyDeviceToHost);
  if(SystemComponents.flag[0]) 
  { 
    //printf("NPT VOLUME MOVE: Cycle %zu, Overlapped!\n", SystemComponents.CURRENTCYCLE);
    Overlap = true;
  }
  /////////////////////////////
  // TOTAL EWALD CALCULATION //
  /////////////////////////////
  //Calculate new alpha and kmax// 
  if(!FF.noCharges)
  {
    bool UseOffSet = true; //Use the second half of the allocated space for xyz
    NewE += Ewald_TotalEnergy(Sim, SystemComponents, UseOffSet);
  }
  //Tail Correction//
  NewE.TailE = TotalTailCorrection(SystemComponents, FF.size, Sim.Box.Volume);
  
  //If the Gibbs Volume Move is Accepted, for the test, assume it is always accepted//
  bool Accept = false;
  double PAcc = 0.0;
  double RN = 0.0;
  if(!Overlap)
  {
    double NMol = Get_TotalNumberOfMolecule_In_Box(SystemComponents);

    CurrentE    = SystemComponents.CreateMol_Energy + SystemComponents.deltaE;
    DeltaE      = NewE - CurrentE;
    double VolumeRatio = Sim.Box.Volume / OldV;
    double VolumeDiff  = Sim.Box.Volume - OldV;
    double Beta        = SystemComponents.Beta;
 
    //This assumes that the two boxes share the same temperature, it might not be true// 
    PAcc = std::exp((NMol+1.0)*std::log(VolumeRatio) - (DeltaE.total() + SystemComponents.Pressure*VolumeDiff) * Beta);
    RN = Get_Uniform_Random();
    if(RN < PAcc) Accept = true;
    //printf("NMol: %.5f, VolumeRatio: %.5f, DeltaE.total(): %.5f, SystemComponents.Pressure: %.5f, VolumeDiff: %.5f, Beta: %.5f\n", NMol, VolumeRatio, DeltaE.total(), SystemComponents.Pressure, VolumeDiff, Beta);
  }
 
  if(Accept)
  {
    SystemComponents.VolumeMoveAccepted += 1.0;
    //Update Energy and positions//
    size_t startComponent = 0; if(!ScaleFirstComponentFramework) startComponent = 1;
    for(size_t comp = startComponent; comp < SystemComponents.NComponents.x; comp++)
    {
      size_t TotSize = SystemComponents.Moleculesize[comp] * SystemComponents.NumberOfMolecule_for_Component[comp];
      if(TotSize * 2 > SystemComponents.Allocate_size[comp]) throw std::runtime_error("Allocate More space for adsorbates on the GPU!!!");
    }
    //Zhao's note: here the energies are only considering VDW + Real//
    Setup_threadblock(totMol, &Nblock, &Nthread);
    //Copy xyz data from new to old, also update box lengths//
    totMol = SystemComponents.TotalNumberOfMolecules - SystemComponents.NumberOfFrameworks;
    CopyScaledPositions<<<Nblock, Nthread>>>(Sim.d_a, SystemComponents.NComponents.x, ScaleFirstComponentFramework, totMol);
    checkCUDAError("Volume Move: Error in CopyScaledPositions\n");
    SystemComponents.deltaE += DeltaE;
    //Update Eik if accepted from tempEik to StoredEik, BUG (adsorbate/framework species all needs to be updated)!!!//
    if(!FF.noCharges)
    {
      std::swap(Sim.Box.tempEik,          Sim.Box.AdsorbateEik);
      std::swap(Sim.Box.tempFrameworkEik, Sim.Box.FrameworkEik);
      SystemComponents.EikAllocateSize = SystemComponents.tempEikAllocateSize;
    }
  }
  else
  {
    Sim.Box.Volume = OldV;
    Revert_Boxsize<<<1,1>>>(Sim.Box, Scale, FF.noCharges, OldV);
    checkCUDAError("Volume Move: Error in Revert_Boxsize\n");
    //DeltaE = MoveEnergy();
  }
  /*
  if(SystemComponents.CURRENTCYCLE > 8000)
  { 
    printf("DeltaE.total: %.5f\n", DeltaE.total());
    printf("CYCLE: %zu, NEWVolume: %.5f (box 0: %.5f (Old: %.5f), Accept?: %s, Accepted: %zu, RN: %.5f, PAcc: %.5f\n", SystemComponents.CURRENTCYCLE, newV, Sim.Box.Volume, OldV, Accept ? "Accepted" : "Rejected", SystemComponents.VolumeMoveAccepted, RN, PAcc);
    if(SystemComponents.CURRENTCYCLE == 9906) DeltaE.print();
  }
  */
}


void NVTGibbsMove(std::vector<Components>& SystemComponents, Simulations*& Sims, ForceField FF, Gibbs& GibbsStatistics)
{
  size_t NBox = SystemComponents.size();
  size_t SelectedBox = 0;
  size_t OtherBox    = 1;
  if(Get_Uniform_Random() > 0.5)
  {
    SelectedBox = 1;
    OtherBox    = 0;
  }
  GibbsStatistics.GibbsBoxStats.x += 1;
  double TotalV = Sims[SelectedBox].Box.Volume + Sims[OtherBox].Box.Volume;

  double OldVA = Sims[SelectedBox].Box.Volume; double OldVB = Sims[OtherBox].Box.Volume;

  double MaxGibbsVolumeChange = GibbsStatistics.MaxGibbsBoxChange;
  double expdV = std::exp(std::log(Sims[SelectedBox].Box.Volume / Sims[OtherBox].Box.Volume) + MaxGibbsVolumeChange * 2.0 * (Get_Uniform_Random() - 0.5));
  double newVA = expdV * TotalV / (1.0 + expdV);
  double newVB = TotalV - newVA;
  double ScaleAB[2] = {0.0, 0.0};
  ScaleAB[SelectedBox] = std::cbrt(newVA / OldVA);
  ScaleAB[OtherBox]    = std::cbrt(newVB / OldVB);
  double LengthASQ = pow(std::cbrt(newVA), 2);
  double LengthBSQ = pow(std::cbrt(newVB), 2);
  
  if(LengthASQ < 4.0*FF.CutOffVDW || LengthASQ < 4.0*FF.CutOffCoul) 
  {
    printf("Cycle: %zu, Box LengthASQ %.5f (%.5f) < Cutoff\n", SystemComponents[0].CURRENTCYCLE, LengthASQ, std::sqrt(LengthASQ));
    return;
  }
  if(LengthBSQ < 4.0*FF.CutOffVDW || LengthBSQ < 4.0*FF.CutOffCoul)
  {
    printf("Cycle: %zu, Box LengthBSQ %.5f (%.5f) < Cutoff\n", SystemComponents[0].CURRENTCYCLE, LengthBSQ, std::sqrt(LengthBSQ));
    return;
  }
  bool ScaleFramework = false;
  size_t Nblock = 0; size_t Nthread = 0; size_t totMol = 0;
  //Check if Allocate_size is greater than or equal to twice of the current size//
  //Then scale the boxes, calculate the new energy//

  double newV[2] = {0.0, 0.0};
  double OldV[2] = {0.0, 0.0};
  newV[SelectedBox] = newVA;
  newV[OtherBox]    = newVB;

  OldV[SelectedBox] = OldVA;
  OldV[OtherBox]    = OldVB;

  MoveEnergy CurrentE[2]; 
  MoveEnergy NewE[2];
  MoveEnergy DeltaE[2];

  bool Overlap = false;

  if(!ScaleFramework)
  {
    for(size_t sim = 0; sim < NBox; sim++)
    { 
      totMol = SystemComponents[sim].TotalNumberOfMolecules - SystemComponents[sim].NumberOfFrameworks;
      for(size_t comp = 1; comp < SystemComponents[sim].NComponents.x; comp++)
      {
        size_t TotSize = SystemComponents[sim].Moleculesize[comp] * SystemComponents[sim].NumberOfMolecule_for_Component[comp];
        if(TotSize * 2 > SystemComponents[sim].Allocate_size[comp]) throw std::runtime_error("Allocate More space for adsorbates on the GPU!!!");
      }
      Setup_threadblock(totMol, &Nblock, &Nthread);

      Sims[sim].Box.Volume = newV[sim];
      //if(Get_TotalNumberOfMolecule_In_Box(SystemComponents[sim]) > 1e-10)
      ScalePositions<<<Nblock, Nthread>>>(Sims[sim].d_a, Sims[sim].Box, ScaleAB[sim], SystemComponents[sim].NComponents.x, ScaleFramework, totMol, FF.noCharges, Sims[sim].device_flag, newV[sim]);
      checkCUDAError("error Scaling Positions in NVTGibbsVolumeMove\n");

      if(Overlap) continue; //Skip after scaling the positions//
      //////////////////////
      // TOTAL VDW + REAL //
      ////////////////////// 
      bool UseOffset = true;
      NewE[sim] = Total_VDW_Coulomb_Energy(Sims[sim], SystemComponents[sim], FF, UseOffset);
      //Check for Overlaps//
      SystemComponents[sim].flag[0] = false;
      cudaMemcpy(SystemComponents[sim].flag, Sims[sim].device_flag, sizeof(bool), cudaMemcpyDeviceToHost);
      if(SystemComponents[sim].flag[0]) 
      {
        Overlap = true;
      }
      /////////////////////////////
      // TOTAL EWALD CALCULATION //
      /////////////////////////////
      //Calculate new alpha and kmax// 
      if(!FF.noCharges)
      {
        bool UseOffSet = true; //Use the second half of the allocated space for xyz
        NewE[sim] += Ewald_TotalEnergy(Sims[sim], SystemComponents[sim], UseOffSet);
      }
      NewE[sim].TailE = TotalTailCorrection(SystemComponents[sim], FF.size, Sims[sim].Box.Volume);
    }
  }
  //If the Gibbs Volume Move is Accepted, for the test, assume it is always accepted//
  bool Accept = false;
  if(!Overlap)
  {
    double NMolA= Get_TotalNumberOfMolecule_In_Box(SystemComponents[SelectedBox]);
    double NMolB= Get_TotalNumberOfMolecule_In_Box(SystemComponents[OtherBox]);
    CurrentE[SelectedBox] = SystemComponents[SelectedBox].CreateMol_Energy + SystemComponents[SelectedBox].deltaE;
    CurrentE[OtherBox]    = SystemComponents[OtherBox].CreateMol_Energy + SystemComponents[OtherBox].deltaE;

    DeltaE[SelectedBox] = NewE[SelectedBox] - CurrentE[SelectedBox];
    DeltaE[OtherBox]    = NewE[OtherBox]    - CurrentE[OtherBox];
    double VolumeRatioA= Sims[SelectedBox].Box.Volume / OldVA;
    double VolumeRatioB= Sims[OtherBox].Box.Volume    / OldVB;

    //This assumes that the two boxes share the same temperature, it might not be true// 
    double Pacc = std::exp(-SystemComponents[SelectedBox].Beta*(DeltaE[0].total()+DeltaE[1].total())+((NMolA+1.0)*std::log(VolumeRatioA))+((NMolB+1.0)*std::log(VolumeRatioB)));
    if(Get_Uniform_Random() < Pacc) Accept = true;
  }
 
  if(Accept)
  {
    GibbsStatistics.GibbsBoxStats.y += 1;
    //Update Energy and positions//
    
    for(size_t sim = 0; sim < NBox; sim++)
    {
      totMol = SystemComponents[sim].TotalNumberOfMolecules - SystemComponents[sim].NumberOfFrameworks;
      for(size_t comp = 1; comp < SystemComponents[sim].NComponents.x; comp++)
      {
        size_t TotSize = SystemComponents[sim].Moleculesize[comp] * SystemComponents[sim].NumberOfMolecule_for_Component[comp];
        if(TotSize * 2 > SystemComponents[sim].Allocate_size[comp]) throw std::runtime_error("Allocate More space for adsorbates on the GPU!!!");
      }
      //Zhao's note: here the energies are only considering VDW + Real//
      SystemComponents[sim].deltaE += DeltaE[sim];
      Setup_threadblock(totMol, &Nblock, &Nthread);
      //Copy xyz data from new to old, also update box lengths//
      if(Get_TotalNumberOfMolecule_In_Box(SystemComponents[sim]) > 1e-10)
      CopyScaledPositions<<<Nblock, Nthread>>>(Sims[sim].d_a, SystemComponents[sim].NComponents.x, ScaleFramework, totMol);
      checkCUDAError("NVTGibbs: Error in CopyScaledPositions\n");
      //Update Eik if accepted from tempEik to StoredEik, BUG (adsorbate/framework species all needs to be updated)!!!//
      if(!FF.noCharges)
      {
        std::swap(Sims[sim].Box.tempEik,          Sims[sim].Box.AdsorbateEik);
        std::swap(Sims[sim].Box.tempFrameworkEik, Sims[sim].Box.FrameworkEik);
        SystemComponents[sim].EikAllocateSize = SystemComponents[sim].tempEikAllocateSize;
      }
    }
  }
  else
  {
    for(size_t sim = 0; sim < NBox; sim++)
    {
      //Revert the Boxsize, if Charges, update kmax and Reciprocal Cutoff//
      Sims[sim].Box.Volume = OldV[sim];
      Revert_Boxsize<<<1,1>>>(Sims[sim].Box, ScaleAB[sim], FF.noCharges, OldV[sim]);
      checkCUDAError("NVTGibbs: Error in Revert_Boxsize\n");
    }
  }
  //DEBUG//
  double NEWVolume = 0.0;
  for(size_t sim = 0; sim < NBox; sim++)
  {
    //DEBUG//
    NEWVolume += Sims[sim].Box.Volume;
  }
  //printf("NEWVolume: %.5f (box 0: %.5f (Old: %.5f / New: %.5f), box 1: %.5f (Old: %.5f / New: %.5f)), InitialTotal: %.5f, Accept?: %s\n", NEWVolume, Sims[0].Box.Volume, OldV[0], newV[0], Sims[1].Box.Volume, OldV[1], newV[1], GibbsStatistics.TotalVolume, Accept ? "Accepted" : "Rejected");
  if(std::abs(NEWVolume - GibbsStatistics.TotalVolume) > 0.1) throw std::runtime_error("VOLUME DRIFTED!!!\n");
}

static inline void Update_Max_GibbsVolume(Gibbs& GibbsStatistics)
{
  if(GibbsStatistics.GibbsBoxStats.x > 0)
  {
    double ratio = static_cast<double>(GibbsStatistics.GibbsBoxStats.x) / static_cast<double>(GibbsStatistics.GibbsBoxStats.y);
    double vandr = ratio/GibbsStatistics.TargetAccRatioVolumeChange;
    if(vandr > 1.5) vandr = 1.5;
    else if(ratio < 0.5) vandr = 0.5;
    GibbsStatistics.MaxGibbsBoxChange*=vandr;
    if(GibbsStatistics.MaxGibbsBoxChange<0.0005)
       GibbsStatistics.MaxGibbsBoxChange=0.0005;
    if(GibbsStatistics.MaxGibbsBoxChange>0.5)
       GibbsStatistics.MaxGibbsBoxChange=0.5;
  }
  GibbsStatistics.TotalGibbsBoxStats.x += GibbsStatistics.GibbsBoxStats.x;
  GibbsStatistics.TotalGibbsBoxStats.y += GibbsStatistics.GibbsBoxStats.y;
  GibbsStatistics.GibbsBoxStats.x = 0; GibbsStatistics.GibbsBoxStats.y = 0;
}
