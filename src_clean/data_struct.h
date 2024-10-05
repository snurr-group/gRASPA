#include <filesystem>

#include <stdio.h>
#include <vector>
#include <string>
#include <complex>
#include <cstdlib>
#include <random>
#include <optional>

//###PATCH_LCLIN_DATA_STRUCT_H###//

//###PATCH_ALLEGRO_DATA_STRUCT_H###//
#define BLOCKSIZE 1024
#define DEFAULTTHREAD 128
double Get_Uniform_Random();

enum MoveTypes {TRANSLATION = 0, ROTATION, SINGLE_INSERTION, SINGLE_DELETION, SPECIAL_ROTATION, INSERTION, DELETION, REINSERTION, CBCF_LAMBDACHANGE, CBCF_INSERTION, CBCF_DELETION, IDENTITY_SWAP, WIDOM};

enum CBMC_Types {CBMC_INSERTION = 0, CBMC_DELETION, REINSERTION_INSERTION, REINSERTION_RETRACE, IDENTITY_SWAP_NEW, IDENTITY_SWAP_OLD};

enum SIMULATION_MODE {CREATE_MOLECULE = 0, INITIALIZATION, EQUILIBRATION, PRODUCTION};

enum LAMBDA_TYPE {SHI_MAGINN = 0, BRICK_CFC};

enum DNN_CalcType {TOTAL = 0, OLD, NEW, REINSERTION_NEW, REINSERTION_OLD, DNN_INSERTION, DNN_DELETION};

enum ADD_MINUS_SIGNS {ADD = 0, MINUS};

enum ROTATION_AXIS {X = 0, Y, Z, SELF_DEFINED};

enum INTERACTION_TYPES {HH = 0, HG, GG};

enum RESTART_FILE_TYPES {RASPA_RESTART = 0, LAMMPS_DATA};

//Zhao's note: For the stage of evaluating total energy of the system//
enum ENERGYEVALSTAGE {INITIAL = 0, CREATEMOL, FINAL, CREATEMOL_DELTA, DELTA, CREATEMOL_DELTA_CHECK, DELTA_CHECK, DRIFT, GPU_DRIFT, AVERAGE, AVERAGE_ERR};

struct EnergyComplex
{
  double x;
  double y;
  double z;
};

struct Complex
{
  double real;
  double imag;
};

struct Units
{
  double MassUnit          = {1.6605402e-27};
  double TimeUnit          = {1e-12};
  double LengthUnit        = {1e-10};
  double energy_to_kelvin  = {1.2027242847};
  double BoltzmannConstant = {1.38e-23};
  double Avogadro          = {6.02214076e23};  // NIST standards [1/mol]
  double gas_constant      = {8.314462618}; // NIST standards [J/mol/K]
};

struct Gibbs
{
  bool    DoGibbs = false;
  double  GibbsBoxProb  = 0.0;
  double  GibbsXferProb = 0.0;
  double  GibbsTime=0.0;
  double  TotalVolume = 0.0;
  double TargetAccRatioVolumeChange = 0.5;

  double MaxGibbsBoxChange = 0.1;
  int2 GibbsBoxStats = {0, 0};
  int2 TotalGibbsBoxStats = {0, 0};
  int2 GibbsXferStats = {0, 0};
};

struct LAMBDA
{
  int    newBin;     //Bins range from (-binsize/2, binsize/2), since lambda is from -1 to 1.
  int    currentBin;
  int    lambdatype = SHI_MAGINN;
  double delta;  //size of the bin in the lambda histogram//
  double WangLandauScalingFactor;
  size_t binsize = {10};
  size_t FractionalMoleculeID;
  std::vector<double>Histogram;
  std::vector<double>biasFactor;
  double2 SET_SCALE(double lambda)
  {
    double2 scaling;
    switch(lambdatype)
    {
      case SHI_MAGINN: //Scale for Coulombic goes with pow(lambda,5)
      {
        scaling.x = lambda;
        scaling.y = std::pow(lambda, 5);
        break;
      }
      case BRICK_CFC: //Scale for Coulombic and VDW are according to the brick CFC code convention
      {
        scaling.x = lambda < 0.5 ? 2.0 * lambda : 1.0;
        scaling.y = lambda < 0.5 ? 0.0 : 2.0 * (lambda - 0.5);
        break;
      }
    }
    return scaling;
  }
};

struct TMMC
{
  std::vector<double3> CMatrix; //x = -1 (delete), y = 0 (other move that doesnot change macrostate), z = +1 (insert)//
  std::vector<double>  WLBias;
  std::vector<double>  TMBias;
  std::vector<double>  ln_g;    //For WL//
  std::vector<double>  lnpi;    //For TM//
  std::vector<double>  forward_lnpi; //Debugging//
  std::vector<double>  reverse_lnpi;
  std::vector<double>  Histogram;
  double WLFactor      = 1.0;
  size_t MaxMacrostate     = 0;
  size_t MinMacrostate     = 0;
  size_t nbinPerMacrostate = 1;
  size_t currentBin = 0;  //Should match what you have for lambda bin//
  //size_t UpdateTMEvery = 1000000;
  size_t UpdateTMEvery = 5000;
  size_t TMUpdateTimes = 0;
  bool   DoTMMC = false;
  bool   DoUseBias = false; //Whether or not to use WL or TM Bias for changing macrostates//
  bool   UseWLBias = false; //Whether to use WL for the Bias//
  bool   UseTMBias = true;  //Whether to use TM for the Bias//
  bool   RejectOutofBound = true; //Whether to reject the move out of the bound of macrostate//
  bool   TMMCRestart = false; //Whether to read a TMMC file from TMMC_Initial folder
  bool   RezeroAfterInitialization = false;
  //Zhao's note: the N below is the Number of Molecule from the OLD STATE//
  //(According to Vince Shen's code)//
  void Update(double Pacc, size_t N, int MoveType) 
  {
    if(!DoTMMC) return;
    if (Pacc > 1.0) Pacc = 1.0; //If Pacc is too big, reduce to 1.0//
    size_t BinLocation = (N - MinMacrostate) * nbinPerMacrostate + currentBin;
    switch(MoveType)
    {
      case TRANSLATION: case ROTATION: case REINSERTION:
      {
        if(RejectOutofBound && ((N > MaxMacrostate) || (N < MinMacrostate))) return;
        CMatrix[BinLocation].y += Pacc; //If for GCMC, the Pacc for these moves is always 1. should check for this//
        ln_g[BinLocation]      += WLFactor; 
        WLBias[BinLocation]     = -ln_g[N]; //WL Bias//
        Histogram[BinLocation] ++;
        break;
      }
      case INSERTION: case SINGLE_INSERTION:
      {
        size_t NewBinLocation = (N - MinMacrostate + 1) * nbinPerMacrostate + currentBin;
        CMatrix[BinLocation].z += Pacc;     //Insertion is the third value//
        CMatrix[BinLocation].y += 1-Pacc;
        //if(OldN > CMatrix.size()) printf("At the limit, OldN: %zu, N: %zu, NewN: %zu\n", OldN, N, NewN);
        if(RejectOutofBound && ((N + 1) > MaxMacrostate))
        {
          Histogram[N] ++; 
          return;
        }
        Histogram[NewBinLocation] ++;
        ln_g[NewBinLocation]  += WLFactor;
        WLBias[NewBinLocation] = -ln_g[N]; //WL Bias//
        break;
      }
      case DELETION: case SINGLE_DELETION:
      {
        size_t NewBinLocation = (N - MinMacrostate - 1) * nbinPerMacrostate + currentBin;
        CMatrix[BinLocation].x += Pacc;  //Deletion is the first value//
        CMatrix[BinLocation].y += 1-Pacc;
        ln_g[BinLocation]      += WLFactor;
        int Update_State = static_cast<int>(BinLocation) - 1;
        if(RejectOutofBound && (Update_State < static_cast<int>(MinMacrostate))) 
        {
          Histogram[BinLocation] ++;
          return;
        }
        Histogram[NewBinLocation] ++;
        ln_g[NewBinLocation]   += WLFactor;
        WLBias[NewBinLocation]  = -ln_g[N]; //WL Bias//
        break;
      }
      case CBCF_LAMBDACHANGE: case CBCF_INSERTION: case CBCF_DELETION:
      {
        printf("Use the SPECIAL FUNCTION FOR TMMC + CBCFC\n");
        break;
        /*
        size_t NewBinLocation = (N - MinMacrostate + 1) * nbinPerMacrostate + currentBin;
        CMatrix[BinLocation].z += Pacc;     //Insertion is the third value//
        CMatrix[BinLocation].y += 1-Pacc;
        //if(OldN > CMatrix.size()) printf("At the limit, OldN: %zu, N: %zu, NewN: %zu\n", OldN, N, NewN);
        if(RejectOutofBound && ((N + 1) > MaxMacrostate)) return;
        Histogram[NewBinLocation] ++;
        ln_g[NewBinLocation]  += WLFactor;
        WLBias[NewBinLocation] = -ln_g[N]; //WL Bias//
        break;
        */
      }
    }
  }
  //Special update function just for CBCF moves//
  void UpdateCBCF(double Pacc, size_t N, int change)
  {
    if(!DoTMMC) return;
    if (Pacc > 1.0) Pacc = 1.0; //If Pacc is too big, reduce to 1.0//
    size_t BinLocation = (N - MinMacrostate) * nbinPerMacrostate + currentBin;
    size_t NewBinLocation = BinLocation + change;
    size_t NTotalBins = Histogram.size();
    if(RejectOutofBound && (NewBinLocation >= NTotalBins || ((int) BinLocation + change) < 0)) return;

    CMatrix[BinLocation].z += Pacc;     //Insertion is the third value//
    CMatrix[BinLocation].y += 1-Pacc;
    Histogram[NewBinLocation] ++;
    ln_g[NewBinLocation]  += WLFactor;
    WLBias[NewBinLocation] = -ln_g[N]; //WL Bias//
  }
  //Zhao's note: the N below is the Number of Molecule from the OLD STATE//
  //The bias is added to the preFactor//
  //Following the way of WL sampling's bias for CBCFC (see mc_cbcfc.h)//
  //Need to see whether this makes sense//
  void ApplyWLBias(double& preFactor, size_t N, int MoveType)
  {
    if(!DoTMMC || !DoUseBias || !UseWLBias) return;
    if(N < MinMacrostate || N > MaxMacrostate) return; //No bias for macrostate out of the bound
    switch(MoveType)
    {
      case TRANSLATION: case ROTATION: case REINSERTION:
      {
        //Do not need the bias for moves that does not change the macrostate//
        break;
      }
      case INSERTION: case SINGLE_INSERTION: case CBCF_INSERTION:
      { 
        if(RejectOutofBound && (N + 1) > MaxMacrostate) return;
        N -= MinMacrostate;
        double TMMCBias = WLBias[N + 1] - WLBias[N];
        TMMCBias   = std::exp(TMMCBias); //See if Minus sign works//
        preFactor *= TMMCBias;
        break;
      }
      case DELETION: case SINGLE_DELETION: case CBCF_DELETION:
      {
        if(RejectOutofBound && (N - 1) < MinMacrostate) return;
        N -= MinMacrostate;
        double TMMCBias = WLBias[N - 1] - WLBias[N];
        TMMCBias   = std::exp(TMMCBias); //See if Minus sign works//
        preFactor *= TMMCBias;
        break;
      }
    }
  }
  void ApplyWLBiasCBCF(double& preFactor, size_t N, int change)
  {
    if(!DoTMMC || !DoUseBias || !UseWLBias) return;
    if(N < MinMacrostate || N > MaxMacrostate) return; //No bias for macrostate out of the bound
    size_t BinLocation = (N - MinMacrostate) * nbinPerMacrostate + currentBin;
    size_t NewBinLocation = BinLocation + change;
    size_t NTotalBins = Histogram.size();
    if(RejectOutofBound && (NewBinLocation >= NTotalBins || ((int) BinLocation + change) < 0)) return;
    double TMMCBias = WLBias[NewBinLocation] - WLBias[BinLocation];
    TMMCBias   = std::exp(TMMCBias); //See if Minus sign works//
    preFactor *= TMMCBias;
  }
  //Following the way of WL sampling's bias for CBCFC (see mc_cbcfc.h)//
  //Need to see whether this makes sense//
  //Zhao's note: the N below is the Number of Molecule from the OLD STATE//
  //The bias is added to the preFactor//
  void ApplyTMBias(double& preFactor, size_t N, int MoveType)
  {
    if(!DoTMMC || !DoUseBias || !UseTMBias) return;
    if(N < MinMacrostate || N > MaxMacrostate) return; //No bias for macrostate out of the bound
    switch(MoveType)
    {
      case TRANSLATION: case ROTATION: case REINSERTION: case CBCF_LAMBDACHANGE:
      {
        //Do not need the bias for moves that does not change the macrostate//
        break;
      }
      case INSERTION: case SINGLE_INSERTION: case CBCF_INSERTION:
      {
        if(RejectOutofBound && (N + 1) > MaxMacrostate) return;
        N -= MinMacrostate;
        double TMMCBias = TMBias[N + 1] - TMBias[N];
        TMMCBias   = std::exp(TMMCBias); //See if Minus sign works//
        preFactor *= TMMCBias;
        break;
      }
      case DELETION: case SINGLE_DELETION: case CBCF_DELETION:
      {
        if(RejectOutofBound && (N - 1) < MinMacrostate) return;
        N -= MinMacrostate;
        double TMMCBias = TMBias[N - 1] - TMBias[N];
        TMMCBias   = std::exp(TMMCBias); //See if Minus sign works//
        preFactor *= TMMCBias;
        break;
      }
    }
  }
  void ApplyTMBiasCBCF(double& preFactor, size_t N, int change)
  {
    if(!DoTMMC || !DoUseBias || !UseWLBias) return;
    if(N < MinMacrostate || N > MaxMacrostate) return; //No bias for macrostate out of the bound
    size_t BinLocation = (N - MinMacrostate) * nbinPerMacrostate + currentBin;
    size_t NewBinLocation = BinLocation + change;
    size_t NTotalBins = Histogram.size();
    if(RejectOutofBound && (NewBinLocation >= NTotalBins || ((int) BinLocation + change) < 0)) return;
    double TMMCBias = TMBias[NewBinLocation] - TMBias[BinLocation];
    TMMCBias   = std::exp(TMMCBias);
    preFactor *= TMMCBias;
  }
  //Add features about CBCFC + TMMC//
  void AdjustTMBias() 
  { 
    if(!DoTMMC || !DoUseBias || !UseTMBias) return;
    TMUpdateTimes ++;
    //printf("Adjusting TMBias\n");
    //First step is to get the lowest and highest visited states in terms of loading//
    size_t MinVisited = 0; size_t MaxVisited = 0;
    size_t nonzeroCount=0;
    //The a, MinVisited, and MaxVisited here do not go out of bound of the vector//
    size_t NTotalBins = Histogram.size();
    for(size_t a = 0; a < NTotalBins; a++)
    {
      if(Histogram[a] != 0)
      {
        if(nonzeroCount==0) MinVisited = a;
        MaxVisited = a;
        nonzeroCount++;
      }
    }
    //From Vince Shen's pseudo code//
    lnpi[MinVisited] = 0.0;
    double Maxlnpi = lnpi[MinVisited];
    //Update the lnpi for the sampled region//
    //x: -1; y: 0; z: +1//
    for(size_t a = MinVisited; a < MaxVisited; a++)
    {
      //Zhao's note: add protection to avoid numerical issues//
      if(CMatrix[a].z   != 0) lnpi[a+1] = lnpi[a]   + std::log(CMatrix[a].z)   - std::log(CMatrix[a].x   + CMatrix[a].y   + CMatrix[a].z);   //Forward//
      forward_lnpi[a+1] = lnpi[a+1];
      if(CMatrix[a+1].x != 0) lnpi[a+1] = lnpi[a+1] - std::log(CMatrix[a+1].x) + std::log(CMatrix[a+1].x + CMatrix[a+1].y + CMatrix[a+1].z); //Reverse//
      reverse_lnpi[a+1] = lnpi[a+1];
      //printf("Loading: %zu, a+1, %zu, lnpi[a]: %.5f, lnpi[a+1]: %.5f\n", a, a+1, lnpi[a], lnpi[a+1]);
      if(lnpi[a+1] > Maxlnpi) Maxlnpi = lnpi[a+1];
    }
    //For the unsampled states, fill them with the MinVisited/MaxVisited stats//
    for(size_t a = 0; a < MinVisited; a++) lnpi[a] = lnpi[MinVisited];
    for(size_t a = MaxVisited; a < NTotalBins; a++) lnpi[a] = lnpi[MaxVisited];
    //Normalize//
    double NormalFactor = 0.0;
    for(size_t a = 0; a < NTotalBins; a++) lnpi[a] -= Maxlnpi;
    for(size_t a = 0; a < NTotalBins; a++) NormalFactor += std::exp(lnpi[a]); //sum of exp(lnpi)//
    //printf("Normalize Factor (Before): %.5f\n", NormalFactor);
    NormalFactor = -std::log(NormalFactor); //Take log of NormalFactor//
    //printf("Normalize Factor (After):  %.5f\n", NormalFactor);
    for(size_t a = 0; a < NTotalBins; a++)
    {
      lnpi[a] += NormalFactor; //Zhao's note: mind the sign//
      TMBias[a]= -lnpi[a];
    }
  }
  //Determine whether to reject the move if it move out of the bounds//
  void TreatAccOutofBound(bool& Accept, size_t N, int MoveType)
  {
    if(!DoTMMC) return;
    if(!Accept || !RejectOutofBound) return; //if the move is already rejected, no need to reject again//
    switch(MoveType)
    {
      case TRANSLATION: case ROTATION: case REINSERTION:
      {
        //Do not need to determine accept/reject for moves that does not change the macrostate//
        break;
      }
      case INSERTION: case SINGLE_INSERTION: 
      {
        if(RejectOutofBound && (N + 1) > MaxMacrostate) Accept = false;
        break;
      }
      case DELETION: case SINGLE_DELETION:
      {
        if(RejectOutofBound && (N - 1) < MinMacrostate) Accept = false;
        break;
      }
    }
  }
  void TreatAccOutofBoundCBCF(bool& Accept, size_t N, int change)
  {
    if(!DoTMMC) return;
    if(!Accept || !RejectOutofBound) return; //if the move is already rejected, no need to reject again//
    size_t BinLocation = (N - MinMacrostate) * nbinPerMacrostate + currentBin;
    size_t NewBinLocation = BinLocation + change;
    size_t NTotalBins = Histogram.size();
    if(NewBinLocation >= NTotalBins) Accept = false;
  }
  //Clear Collection matrix stats (used after initialization cycles)
  void ClearCMatrix()
  {
    if(!DoTMMC || !RezeroAfterInitialization) return;
    double3 temp = {0.0, 0.0, 0.0};
    std::fill(CMatrix.begin(),  CMatrix.end(),  temp);
    std::fill(Histogram.begin(),  Histogram.end(),  0.0);
    std::fill(lnpi.begin(), lnpi.end(), 0.0);
    std::fill(TMBias.begin(), TMBias.end(), 1.0);
  }
};

//Zhao's note: keep track of the Rosenbluth weights during the simulation//
struct RosenbluthWeight
{
  double3 Total          = {0.0, 0.0, 0.0};
  double3 WidomInsertion = {0.0, 0.0, 0.0};
  double3 Insertion      = {0.0, 0.0, 0.0};
  double3 Deletion       = {0.0, 0.0, 0.0};
  //NOTE: DO NOT RECORD FOR REINSERTIONS, SINCE THE DELETION PART OF REINSERTION IS MODIFIED//
};

struct Move_Statistics
{
  //Translation Move//
  double TranslationProb        =0.0;
  double RotationProb           =0.0;
  double SpecialRotationProb    =0.0;
  double WidomProb              =0.0;
  double SwapProb               =0.0;
  double VolumeMoveProb         =0.0;
  double GibbsSwapProb          =0.0;
  double ReinsertionProb        =0.0;
  double IdentitySwapProb       =0.0;
  double CBCFProb               =0.0;
  double GibbsVolumeMoveProb    =0.0;
  double TotalProb              =0.0;
  //Translation Move//
  int TranslationAccepted = 0; //zeroed when max translation updated 
  int TranslationTotal = 0;    //zeroed when max translation updated
  int CumTranslationAccepted = 0; //Cumulative
  int CumTranslationTotal = 0;
  double TranslationAccRatio = 0.0;
  //Rotation Move//
  int RotationAccepted = 0;    //zeroed when max rotation updated
  int RotationTotal = 0;       //zeroed when max rotation updated
  int CumRotationAccepted = 0; //Cumulative
  int CumRotationTotal = 0;
  double RotationAccRatio = 0;
  //Special Rotation Move//
  int SpecialRotationAccepted = 0;
  int SpecialRotationTotal = 0;
  double SpecialRotationAccRatio = 0;
  //Insertion Move//
  size_t InsertionTotal = 0;
  size_t InsertionAccepted = 0;
  //Deletion Move//
  size_t DeletionTotal = 0;
  size_t DeletionAccepted = 0;
  //Reinsertion Move//
  size_t ReinsertionTotal = 0;
  size_t ReinsertionAccepted = 0;
  //CBCFSwap Move//
  size_t CBCFTotal = 0;
  size_t CBCFAccepted = 0;
  size_t CBCFInsertionTotal = 0;
  size_t CBCFInsertionAccepted = 0;
  size_t CBCFLambdaTotal = 0;
  size_t CBCFLambdaAccepted = 0;
  size_t CBCFDeletionTotal = 0;
  size_t CBCFDeletionAccepted = 0;
  //Identity Swap Move//
  std::vector<size_t>IdentitySwap_Total_TO;
  std::vector<size_t>IdentitySwap_Acc_TO;
  size_t IdentitySwapAddAccepted=0;
  size_t IdentitySwapAddTotal=0;
  size_t IdentitySwapRemoveAccepted=0;
  size_t IdentitySwapRemoveTotal=0;

  size_t BlockID = 0; //Keep track of the current Block for Averages//
  std::vector<double2>MolAverage;
  //Cross terms for each component with other components: Na x Nb, for here, Na is fixed//
  //2 dimensions: Nb (component b) x Nblocks//
  std::vector<std::vector<double>>MolSQPerComponent;
  //x: average; y: average^2; z: Number of Widom insertion performed//
  std::vector<RosenbluthWeight>Rosen; //vector over Nblocks//
  void NormalizeProbabilities()
  {
    //Zhao's note: the probabilities here are what we defined in simulation.input, raw values//
    TotalProb+=TranslationProb;
    TotalProb+=RotationProb;
    TotalProb+=SpecialRotationProb;
    TotalProb+=WidomProb;
    TotalProb+=ReinsertionProb;
    TotalProb+=IdentitySwapProb;
    TotalProb+=SwapProb;
    TotalProb+=VolumeMoveProb;
    TotalProb+=GibbsSwapProb;
    TotalProb+=CBCFProb;
    TotalProb+=GibbsVolumeMoveProb;
    if(TotalProb > 1e-10)
    {
      //printf("TotalProb: %.5f\n", TotalProb);
      TranslationProb    /=TotalProb;
      RotationProb       /=TotalProb;
      SpecialRotationProb/=TotalProb;
      WidomProb          /=TotalProb;
      SwapProb           /=TotalProb;
      GibbsSwapProb      /=TotalProb;
      CBCFProb           /=TotalProb;
      ReinsertionProb    /=TotalProb;
      IdentitySwapProb   /=TotalProb;
      GibbsVolumeMoveProb/=TotalProb;
      TotalProb = 1.0;
    }
    RotationProb        += TranslationProb;
    SpecialRotationProb += RotationProb;
    WidomProb           += SpecialRotationProb;
    ReinsertionProb     += WidomProb;
    IdentitySwapProb    += ReinsertionProb;
    CBCFProb            += IdentitySwapProb;
    SwapProb            += CBCFProb;
    VolumeMoveProb      += SwapProb;
    GibbsSwapProb       += VolumeMoveProb;
    GibbsVolumeMoveProb += GibbsSwapProb;
  }
  void PrintProbabilities()
  {
    printf("==================================================\n");
    printf("ACCUMULATED Probabilities:\n");
    printf("Translation Probability:      %.5f\n", TranslationProb);
    printf("Rotation Probability:         %.5f\n", RotationProb);
    printf("Special Rotation Probability: %.5f\n", SpecialRotationProb);
    printf("Widom Probability:            %.5f\n", WidomProb);
    printf("Reinsertion Probability:      %.5f\n", ReinsertionProb);
    printf("Identity Swap Probability:    %.5f\n", IdentitySwapProb);
    printf("CBCF Swap Probability:        %.5f\n", CBCFProb);
    printf("Swap Probability:             %.5f\n", SwapProb);
    printf("Volume Probability:           %.5f\n", VolumeMoveProb);
    printf("Gibbs Swap Probability:       %.5f\n", GibbsSwapProb);
    printf("Gibbs Volume Probability:     %.5f\n", GibbsVolumeMoveProb);
    printf("Sum of Probabilities:         %.5f\n", TotalProb);
    printf("==================================================\n");
  }
  void RecordRosen(double R, int MoveType)
  {
    if(MoveType != INSERTION && MoveType != DELETION && MoveType != WIDOM) return;
    double R2 = R * R;
    Rosen[BlockID].Total.x += R;
    Rosen[BlockID].Total.y += R * R;
    Rosen[BlockID].Total.z += 1.0;
    if(MoveType == INSERTION)
    {
      Rosen[BlockID].Insertion.x += R;
      Rosen[BlockID].Insertion.y += R * R;
      Rosen[BlockID].Insertion.z += 1.0;
    }
    else if(MoveType == DELETION)
    {
      Rosen[BlockID].Deletion.x += R;
      Rosen[BlockID].Deletion.y += R * R;
      Rosen[BlockID].Deletion.z += 1.0;
    }
    else if(MoveType == WIDOM)
    {
      Rosen[BlockID].WidomInsertion.x += R;
      Rosen[BlockID].WidomInsertion.y += R * R;
      Rosen[BlockID].WidomInsertion.z += 1.0;
    }
  }
  void ClearRosen(size_t BlockID)
  {
    Rosen[BlockID].Total          = {0.0, 0.0, 0.0};
    Rosen[BlockID].Insertion      = {0.0, 0.0, 0.0};
    Rosen[BlockID].Deletion       = {0.0, 0.0, 0.0};
    Rosen[BlockID].WidomInsertion = {0.0, 0.0, 0.0};
  }
  void Record_Move_Total(int MoveType)
  {
    switch(MoveType)
    {
      case TRANSLATION: {TranslationTotal++; break; }
      case ROTATION: {RotationTotal++; break; }
      case SPECIAL_ROTATION: {SpecialRotationTotal++; break; }
      case INSERTION: case SINGLE_INSERTION: {InsertionTotal++; break; }
      case DELETION:  case SINGLE_DELETION:  {DeletionTotal++; break; }
    }
  }
  void Record_Move_Accept(int MoveType)
  {
    switch(MoveType)
    {
      case TRANSLATION: {TranslationAccepted++; break; }
      case ROTATION: {RotationAccepted++; break; }
      case SPECIAL_ROTATION: {SpecialRotationAccepted++; break; }
      case INSERTION: case SINGLE_INSERTION: {InsertionAccepted++; break; }
      case DELETION:  case SINGLE_DELETION:  {DeletionAccepted++; break; }
    }
  }
};

struct SystemEnergies
{
  double CreateMol_running_energy={0.0};
  double running_energy   = {0.0};
  //Total Energies//
  double InitialEnergy    = {0.0};
  double CreateMolEnergy  = {0.0};
  double FinalEnergy      = {0.0};
  //VDW//
  double InitialVDW       = {0.0};
  double CreateMolVDW     = {0.0};
  double FinalVDW         = {0.0};
  //Real Part Coulomb//
  double InitialReal      = {0.0};
  double CreateMolReal    = {0.0};
  double FinalReal        = {0.0};
  //Ewald Energy//
  double InitialEwaldE    = {0.0};
  double CreateMolEwaldE  = {0.0};
  double FinalEwaldE      = {0.0};
  //HostGuest Ewald//
  double InitialHGEwaldE  = {0.0};
  double CreateMolHGEwaldE= {0.0};
  double FinalHGEwaldE    = {0.0};
  //Tail Correction//
  double InitialTailE     = {0.0};
  double CreateMolTailE   = {0.0};
  double FinalTailE       = {0.0};
  //DNN Energy//
  double InitialDNN       = {0.0};
  double CreateMolDNN     = {0.0};
  double FinalDNN         = {0.0};
};

struct Tail
{
  bool   UseTail = {false};
  double Energy  = {0.0};
};

//Temporary energies for a move//
//Zhao's note: consider making this the default return variable for moves, like RASPA-3?//
struct MoveEnergy
{
  double storedHGVDW=0.0;
  double storedHGReal=0.0;
  double storedHGEwaldE=0.0;
  // van der Waals //
  double HHVDW=0.0;
  double HGVDW=0.0;
  double GGVDW=0.0;
  // Real Part of Coulomb //
  double HHReal=0.0;
  double HGReal=0.0;
  double GGReal=0.0;
  // Long-Range Ewald Energy //
  double HHEwaldE=0.0;
  double HGEwaldE=0.0;
  double GGEwaldE=0.0;
  // Other Energies //
  double TailE =0.0;
  double DNN_E =0.0;
  double total()
  {
    return HHVDW + HGVDW + GGVDW + 
           HHReal + HGReal + GGReal + 
           HHEwaldE + HGEwaldE + GGEwaldE + 
           TailE + DNN_E;
  };
  void take_negative()
  {
    storedHGVDW *= -1.0;
    storedHGReal*= -1.0;
    storedHGEwaldE  *= -1.0;
    HHVDW *= -1.0; HHReal *= -1.0;
    HGVDW *= -1.0; HGReal *= -1.0;
    GGVDW *= -1.0; GGReal *= -1.0;
    HHEwaldE *= -1.0; HGEwaldE *= -1.0; GGEwaldE  *= -1.0;
    TailE     *= -1.0;
    DNN_E     *= -1.0;
  };
  void zero()
  {
    storedHGVDW =0.0;
    storedHGReal=0.0;
    storedHGEwaldE=0.0;
    HHVDW=0.0; HHReal=0.0;
    HGVDW=0.0; HGReal=0.0;
    GGVDW=0.0; GGReal=0.0;
    HHEwaldE =0.0;
    HGEwaldE =0.0;
    GGEwaldE =0.0;
    TailE    =0.0;
    DNN_E    =0.0;
  };
  void print()
  {
    printf("HHVDW: %.5f, HHReal: %.5f, HGVDW: %.5f, HGReal: %.5f, GGVDW: %.5f, GGReal: %.5f, HHEwaldE: %.5f,\n HGEwaldE: %.5f,\n GGEwaldE: %.5f, TailE: %.5f, DNN_E: %.5f\n", HHVDW, HHReal, HGVDW, HGReal, GGVDW, GGReal, HHEwaldE, HGEwaldE, GGEwaldE, TailE, DNN_E);
    printf("Stored HGVDW: %.5f, Stored HGReal: %.5f, Stored HGEwaldE: %.5f\n", storedHGVDW, storedHGReal, storedHGEwaldE);
  };
  double DNN_Correction() //Using DNN energy to replace HGVDW, HGReal and HGEwaldE//
  {
    double correction = DNN_E - HGVDW - HGReal - HGEwaldE;
    storedHGVDW = HGVDW; 
    storedHGReal= HGReal;
    storedHGEwaldE  = HGEwaldE;
    HGVDW = 0.0; 
    HGReal= 0.0;
    HGEwaldE = 0.0;
    return correction;
  };
};
/*
__host__ MoveEnergy MoveEnergy_Multiply(MoveEnergy A, MoveEnergy B)
{
  MoveEnergy X;
  X.storedHGVDW    = A.storedHGVDW * B.storedHGVDW;
  X.storedHGReal   = A.storedHGReal * B.storedHGReal;
  X.storedHGEwaldE = A.storedHGEwaldE * B.storedHGEwaldE;

  X.HHVDW    = A.HHVDW    * B.HHVDW;
  X.HGVDW    = A.HGVDW    * B.HGVDW;
  X.GGVDW    = A.GGVDW    * B.GGVDW;
  X.HHReal   = A.HHReal   * B.HHReal;
  X.HGReal   = A.HGReal   * B.HGReal;
  X.GGReal   = A.GGReal   * B.GGReal;
  X.HHEwaldE = A.HHEwaldE * B.HHEwaldE;
  X.HGEwaldE = A.HGEwaldE * B.HGEwaldE;
  X.GGEwaldE = A.GGEwaldE * B.GGEwaldE;
  X.TailE    = A.TailE    * B.TailE;
  X.DNN_E    = A.DNN_E    * B.DNN_E;
  return X;
}
__host__ MoveEnergy MoveEnergy_DIVIDE_DOUBLE(MoveEnergy A, double B)
{
  MoveEnergy X;
  double OneOverB = 1.0 / B;
  X.storedHGVDW     = A.storedHGVDW    * OneOverB;
  X.storedHGReal    = A.storedHGReal   * OneOverB;
  X.storedHGEwaldE  = A.storedHGEwaldE * OneOverB;

  X.HHVDW    = A.HHVDW * OneOverB;
  X.HGVDW    = A.HGVDW * OneOverB;
  X.GGVDW    = A.GGVDW * OneOverB;
  X.HHReal   = A.HHReal * OneOverB;
  X.HGReal   = A.HGReal * OneOverB;
  X.GGReal   = A.GGReal * OneOverB;
  X.HHEwaldE = A.HHEwaldE * OneOverB;
  X.HGEwaldE = A.HGEwaldE * OneOverB;
  X.GGEwaldE = A.HHEwaldE * OneOverB;
  X.TailE    = A.TailE * OneOverB;
  X.DNN_E    = A.DNN_E * OneOverB;
  return X;
}
__host__ void operator +=(MoveEnergy& A, MoveEnergy B)
{
  A.storedHGVDW     += B.storedHGVDW;
  A.storedHGReal    += B.storedHGReal;
  A.storedHGEwaldE  += B.storedHGEwaldE;

  A.HHVDW    += B.HHVDW;
  A.HGVDW    += B.HGVDW;
  A.GGVDW    += B.GGVDW;
  A.HHReal   += B.HHReal;
  A.HGReal   += B.HGReal;
  A.GGReal   += B.GGReal;
  A.HHEwaldE += B.HHEwaldE;
  A.HGEwaldE += B.HGEwaldE;
  A.GGEwaldE += B.GGEwaldE;
  A.TailE    += B.TailE;
  A.DNN_E    += B.DNN_E;
}
*/
struct Atoms
{
  double3* pos;
  double*  scale;
  double*  charge;
  double*  scaleCoul;
  size_t*  Type;
  size_t*  MolID;
  size_t   Molsize;
  size_t   size;
  size_t   Allocate_size;
};

struct FRAMEWORK_COMPONENT_LISTS
{
  bool   SeparatedComponent = false; //By false, it means the framework is assumed as a whole by default//
  size_t Number_of_Molecules_for_Framework_component;
  size_t Number_of_atoms_for_each_molecule;
  std::vector<std::vector<size_t>>Atom_Indices_for_Molecule;
};

struct PseudoAtomDefinitions //Always a host struct, never on the device
{
  std::vector<std::string> Name;
  std::vector<std::string> Symbol; //Symbol name for each pseudo-atom
  std::vector<std::string> UniqueSymbol; //all the unique Symbol list //
  std::vector<size_t> SymbolIndex; //It has the size of the number of pseudo atoms, it tells the ID of the symbol for the pseudo-atoms, e.g., CO2->C->2
  std::vector<double> oxidation;
  std::vector<double> mass;
  std::vector<double> charge;
  std::vector<double> polar; //polarizability
  size_t MatchUniqueSymbolTypeFromSymbolName(std::string& SymbolName)
  {
    size_t SymbolIdx = UniqueSymbol.size();
    for(size_t i = 0; i < UniqueSymbol.size(); i++)
    {
      if(SymbolName == UniqueSymbol[i])
      {
        SymbolIdx = i; break;
      }
    }
    return SymbolIdx;
  }
  size_t GetSymbolIdxFromPseudoAtomTypeIdx(size_t Type)
  {
    return SymbolIndex[Type];
  }
};

struct ForceField
{
  double* epsilon;
  double* sigma;
  double* z; // a third term
  double* shift;
  int*    FFType; //type of force field calculation  
  bool    noCharges;
  bool    VDWRealBias = true; //By default, the CBMC moves use VDW + Real Biasing//
  size_t  size;
  double  OverlapCriteria;
  double  CutOffVDW;       // Square of cutoff for vdW interaction //
  double  CutOffCoul;      // Square of cutoff for Coulombic interaction //
  //double  Prefactor;
  //double  Alpha;
};

struct NeighList
{
  std::vector<std::vector<int3>>List;
  std::vector<std::vector<int3>>FrameworkList;
  std::vector<size_t>cumsum_neigh_per_atom;
  size_t nedges = 0;
};

struct Boxsize
{
  Complex* eik_x;
  Complex* eik_y;
  Complex* eik_z;
  Complex* AdsorbateEik;
  Complex* FrameworkEik;
  Complex* tempEik;
  Complex* tempFrameworkEik;

  double*  Cell;
  double*  InverseCell;
  double   Volume;
  double   ReciprocalCutOff;
  double   Prefactor;
  double   Alpha;
  double   tol1; //For Ewald, see read_Ewald_Parameters_from_input function//
  
  bool     Cubic;
  bool     UseLAMMPSEwald = false;
  int3     kmax;
};
//###PATCH_ALLEGRO_H###//

struct Components
{
  int3    NComponents;                                // Total components (x), Framework Components (y), Guest Components (z)
  int3    NumberofUnitCells;
  size_t  MoviesEvery = 5000;                         // Write Movies (LAMMPS data file) every X MC Steps/Cycles
  size_t  PrintStatsEvery = 5000;                     // Write instantaneous loading and energy to screen every X MC Steps/Cycles

  size_t  Nblock=5;                                 // Number of Blocks for block averages
  size_t  CURRENTCYCLE=0;
  double  DNNPredictTime=0.0; 
  double  DNNFeatureTime=0.0;
  double  DNNGPUTime=0.0;
  double  DNNSortTime=0.0;
  double  DNNstdsortTime=0.0;
  double  DNNFeaturizationTime=0.0;
  size_t  TotalNumberOfMolecules;                     // Total Number of Molecules (including framework)
  size_t  NumberOfFrameworks;                         // Total Number of framework species, usually 1.
  double  Temperature=0.0;
  double  Pressure=0.0;
  double  Beta;                                       // Inverse Temperature 

  size_t  EnergyEvalTimes = 0;

  bool*   flag;                                       // flags for checking overlaps (on host), device version in Simulations struct//
  size_t EikAllocateSize = 0;
  size_t tempEikAllocateSize = 0;

  std::vector<FRAMEWORK_COMPONENT_LISTS>FrameworkComponentDef;

  MoveEnergy Initial_Energy;
  MoveEnergy CreateMol_Energy;
  MoveEnergy Final_Energy;
  MoveEnergy GPU_Energy;
  //Zhao's note: for average and standard deviations for energies
  std::vector<MoveEnergy> BookKeepEnergy;
  std::vector<MoveEnergy> BookKeepEnergy_SQ;
  MoveEnergy AverageEnergy;
  MoveEnergy AverageEnergy_Errorbar;

  //NPT VOLUME MOVE//
  bool   PerformVolumeMove      = false;
  int    VolumeMoveAttempts     = 0;
  int    VolumeMoveAccepted     = 0;
  int    VolumeMoveTotalAttempts= 0;
  int    VolumeMoveTotalAccepted= 0;
  double VolumeMoveMaxChange    = 0.025;
  double VolumeMoveProbability  = 0.0;
  double VolumeMoveTargetAccRatio = 0.5;
  double VolumeMoveTime = 0.0;
  std::vector<double2>VolumeAverage;
  std::vector<std::vector<double2>>DensityPerComponent;
  //std::vector<double2>EnergyAverage;
  /*
  //Zhao's note: do not use pass by ref for DeltaE
  void Gather_Averages_MoveEnergy(int Cycles, int Blocksize, MoveEnergy DeltaE)
  {
    size_t blockID = Cycles/Blocksize;
    if(blockID >= Nblock) blockID --;
    if(blockID == Nblock-1)
    {
      if(Cycles % Blocksize != 0) Blocksize += Cycles % Blocksize;
    }
    //Get total energy//
    //MoveEnergy UpdateDeltaE = ;
    BookKeepEnergy[blockID]    += MoveEnergy_DIVIDE_DOUBLE(DeltaE, static_cast<double>(Blocksize));
    BookKeepEnergy_SQ[blockID] += MoveEnergy_Multiply(MoveEnergy_DIVIDE_DOUBLE(DeltaE, static_cast<double>(Blocksize)), DeltaE);
  }
  void Calculate_Overall_Averages_MoveEnergy(int Blocksize)
  {
    //Calculate just the overall, now//
    for(size_t i = 0; i < Nblock; i++)
    {
      AverageEnergy          += MoveEnergy_DIVIDE_DOUBLE(BookKeepEnergy[i], static_cast<double>(Nblock));
      AverageEnergy_Errorbar += MoveEnergy_Multiply(MoveEnergy_DIVIDE_DOUBLE(BookKeepEnergy[i], static_cast<double>(Nblock)), BookKeepEnergy[i]);
    }
  }
  */

  Atoms*  HostSystem;                                 //CPU pointers for storing Atoms (d_a in Simulations is the GPU Counterpart)//
  Atoms   TempSystem;                                 //For temporary data storage//
  void Copy_GPU_Data_To_Temp(Atoms GPU_System, size_t start, size_t size)
  {
    cudaMemcpy(TempSystem.pos,       &GPU_System.pos[start],       size * sizeof(double3), cudaMemcpyDeviceToHost);
    cudaMemcpy(TempSystem.scale,     &GPU_System.scale[start],     size * sizeof(double),  cudaMemcpyDeviceToHost);
    cudaMemcpy(TempSystem.charge,    &GPU_System.charge[start],    size * sizeof(double),  cudaMemcpyDeviceToHost);
    cudaMemcpy(TempSystem.scaleCoul, &GPU_System.scaleCoul[start], size * sizeof(double),  cudaMemcpyDeviceToHost);
    cudaMemcpy(TempSystem.Type,      &GPU_System.Type[start],      size * sizeof(size_t),  cudaMemcpyDeviceToHost);
    cudaMemcpy(TempSystem.MolID,     &GPU_System.MolID[start],     size * sizeof(size_t),  cudaMemcpyDeviceToHost);
  }
  MoveEnergy CreateMoldeltaE;
  MoveEnergy deltaE;
  double  FrameworkEwald=0.0;
  bool    HasTailCorrection = false;                // An overall flag for tail correction
  bool    ReadRestart = false;                      // Whether to use restart files //Zhao's note: this can be either RASPA-2-type Restart file or LAMMPS data file //
  int     RestartInputFileType = RASPA_RESTART;          // can choose from: RASPA_RESTART or LAMMPS_DATA (see enum at the beginning of this file)
  bool    Read_BoxsizeRestart = false;        // Whether to read boxsize from initial configuration file //
  bool    SingleSwap=false;
  ///////////////////////////
  // DNN Related Variables //
  ///////////////////////////
  //General DNN Flags//
  bool UseDNNforHostGuest = false;
  size_t TranslationRotationDNNReject=0;
  size_t ReinsertionDNNReject=0;
  size_t InsertionDNNReject=0;
  size_t DeletionDNNReject=0;
  size_t SingleSwapDNNReject=0;
  //DNN and Host-Guest Drift//
  double SingleMoveDNNDrift=0.0;
  double ReinsertionDNNDrift=0.0;
  double InsertionDNNDrift=0.0;
  double DeletionDNNDrift=0.0;
  double SingleSwapDNNDrift=0.0;
  double DNNDrift = 100000.0;
  double DNNEnergyConversion;
  bool UseAllegro = false;
  bool UseLCLin = false;
  //###PATCH_ALLEGRO_VARIABLES###//

  //###PATCH_LCLIN_VARIABLES###//
  std::vector<std::string>ModelName;                  // Name (folder) of the stored model
  std::vector<std::string>InputLayer;                 // Name of the input layer, run cli to get it
  size_t* device_InverseIndexList;                    // device_pointer for knowing which pair of interaction is stored in where
  bool*   ConsiderThisAdsorbateAtom;                  // device pointer
  double* device_Distances;                           // device_pointer for storing pair-wise distances//
 
  std::vector<std::vector<double>> EnergyTimesNumberOfMolecule; // Book-keeping Energy times the number of molecules for heat of adsorption
 
  std::vector<bool>   hasPartialCharge;               // Whether this component has partial charge
  std::vector<bool>   hasfractionalMolecule;          // Whether this component has fractional molecules
  std::vector<LAMBDA> Lambda;                         // Vector of Lambda struct
  std::vector<TMMC>   Tmmc;                           // Vector of TMMC struct
  std::vector<bool>   rigid;                          // Determine if the component is rigid.
  std::vector<double> ExclusionIntra;                 // For Ewald Summation, Intra Molecular exclusion
  std::vector<double> ExclusionAtom;                  // For Ewald Summation, Atom-self exclusion
  std::vector<std::string> MoleculeName;              // Name of the molecule
  std::vector<size_t> Moleculesize;                   // Number of Pseudo-Atoms in the molecule
  std::vector<size_t> NumberOfMolecule_for_Component; // Number of Molecules for each component (including fractional molecule)
  std::vector<size_t> Allocate_size;                  // Allocate size 
  std::vector<size_t> NumberOfCreateMolecules;        // Number of Molecules needed to create for every component
  std::vector<double> MolFraction;                    // Mol fraction of the component(excluding the framework)
  std::vector<double> IdealRosenbluthWeight;          // Ideal Rosenbluth weight
  std::vector<double> FugacityCoeff;                  // Fugacity Coefficient //Zhao's note: We can use negative FugacityCoeff for flag of using EOS for the fugacity Coefficient

  std::vector<Move_Statistics>Moves;                  // Move statistics: total, acceptance, etc.
  std::vector<double3> MaxTranslation;
  std::vector<double3> MaxRotation;
  std::vector<double3> MaxSpecialRotation;
  std::vector<double>Tc;                              // Critical Temperature of the component
  std::vector<double>Pc;                              // Critical Pressure of the component
  std::vector<double>Accentric;                       // Accentric Factor of the component
  std::vector<Tail>TailCorrection;                    // Tail Correction
  std::vector<double>MolecularWeight;                 // Molecular Weight of the component
  ForceField FF;
  PseudoAtomDefinitions PseudoAtoms;
  std::vector<size_t>NumberOfPseudoAtoms;             // NumberOfPseudoAtoms
  std::vector<std::vector<int2>>NumberOfPseudoAtomsForSpecies;     // NumberOfPseudoAtomsForSpecies
  std::vector<std::complex<double>> eik_xy;           
  std::vector<std::complex<double>> eik_x;  
  std::vector<std::complex<double>> eik_y;
  std::vector<std::complex<double>> eik_z;
  std::vector<std::complex<double>> AdsorbateEik;        // Stored Ewald Vectors for Adsorbate
  std::vector<std::complex<double>> FrameworkEik;        // Stored Ewald Vectors for Framework
  std::vector<std::complex<double>> tempEik;             // Ewald Vector for temporary storage
  size_t StructureFactor_Multiplier = 2;                 // Add extra structure factor storage for volume moves//
  size_t MatchMoleculeNameToComponentID(std::string Name)
  {
    for(size_t i = 0; i < MoleculeName.size(); i++)
    if(MoleculeName[i] == Name)
    {
      return i;
    }
    throw std::runtime_error("CANNOT find Molecule Names match " + Name + " !!!! CHECK YOUR FILE!");
  }
  void UpdatePseudoAtoms(int MoveType, size_t component)
  {
    switch(MoveType)
    {
      case INSERTION: case SINGLE_INSERTION: case CBCF_INSERTION:
      {
        for(size_t i = 0; i < NumberOfPseudoAtomsForSpecies[component].size(); i++)
        {
          size_t type = NumberOfPseudoAtomsForSpecies[component][i].x; size_t Num = NumberOfPseudoAtomsForSpecies[component][i].y;
          NumberOfPseudoAtoms[type] += Num;
        }
        break;
      }
      case DELETION: case SINGLE_DELETION: case CBCF_DELETION:
      {
        for(size_t i = 0; i < NumberOfPseudoAtomsForSpecies[component].size(); i++)
        {
          size_t type = NumberOfPseudoAtomsForSpecies[component][i].x; size_t Num = NumberOfPseudoAtomsForSpecies[component][i].y;
          NumberOfPseudoAtoms[type] -= Num;
        }
        break;
      }
      case TRANSLATION: case ROTATION: case REINSERTION: case CBCF_LAMBDACHANGE:
        break;
    }
  }
  FILE* OUTPUT = stderr;
};



struct Simulations //For multiple simulations//
{
  Atoms*  d_a;                  // Pointer For Atom Data in the Simulation Box //
  Atoms   Old;                  // Temporary data storage for Old Configuration //
  Atoms   New;                  // Temporary data storage for New Configuration //
  int2*   ExcludeList;          // Atoms to exclude during energy calculations: x: component, y: molecule-ID (may need to add z and make it int3, z: atom-ID)
  double* Blocksum;             // Block sums for partial reduction //
  bool*   device_flag;          // flags for overlaps on the device //
  size_t  start_position;       // Start position for reading data in d_a when proposing a trial position for moves //
  size_t  Nblocks;              // Number of blocks for energy calculation, NOT block averages! //
  Boxsize Box;                  // Each simulation (system) has its own box //
};



struct WidomStruct
{
  bool                UseGPUReduction;                  // For calculating the energies for each bead
  bool                Useflag;                          // For using flags (for skipping reduction)
  size_t              NumberWidomTrials = 8;            // Number of Trial Positions for the first bead //
  size_t              NumberWidomTrialsOrientations = 8;// Number of Trial Orientations 
  size_t              WidomFirstBeadAllocatesize;       //space allocated for WidomFirstBeadResult
};

static __global__ void Aaccess_device_random(double3* device_random)
{
  device_random[0] = {2.3, 4.5, 6.7};
  printf("device_random[0] = %.5f %.5f %.5f\n", device_random[0].x, device_random[0].y, device_random[0].z);
}


struct RandomNumber
{
  double3* host_random;
  double3* device_random;
  int      RANDOMSEED = 0;
  size_t   randomsize;
  size_t   offset=0;
  size_t   Rounds=0;
  void AllocateRandom() //Allocate space for random numbers on cpu//
  {
    host_random = (double3*) malloc(randomsize*sizeof(double3));
    host_random[0] = {1.0, 1.0, 0.5};
  }
  void ResetRandom() //Regenerates random numbers on cpu, then transfer to gpu//
  {
    offset = 0;
    for (size_t i = 0; i < randomsize; i++) 
    {
      host_random[i].x = Get_Uniform_Random();
      host_random[i].y = Get_Uniform_Random();
      host_random[i].z = Get_Uniform_Random();
      /*
      if(i < 100)
      {
        printf("INDEX: %zu, RANDOM: %.5f %.5f %.5f\n", i, host_random[i].x, host_random[i].y, host_random[i].z);
      }
      */
    }

    for(size_t i = randomsize * 3; i < 1000000; i++) Get_Uniform_Random();

    cudaMemcpy(device_random, host_random, randomsize * sizeof(double3), cudaMemcpyHostToDevice);
    Rounds ++;
  }
  
  void DeviceRandom() //Allocate space on gpu, generate random numbers//
  {
    cudaMalloc(&device_random, randomsize * sizeof(double3));
    ResetRandom();
    Aaccess_device_random<<<1,1>>>(device_random);
    cudaMemcpy(host_random, device_random, randomsize * sizeof(double3), cudaMemcpyDeviceToHost);
  }
  
  void Check(size_t change) //check the usage of random numbers, if used up, regenerate//
  {
    if((offset + change) >= randomsize) ResetRandom();
  }
  void Update(size_t change) //update usage of random numbers//
  {
    offset += change;
  }
  void Setup(size_t SIZE)
  {
    std::srand(RANDOMSEED); //Zhao's note: RANDOMSEED is read when reading the input file//
    randomsize = SIZE;
    AllocateRandom();
    DeviceRandom();
    Rounds = 0;
  }
};

struct Atom_FF //Atom definitions, epsilon, sigma, charge//
{
  std::string Name;
  double epsilon;
  double sigma;
  bool   shift = false;
  bool   tail  = false;
};

struct Input_Container
{
  std::vector<Atom_FF>AtomFF;
  std::vector<double>Mix_Epsilon;
  std::vector<double>Mix_Sigma;
  std::vector<double>Mix_Shift;
  std::vector<Tail>Mix_Tail; //See Tail Struct: a bool and a double//
  std::vector<double>Mix_Z;
  std::vector<int>Mix_Type; //Forcefield types, e.g. LJ//
  double CutOffVDW = 12.0 * 12.0;
  double CutOffCoul= 12.0 * 12.0;
  bool    VDWRealBias = true; //By default, the CBMC moves use VDW + Real Biasing//
  double  OverlapCriteria;

  bool    noCharges;
  double  EwaldPrecision = 1e-6;
};

struct Variables
{
  //Some other important variables/keywords for input//
  int NumberOfInitializationCycles = 0;
  int NumberOfEquilibrationCycles  = 0;
  int NumberOfProductionCycles     = 0;

  size_t StructureFactor_Multiplier = 2; //Add extra structure factor storage for volume moves//

  size_t Allocate_space_Adsorbate = 0;

  size_t MaxStepPerCycle = 0;
  bool SetMaxStep = false;
  int SimulationMode = INITIALIZATION;
  //Simulation Structs//
  Units Constants; //Physical constants for the simulation//

  Input_Container Input;

  ForceField FF;
  std::vector<double>TEST;
  std::vector<std::vector<double>>Ttwo;
  void set_TEST(const std::vector<double>& new_data) 
  {
    TEST = new_data;
  }
  const std::vector<double>& get_TEST() const
  {
    return TEST;
  }
  ForceField device_FF;
  PseudoAtomDefinitions PseudoAtoms;
  Simulations* Sims;
  RandomNumber Random;
  Components TempComponents; //template components
  WidomStruct TempWidom;     //template widom
  std::vector<Components> SystemComponents; //Propagate from TempComponents;
  std::vector<WidomStruct> Widom;           //Propagate from TempWidom;
  std::vector<Boxsize>Box;

  Gibbs GibbsStatistics;  //Gibbs Volume + Xfer moves stats
};

/*
//PYBIND11 STUFF//
template<typename T>
py::list Convert_Pointer_To_PyList(size_t size, T* point)
{
  py::list temp;
  for(size_t i = 0; i < size; i++) temp.append(point[i]);
  return temp;
}
*/
