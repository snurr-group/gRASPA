#include <complex>
#include <vector>
#include <thrust/complex.h>
#include <thrust/device_vector.h>
//#include <numbers>
#define kx_max_unsigned 8
#define ky_max_unsigned 8
#define kz_max_unsigned 8
#define kx_max 8
#define ky_max 8
#define kz_max 8
#define M_PI           3.14159265358979323846
void matrix_multiply_by_vector(double* a, double* b, double* c) //3x3(9*1) matrix (a) times 3x1(3*1) vector (b), a*b=c//
{
  c[0]=a[0*3+0]*b[0]+a[1*3+0]*b[1]+a[2*3+0]*b[2];
  c[1]=a[0*3+1]*b[0]+a[1*3+1]*b[1]+a[2*3+1]*b[2];
  c[2]=a[0*3+2]*b[0]+a[1*3+2]*b[1]+a[2*3+2]*b[2];
}

void JUST_Ewald_PBC_CPU(double* posvec, double* Cell, double* InverseCell, int* OtherParams)
{
  switch (OtherParams[0])//cubic/cuboid
      {
      case 0:
      {
        posvec[0] = posvec[0] - static_cast<int>(posvec[0] * InverseCell[0*3+0] + ((posvec[0] >= 0.0) ? 0.5 : -0.5)) * Cell[0*3+0];
        posvec[1] = posvec[1] - static_cast<int>(posvec[1] * InverseCell[1*3+1] + ((posvec[1] >= 0.0) ? 0.5 : -0.5)) * Cell[1*3+1];
        posvec[2] = posvec[2] - static_cast<int>(posvec[2] * InverseCell[2*3+2] + ((posvec[2] >= 0.0) ? 0.5 : -0.5)) * Cell[2*3+2];
        break;
      }
      default: //regardless of shape
      {
        double s[3] = {0.0, 0.0, 0.0};
        s[0]=InverseCell[0*3+0]*posvec[0]+InverseCell[1*3+0]*posvec[1]+InverseCell[2*3+0]*posvec[2];
        s[1]=InverseCell[0*3+1]*posvec[0]+InverseCell[1*3+1]*posvec[1]+InverseCell[2*3+1]*posvec[2];
        s[2]=InverseCell[0*3+2]*posvec[0]+InverseCell[1*3+2]*posvec[1]+InverseCell[2*3+2]*posvec[2];

        s[0] -= static_cast<int>(s[0] + ((s[0] >= 0.0) ? 0.5 : -0.5));
        s[1] -= static_cast<int>(s[1] + ((s[1] >= 0.0) ? 0.5 : -0.5));
        s[2] -= static_cast<int>(s[2] + ((s[2] >= 0.0) ? 0.5 : -0.5));
        // convert from abc to xyz
        posvec[0]=Cell[0*3+0]*s[0]+Cell[1*3+0]*s[1]+Cell[2*3+0]*s[2];
        posvec[1]=Cell[0*3+1]*s[0]+Cell[1*3+1]*s[1]+Cell[2*3+1]*s[2];
        posvec[2]=Cell[0*3+2]*s[0]+Cell[1*3+2]*s[1]+Cell[2*3+2]*s[2];
        break;
      }
      }

}

inline void get_component_thread(size_t i, Atoms* System, Components SystemComponents, size_t *component_i, size_t *position_i)
{
  size_t comp = 0; size_t posi = i; size_t totalsize= 0;
  for(size_t ijk = 0; ijk < SystemComponents.Total_Components; ijk++)
  {
    totalsize += System[ijk].size;
    if(posi >= totalsize)
    {
      comp++;
      posi -= System[ijk].size;
    }
  }
  *component_i = comp; *position_i = posi;
}

/*double Ewald_Total_DIFF(Boxsize Box, Atoms NewMol, ForceField FF, Components SystemComponents, size_t Oldsize)
{
  if(FF.noCharges) return 0.0;
  double alpha = FF.FFParams[4]; double alpha_squared = alpha * alpha;
  double prefactor = FF.FFParams[3] * (2.0 * M_PI / Box.Volume); printf("prefactor: %.10f\n", prefactor);
 
  double ewaldE = 0.0;
 
  double ax[3] = {Box.InverseCell[0], Box.InverseCell[3], Box.InverseCell[6]}; //printf("ax: %.10f, %.10f, %.10f\n", Box.InverseCell[0], Box.InverseCell[3], Box.InverseCell[6]);
  double ay[3] = {Box.InverseCell[1], Box.InverseCell[4], Box.InverseCell[7]}; //printf("ay: %.10f, %.10f, %.10f\n", Box.InverseCell[1], Box.InverseCell[4], Box.InverseCell[7]);
  double az[3] = {Box.InverseCell[2], Box.InverseCell[5], Box.InverseCell[8]}; //printf("az: %.10f, %.10f, %.10f\n", Box.InverseCell[2], Box.InverseCell[5], Box.InverseCell[8]);
  
  size_t numberOfAtoms = NewMol.size;
  size_t numberOfWaveVectors = (kx_max_unsigned + 1) * 2 * (ky_max_unsigned + 1) * 2 * (kz_max_unsigned + 1);
  std::vector<std::complex<double>>eik_x(numberOfAtoms * (kx_max_unsigned + 1));
  std::vector<std::complex<double>>eik_y(numberOfAtoms * (ky_max_unsigned + 1));
  std::vector<std::complex<double>>eik_z(numberOfAtoms * (kz_max_unsigned + 1));
  std::vector<std::complex<double>>eik_xy(numberOfAtoms);
  std::vector<std::complex<double>>storedEik(numberOfWaveVectors);
  // Construct exp(ik.r) for atoms and k-vectors kx, ky, kz = 0, 1 explicitly
  for(size_t i = 0; i != Oldsize; ++i)
  {
    //determine the component for i
    double pos[3] = {NewMol.x[i], NewMol.y[i], NewMol.z[i]};
    eik_x[i + 0 * numberOfAtoms] = std::complex<double>(1.0, 0.0);
    eik_y[i + 0 * numberOfAtoms] = std::complex<double>(1.0, 0.0);
    eik_z[i + 0 * numberOfAtoms] = std::complex<double>(1.0, 0.0);
    double s[3]; matrix_multiply_by_vector(Box.InverseCell, pos, s); for(size_t j = 0; j < 3; j++) s[j]*=2*M_PI;
    eik_x[i + 1 * numberOfAtoms] = std::complex<double>(std::cos(s[0]), std::sin(s[0]));
    eik_y[i + 1 * numberOfAtoms] = std::complex<double>(std::cos(s[1]), std::sin(s[1]));
    eik_z[i + 1 * numberOfAtoms] = std::complex<double>(std::cos(s[2]), std::sin(s[2]));
  }
  for(size_t i = Oldsize; i != NewMol.size; ++i)
  {
    //determine the component for i
    double pos[3] = {NewMol.x[i], NewMol.y[i], NewMol.z[i]};
    eik_x[i + 0 * numberOfAtoms] = std::complex<double>(1.0, 0.0);
    eik_y[i + 0 * numberOfAtoms] = std::complex<double>(1.0, 0.0);
    eik_z[i + 0 * numberOfAtoms] = std::complex<double>(1.0, 0.0);
    double s[3]; matrix_multiply_by_vector(Box.InverseCell, pos, s); for(size_t j = 0; j < 3; j++) s[j]*=2*M_PI;
    eik_x[i + 1 * numberOfAtoms] = std::complex<double>(std::cos(s[0]), std::sin(s[0]));
    eik_y[i + 1 * numberOfAtoms] = std::complex<double>(std::cos(s[1]), std::sin(s[1]));
    eik_z[i + 1 * numberOfAtoms] = std::complex<double>(std::cos(s[2]), std::sin(s[2]));
  }
  // Calculate remaining positive kx, ky and kz by recurrence
  for(size_t kx = 2; kx <= kx_max_unsigned; ++kx)
  {
    for(size_t i = 0; i != numberOfAtoms; ++i)
    {
      eik_x[i + kx * numberOfAtoms] = eik_x[i + (kx - 1) * numberOfAtoms] * eik_x[i + 1 * numberOfAtoms];
    }
  }
  for(size_t ky = 2; ky <= ky_max_unsigned; ++ky)
  {
    for(size_t i = 0; i != numberOfAtoms; ++i)
    {
      eik_y[i + ky * numberOfAtoms] = eik_y[i + (ky - 1) * numberOfAtoms] * eik_y[i + 1 * numberOfAtoms];
    }
  }
  for(size_t kz = 2; kz <= kz_max_unsigned; ++kz)
  {
    for(size_t i = 0; i != numberOfAtoms; ++i)
    {
      eik_z[i + kz * numberOfAtoms] = eik_z[i + (kz - 1) * numberOfAtoms] * eik_z[i + 1 * numberOfAtoms];
    }
  }
  //for(size_t i = 0; i < eik_x.size(); i++) printf("eik_x[%zu]: (%.10f, %.10f)\n", i, eik_x[i].real(), eik_x[i].imag());
  //for(size_t i = 0; i < eik_y.size(); i++) printf("eik_y[%zu]: (%.10f, %.10f)\n", i, eik_y[i].real(), eik_y[i].imag());
  //for(size_t i = 0; i < eik_z.size(); i++) printf("eik_z[%zu]: (%.10f, %.10f)\n", i, eik_z[i].real(), eik_z[i].imag());
  size_t nvec = 0;
  //for debugging
  size_t kxcount = 0; size_t kycount = 0; size_t kzcount = 0; size_t kzinactive = 0;
  for(std::make_signed_t<std::size_t> kx = 0; kx <= kx_max; ++kx)
  {
    double kvec_x[3]; for(size_t j = 0; j < 3; j++) kvec_x[j] = 2.0 * M_PI * static_cast<double>(kx) * ax[j];
    // Only positive kx are used, the negative kx are taken into account by the factor of two
    double factor = (kx == 0) ? (1.0 * prefactor) : (2.0 * prefactor);
    //printf("kx: %zu, kvec_x is %.10f, %.10f, %.10f, factor: %.10f\n", kx, kvec_x[0], kvec_x[1], kvec_x[2], factor);
    
    for(std::make_signed_t<std::size_t> ky = -ky_max; ky <= ky_max; ++ky)
    {
      double kvec_y[3]; for(size_t j = 0; j < 3; j++) kvec_y[j] = 2.0 * M_PI * static_cast<double>(ky) * ay[j];
      // Precompute and store eik_x * eik_y outside the kz-loop
      for(size_t i = 0; i != numberOfAtoms; ++i)
      {
        std::complex<double> eiky_temp = eik_y[i + numberOfAtoms * static_cast<size_t>(std::abs(ky))];
        eiky_temp.imag(ky>=0 ? eiky_temp.imag() : -eiky_temp.imag());
        eik_xy[i] = eik_x[i + numberOfAtoms * static_cast<size_t>(kx)] * eiky_temp;
      }

      for(std::make_signed_t<std::size_t> kz = -kz_max; kz <= kz_max; ++kz)
      {
        // Ommit kvec==0
        if((kx * kx + ky * ky + kz * kz) != 0)
        {
          double kvec_z[3]; for(size_t j = 0; j < 3; j++) kvec_z[j] = 2.0 * M_PI * static_cast<double>(kz) * az[j];
          std::complex<double> cksum(0.0, 0.0);
          std::complex<double> cksum_old(0.0,0.0);
          for(size_t i = 0; i != Oldsize; ++i)
          {
            std::complex<double> eikz_temp = eik_z[i + numberOfAtoms * static_cast<size_t>(std::abs(kz))];
            eikz_temp.imag(kz>=0 ? eikz_temp.imag() : -eikz_temp.imag());
            size_t comp = 0; size_t posi = 0; get_component_thread(i, Host_System, SystemComponents, &comp, &posi);
            double charge = Host_System[comp].charge[posi];
            double scaling = Host_System[comp].scaleCoul[posi];
            cksum_old += scaling * charge * (eik_xy[i] * eikz_temp);
            std::complex<double> tempval(0.0, 0.0); tempval = scaling * charge * (eik_xy[i] * eikz_temp);
            //printf("kxyz: %d, %d, %d, i: %lu, val: (%.10f, %.10f)\n", (int) kx, (int) ky, (int) kz, i, tempval.real(), tempval.imag());
          }
          for(size_t i = Oldsize; i != NewMol.size; i++)
          {
            std::complex<double> eikz_temp = eik_z[i + numberOfAtoms * static_cast<size_t>(std::abs(kz))];
            eikz_temp.imag(kz>=0 ? eikz_temp.imag() : -eikz_temp.imag());
            size_t comp = 0; size_t posi = 0; get_component_thread(i, Host_System, SystemComponents, &comp, &posi);
            double charge = Host_System[comp].charge[posi];
            double scaling = Host_System[comp].scaleCoul[posi];
            cksum += scaling * charge * (eik_xy[i] * eikz_temp);
            std::complex<double> tempval(0.0, 0.0); tempval = scaling * charge * (eik_xy[i] * eikz_temp);
          }
          //double rksq = (kvec_x + kvec_y + kvec_z).length_squared();
          double tempkvec[3] = {kvec_x[0]+kvec_y[0]+kvec_z[0], kvec_x[1]+kvec_y[1]+kvec_z[1], kvec_x[2]+kvec_y[2]+kvec_z[2]};
          double rksq = tempkvec[0]*tempkvec[0] + tempkvec[1]*tempkvec[1] + tempkvec[2]*tempkvec[2];
          double temp = factor * std::exp((-0.25 / alpha_squared) * rksq) / rksq;
          ewaldE += std::norm(storedEik[nvec] + cksum - cksum_old);
          ewaldE -= temp * std::norm(storedEik[nvec]);
          //double tempsum = temp * (cksum.real() * cksum.real() + cksum.imag() * cksum.imag());
          ewaldE += tempsum;
          //printf("outer sum: kxyz: %d, %d, %d, cksum: (%.10f, %.10f), %.10f, %.10f\n", static_cast<int> (kx), static_cast<int> (ky), static_cast<int> (kz),cksum.real(), cksum.imag(), temp, tempsum);
          storedEik[nvec] = cksum;
          ++nvec;
          kzcount++;
        }
        kzinactive++;
      }
      kycount++;
    }
    kxcount++;
  }

  //printf("nvec is %zu, kxyz count: %zd %zd %zd, inactive: %zu, Pre-subtract-Energy: %.10f\n", nvec, kxcount, kycount, kzcount, kzinactive, ewaldE);

  // Subtract self-energy
  double prefactor_self = FF.FFParams[3] * alpha / std::sqrt(M_PI);
  for(size_t i = 0; i != numberOfAtoms; ++i)
  {
    size_t comp = 0; size_t posi = 0; get_component_thread(i, Host_System, SystemComponents, &comp, &posi);
    double charge = Host_System[comp].charge[posi];
    double scaling = Host_System[comp].scaleCoul[posi];
    ewaldE -= prefactor_self * scaling * charge * scaling * charge;
  }
  //printf("ewaldE: %.10f\n", ewaldE);


  // Subtract exclusion-energy, Zhao's note: taking out the pairs of energies that belong to the same molecule
  size_t j_count = 0;
  for(size_t l = 0; l != SystemComponents.Total_Components; ++l)
  {
      for(size_t i = 0; i != Host_System[l].size - 1; i++)
      {
        double factorA = Host_System[l].scaleCoul[i] * Host_System[l].charge[i];
        double posA[3] = {Host_System[l].x[i], Host_System[l].y[i], Host_System[l].z[i]};
        for(size_t j = i + 1; j != Host_System[l].size; j++)
        {
          double factorB = Host_System[l].scaleCoul[j] * Host_System[l].charge[j];
          double posB[3] = {Host_System[l].x[j], Host_System[l].y[j], Host_System[l].z[j]};

          double posvec[3] = {posA[0]-posB[0], posA[1]-posB[1], posA[2]-posB[2]};
          JUST_Ewald_PBC_CPU(posvec, Box.Cell, Box.InverseCell, FF.OtherParams);
          double rr_dot = posvec[0]*posvec[0] + posvec[1]*posvec[1] + posvec[2]*posvec[2];
          double r = std::sqrt(rr_dot);

          ewaldE -= FF.FFParams[3] * factorA * factorB * std::erf(alpha * r) / r;
          j_count++;
        }
      }
  }
  printf("j_count: %zu, Post-subtract-Energy: %.10f\n", j_count, ewaldE);

  // Handle net-charges
  for(size_t i = 0; i != components.size(); ++i)
  {
    for(size_t j = 0; j != components.size(); ++j)
    {
      energyStatus.ewald += CoulombicFourierEnergySingleIon * netCharge[i] * netCharge[j];
    }
  }
  return ewaldE;
}
*/

double Ewald_Total(Boxsize Box, Atoms* Host_System, ForceField FF, Components SystemComponents)
{
  if(FF.noCharges) return 0.0;
  double alpha = FF.FFParams[4]; double alpha_squared = alpha * alpha;
  double prefactor = FF.FFParams[3] * (2.0 * M_PI / Box.Volume); printf("prefactor: %.10f\n", prefactor);
 
  double ewaldE = 0.0;
 
  double ax[3] = {Box.InverseCell[0], Box.InverseCell[3], Box.InverseCell[6]}; //printf("ax: %.10f, %.10f, %.10f\n", Box.InverseCell[0], Box.InverseCell[3], Box.InverseCell[6]);
  double ay[3] = {Box.InverseCell[1], Box.InverseCell[4], Box.InverseCell[7]}; //printf("ay: %.10f, %.10f, %.10f\n", Box.InverseCell[1], Box.InverseCell[4], Box.InverseCell[7]);
  double az[3] = {Box.InverseCell[2], Box.InverseCell[5], Box.InverseCell[8]}; //printf("az: %.10f, %.10f, %.10f\n", Box.InverseCell[2], Box.InverseCell[5], Box.InverseCell[8]);
  
  size_t numberOfAtoms = 0;
  for(size_t i=0; i < SystemComponents.Total_Components; i++) //Skip the first one(framework)
  {
    numberOfAtoms  += SystemComponents.Moleculesize[i] * SystemComponents.NumberOfMolecule_for_Component[i];
  }
  size_t numberOfWaveVectors = (kx_max_unsigned + 1) * 2 * (ky_max_unsigned + 1) * 2 * (kz_max_unsigned + 1);
  std::vector<std::complex<double>>eik_x(numberOfAtoms * (kx_max_unsigned + 1));
  std::vector<std::complex<double>>eik_y(numberOfAtoms * (ky_max_unsigned + 1));
  std::vector<std::complex<double>>eik_z(numberOfAtoms * (kz_max_unsigned + 1));
  std::vector<std::complex<double>>eik_xy(numberOfAtoms);
  std::vector<std::complex<double>>storedEik(numberOfWaveVectors);
  // Construct exp(ik.r) for atoms and k-vectors kx, ky, kz = 0, 1 explicitly
  for(size_t i = 0; i != numberOfAtoms; ++i)
  {
    //determine the component for i
    size_t comp = 0; size_t posi = 0; get_component_thread(i, Host_System, SystemComponents, &comp, &posi);
    double pos[3] = {Host_System[comp].x[posi], Host_System[comp].y[posi], Host_System[comp].z[posi]};
    eik_x[i + 0 * numberOfAtoms] = std::complex<double>(1.0, 0.0);
    eik_y[i + 0 * numberOfAtoms] = std::complex<double>(1.0, 0.0);
    eik_z[i + 0 * numberOfAtoms] = std::complex<double>(1.0, 0.0);
    double s[3]; matrix_multiply_by_vector(Box.InverseCell, pos, s); for(size_t j = 0; j < 3; j++) s[j]*=2*M_PI;
    eik_x[i + 1 * numberOfAtoms] = std::complex<double>(std::cos(s[0]), std::sin(s[0]));
    eik_y[i + 1 * numberOfAtoms] = std::complex<double>(std::cos(s[1]), std::sin(s[1]));
    eik_z[i + 1 * numberOfAtoms] = std::complex<double>(std::cos(s[2]), std::sin(s[2]));
  }
  // Calculate remaining positive kx, ky and kz by recurrence
  for(size_t kx = 2; kx <= kx_max_unsigned; ++kx)
  {
    for(size_t i = 0; i != numberOfAtoms; ++i)
    {
      eik_x[i + kx * numberOfAtoms] = eik_x[i + (kx - 1) * numberOfAtoms] * eik_x[i + 1 * numberOfAtoms];
    }
  }
  for(size_t ky = 2; ky <= ky_max_unsigned; ++ky)
  {
    for(size_t i = 0; i != numberOfAtoms; ++i)
    {
      eik_y[i + ky * numberOfAtoms] = eik_y[i + (ky - 1) * numberOfAtoms] * eik_y[i + 1 * numberOfAtoms];
    }
  }
  for(size_t kz = 2; kz <= kz_max_unsigned; ++kz)
  {
    for(size_t i = 0; i != numberOfAtoms; ++i)
    {
      eik_z[i + kz * numberOfAtoms] = eik_z[i + (kz - 1) * numberOfAtoms] * eik_z[i + 1 * numberOfAtoms];
    }
  }
  //for(size_t i = 0; i < eik_x.size(); i++) printf("eik_x[%zu]: (%.10f, %.10f)\n", i, eik_x[i].real(), eik_x[i].imag());
  //for(size_t i = 0; i < eik_y.size(); i++) printf("eik_y[%zu]: (%.10f, %.10f)\n", i, eik_y[i].real(), eik_y[i].imag());
  //for(size_t i = 0; i < eik_z.size(); i++) printf("eik_z[%zu]: (%.10f, %.10f)\n", i, eik_z[i].real(), eik_z[i].imag());
  size_t nvec = 0;
  //for debugging
  size_t kxcount = 0; size_t kycount = 0; size_t kzcount = 0; size_t kzinactive = 0;
  for(std::make_signed_t<std::size_t> kx = 0; kx <= kx_max; ++kx)
  {
    double kvec_x[3]; for(size_t j = 0; j < 3; j++) kvec_x[j] = 2.0 * M_PI * static_cast<double>(kx) * ax[j];
    // Only positive kx are used, the negative kx are taken into account by the factor of two
    double factor = (kx == 0) ? (1.0 * prefactor) : (2.0 * prefactor);
    //printf("kx: %zu, kvec_x is %.10f, %.10f, %.10f, factor: %.10f\n", kx, kvec_x[0], kvec_x[1], kvec_x[2], factor);
    
    for(std::make_signed_t<std::size_t> ky = -ky_max; ky <= ky_max; ++ky)
    {
      double kvec_y[3]; for(size_t j = 0; j < 3; j++) kvec_y[j] = 2.0 * M_PI * static_cast<double>(ky) * ay[j];
      // Precompute and store eik_x * eik_y outside the kz-loop
      for(size_t i = 0; i != numberOfAtoms; ++i)
      {
        std::complex<double> eiky_temp = eik_y[i + numberOfAtoms * static_cast<size_t>(std::abs(ky))];
        eiky_temp.imag(ky>=0 ? eiky_temp.imag() : -eiky_temp.imag());
        eik_xy[i] = eik_x[i + numberOfAtoms * static_cast<size_t>(kx)] * eiky_temp;
      }

      for(std::make_signed_t<std::size_t> kz = -kz_max; kz <= kz_max; ++kz)
      {
        // Ommit kvec==0
        if((kx * kx + ky * ky + kz * kz) != 0)
        {
          double kvec_z[3]; for(size_t j = 0; j < 3; j++) kvec_z[j] = 2.0 * M_PI * static_cast<double>(kz) * az[j];
          std::complex<double> cksum(0.0, 0.0);
          for(size_t i = 0; i != numberOfAtoms; ++i)
          {
            std::complex<double> eikz_temp = eik_z[i + numberOfAtoms * static_cast<size_t>(std::abs(kz))];
            eikz_temp.imag(kz>=0 ? eikz_temp.imag() : -eikz_temp.imag());
            size_t comp = 0; size_t posi = 0; get_component_thread(i, Host_System, SystemComponents, &comp, &posi);
            double charge = Host_System[comp].charge[posi];
            double scaling = Host_System[comp].scaleCoul[posi];
            cksum += scaling * charge * (eik_xy[i] * eikz_temp);
            std::complex<double> tempval(0.0, 0.0); tempval = scaling * charge * (eik_xy[i] * eikz_temp);
            //printf("kxyz: %d, %d, %d, i: %lu, val: (%.10f, %.10f)\n", (int) kx, (int) ky, (int) kz, i, tempval.real(), tempval.imag());
          }
          //double rksq = (kvec_x + kvec_y + kvec_z).length_squared();
          double tempkvec[3] = {kvec_x[0]+kvec_y[0]+kvec_z[0], kvec_x[1]+kvec_y[1]+kvec_z[1], kvec_x[2]+kvec_y[2]+kvec_z[2]};
          double rksq = tempkvec[0]*tempkvec[0] + tempkvec[1]*tempkvec[1] + tempkvec[2]*tempkvec[2];
          double temp = factor * std::exp((-0.25 / alpha_squared) * rksq) / rksq;
          double tempsum = temp * (cksum.real() * cksum.real() + cksum.imag() * cksum.imag());
          ewaldE += tempsum;
          //printf("outer sum: kxyz: %d, %d, %d, cksum: (%.10f, %.10f), %.10f, %.10f\n", static_cast<int> (kx), static_cast<int> (ky), static_cast<int> (kz),cksum.real(), cksum.imag(), temp, tempsum);
          storedEik[nvec] = cksum;
          ++nvec;
          kzcount++;
        }
        kzinactive++;
      }
      kycount++;
    }
    kxcount++;
  }

  //printf("nvec is %zu, kxyz count: %zd %zd %zd, inactive: %zu, Pre-subtract-Energy: %.10f\n", nvec, kxcount, kycount, kzcount, kzinactive, ewaldE);

  // Subtract self-energy
  double prefactor_self = FF.FFParams[3] * alpha / std::sqrt(M_PI);
  for(size_t i = 0; i != numberOfAtoms; ++i)
  {
    size_t comp = 0; size_t posi = 0; get_component_thread(i, Host_System, SystemComponents, &comp, &posi);
    double charge = Host_System[comp].charge[posi];
    double scaling = Host_System[comp].scaleCoul[posi];
    ewaldE -= prefactor_self * scaling * charge * scaling * charge;
  }
  //printf("ewaldE: %.10f\n", ewaldE);


  // Subtract exclusion-energy, Zhao's note: taking out the pairs of energies that belong to the same molecule
  size_t j_count = 0;
  for(size_t l = 0; l != SystemComponents.Total_Components; ++l)
  {
      for(size_t i = 0; i != Host_System[l].size - 1; i++)
      {
        double factorA = Host_System[l].scaleCoul[i] * Host_System[l].charge[i];
        double posA[3] = {Host_System[l].x[i], Host_System[l].y[i], Host_System[l].z[i]};
        for(size_t j = i + 1; j != Host_System[l].size; j++)
        {
          double factorB = Host_System[l].scaleCoul[j] * Host_System[l].charge[j];
          double posB[3] = {Host_System[l].x[j], Host_System[l].y[j], Host_System[l].z[j]};

          double posvec[3] = {posA[0]-posB[0], posA[1]-posB[1], posA[2]-posB[2]};
          JUST_Ewald_PBC_CPU(posvec, Box.Cell, Box.InverseCell, FF.OtherParams);
          double rr_dot = posvec[0]*posvec[0] + posvec[1]*posvec[1] + posvec[2]*posvec[2];
          double r = std::sqrt(rr_dot);

          ewaldE -= FF.FFParams[3] * factorA * factorB * std::erf(alpha * r) / r;
          j_count++;
        }
      }
  }
  printf("j_count: %zu, Post-subtract-Energy: %.10f\n", j_count, ewaldE);

/*  // Handle net-charges
  for(size_t i = 0; i != components.size(); ++i)
  {
    for(size_t j = 0; j != components.size(); ++j)
    {
      energyStatus.ewald += CoulombicFourierEnergySingleIon * netCharge[i] * netCharge[j];
    }
  }*/
  return ewaldE;
}

__device__ void GPU_matrix_multiply_by_vector(double* a, double* b, double* c) //3x3(9*1) matrix (a) times 3x1(3*1) vector (b), a*b=c//
{
  c[0]=a[0*3+0]*b[0]+a[1*3+0]*b[1]+a[2*3+0]*b[2];
  c[1]=a[0*3+1]*b[0]+a[1*3+1]*b[1]+a[2*3+1]*b[2];
  c[2]=a[0*3+2]*b[0]+a[1*3+2]*b[1]+a[2*3+2]*b[2];
}

__device__ void GPU_JUST_Ewald_PBC(double* posvec, double* Cell, double* InverseCell, int* OtherParams)
{
  switch (OtherParams[0])//cubic/cuboid
  {
  case 0:
  {
    posvec[0] = posvec[0] - static_cast<int>(posvec[0] * InverseCell[0*3+0] + ((posvec[0] >= 0.0) ? 0.5 : -0.5)) * Cell[0*3+0];
    posvec[1] = posvec[1] - static_cast<int>(posvec[1] * InverseCell[1*3+1] + ((posvec[1] >= 0.0) ? 0.5 : -0.5)) * Cell[1*3+1];
    posvec[2] = posvec[2] - static_cast<int>(posvec[2] * InverseCell[2*3+2] + ((posvec[2] >= 0.0) ? 0.5 : -0.5)) * Cell[2*3+2];
    break;
  }
  default: //regardless of shape
  {
    double s[3] = {0.0, 0.0, 0.0};
    s[0]=InverseCell[0*3+0]*posvec[0]+InverseCell[1*3+0]*posvec[1]+InverseCell[2*3+0]*posvec[2];
    s[1]=InverseCell[0*3+1]*posvec[0]+InverseCell[1*3+1]*posvec[1]+InverseCell[2*3+1]*posvec[2];
    s[2]=InverseCell[0*3+2]*posvec[0]+InverseCell[1*3+2]*posvec[1]+InverseCell[2*3+2]*posvec[2];

    s[0] -= static_cast<int>(s[0] + ((s[0] >= 0.0) ? 0.5 : -0.5));
    s[1] -= static_cast<int>(s[1] + ((s[1] >= 0.0) ? 0.5 : -0.5));
    s[2] -= static_cast<int>(s[2] + ((s[2] >= 0.0) ? 0.5 : -0.5));
    // convert from abc to xyz
    posvec[0]=Cell[0*3+0]*s[0]+Cell[1*3+0]*s[1]+Cell[2*3+0]*s[2];
    posvec[1]=Cell[0*3+1]*s[0]+Cell[1*3+1]*s[1]+Cell[2*3+1]*s[2];
    posvec[2]=Cell[0*3+2]*s[0]+Cell[1*3+2]*s[1]+Cell[2*3+2]*s[2];
    break;
  }
  }
}

__device__ void GPU_get_component_thread(size_t i, Atoms* System, size_t NumberOfComponents, size_t *component_i, size_t *position_i)
{
  size_t comp = 0; size_t posi = i; size_t totalsize= 0;
  for(size_t ijk = 0; ijk < NumberOfComponents; ijk++)
  {
    totalsize += System[ijk].size;
    if(posi >= totalsize)
    {
      comp++;
      posi -= System[ijk].size;
    }
  }
  *component_i = comp; *position_i = posi;
}

__device__ Complex multiply(Complex a, Complex b) //a*b = c for complex numbers//
{
  Complex c;
  c.real = a.real*b.real - a.imag*b.imag;
  c.imag = a.real*b.imag + a.imag*b.real;
  return c;
}

Complex* CUDA_allocate_Complex_array(size_t N)
{
  Complex* device_x;
  cudaMalloc(&device_x, N * sizeof(Complex)); checkCUDAError("Error allocating Malloc");
  //cudaMemcpy(device_x, x, N * sizeof(Complex), cudaMemcpyHostToDevice); checkCUDAError("size_t Error Memcpy");
  return device_x;
}

void GPU_ewald_alloc(Boxsize& Box, Boxsize& device_Box, Atoms* System, ForceField FF, size_t NumberOfComponents)
{
  if(FF.noCharges) return;
  double alpha = FF.FFParams[4]; double alpha_squared = alpha * alpha;
  double prefactor = FF.FFParams[3] * (2.0 * M_PI / Box.Volume); printf("prefactor: %.10f\n", prefactor);
  size_t numberOfAtoms = 0;
  for(size_t i=0; i < NumberOfComponents; i++) //Skip the first one(framework)
  {
    numberOfAtoms  += System[i].size;
  }
  //printf("numberOfAtoms: %lu\n", numberOfAtoms);
  size_t numberOfWaveVectors = (kx_max_unsigned + 1) * 2 * (ky_max_unsigned + 1) * 2 * (kz_max_unsigned + 1);
  size_t xsize = numberOfAtoms * (kx_max_unsigned + 1); //Complex eik_x[xsize];
  size_t ysize = numberOfAtoms * (ky_max_unsigned + 1); //Complex eik_y[ysize];
  size_t zsize = numberOfAtoms * (kz_max_unsigned + 1); //Complex eik_z[zsize];
  //Complex eik_xy[numberOfAtoms];
  //Complex storedEik[numberOfWaveVectors];
  device_Box.eik_x     = CUDA_allocate_Complex_array(xsize);
  device_Box.eik_y     = CUDA_allocate_Complex_array(ysize);
  device_Box.eik_z     = CUDA_allocate_Complex_array(zsize);
  device_Box.eik_xy    = CUDA_allocate_Complex_array(numberOfAtoms);
  device_Box.storedEik = CUDA_allocate_Complex_array(numberOfWaveVectors);
  //Allocated sizes of the ewald vectors//
  device_Box.alloc_x_size  = xsize; device_Box.alloc_y_size = ysize; device_Box.alloc_z_size = zsize; 
  device_Box.alloc_xy_size = numberOfAtoms; device_Box.alloc_wavevector = numberOfWaveVectors;
}

__global__ void Setup_Ewald_Vector(Boxsize Box, Atoms* System, size_t numberOfAtoms, size_t NumberOfComponents)
{
    // Construct exp(ik.r) for atoms and k-vectors kx, ky, kz = 0, 1 explicitly
  for(size_t i = 0; i != numberOfAtoms; ++i)
  {
    //determine the component for i
    Complex tempcomplex; tempcomplex.real = 1.0; tempcomplex.imag = 0.0;
    size_t comp = 0; size_t posi = 0; GPU_get_component_thread(i, System, NumberOfComponents, &comp, &posi);
    double pos[3] = {System[comp].x[posi], System[comp].y[posi],System[comp].z[posi]};
    Box.eik_x[i + 0 * numberOfAtoms] = tempcomplex;
    Box.eik_y[i + 0 * numberOfAtoms] = tempcomplex;
    Box.eik_z[i + 0 * numberOfAtoms] = tempcomplex;
    double s[3]; GPU_matrix_multiply_by_vector(Box.InverseCell, pos, s); for(size_t j = 0; j < 3; j++) s[j]*=2*M_PI;
    tempcomplex.real = std::cos(s[0]); tempcomplex.imag = std::sin(s[0]); Box.eik_x[i + 1 * numberOfAtoms] = tempcomplex;
    tempcomplex.real = std::cos(s[1]); tempcomplex.imag = std::sin(s[1]); Box.eik_y[i + 1 * numberOfAtoms] = tempcomplex;
    tempcomplex.real = std::cos(s[2]); tempcomplex.imag = std::sin(s[2]); Box.eik_z[i + 1 * numberOfAtoms] = tempcomplex;
  }
  // Calculate remaining positive kx, ky and kz by recurrence
  for(size_t kx = 2; kx <= kx_max_unsigned; ++kx)
  {
    for(size_t i = 0; i != numberOfAtoms; ++i)
    {
      Box.eik_x[i + kx * numberOfAtoms] = multiply(Box.eik_x[i + (kx - 1) * numberOfAtoms], Box.eik_x[i + 1 * numberOfAtoms]);
    }
  }
  for(size_t ky = 2; ky <= ky_max_unsigned; ++ky)
  {
    for(size_t i = 0; i != numberOfAtoms; ++i)
    {
      Box.eik_y[i + ky * numberOfAtoms] = multiply(Box.eik_y[i + (ky - 1) * numberOfAtoms], Box.eik_y[i + 1 * numberOfAtoms]);
    }
  }
  for(size_t kz = 2; kz <= kz_max_unsigned; ++kz)
  {
    for(size_t i = 0; i != numberOfAtoms; ++i)
    {
      Box.eik_z[i + kz * numberOfAtoms] = multiply(Box.eik_z[i + (kz - 1) * numberOfAtoms], Box.eik_z[i + 1 * numberOfAtoms]);
    }
  }
}

__global__ void GPU_Ewald_Total_Box(Boxsize Box, Atoms* System, ForceField FF, size_t NumberOfComponents, Complex* ewaldE, size_t NumberOfAtoms)
{
  size_t threadID = blockIdx.x * blockDim.x + threadIdx.x;
  ewaldE[threadID].real = 0.0; ewaldE[threadID].imag = 0.0;
  //divide threads into kx, ky, kz, and NumberOfAtoms//
  //kx: from 0 to kx_max; ky: from -ky_max to ky_max; kz: from -kz_max to kz_max;//
  size_t total_kpoints = (kx_max_unsigned + 1) * (2 * ky_max_unsigned + 1) * (2 * kz_max_unsigned + 1);
  if(threadID < NumberOfAtoms * total_kpoints)
  {
  size_t AtomID = threadID%NumberOfAtoms;
  size_t kxyz = threadID/NumberOfAtoms;
  int kz = kxyz%(2 * kz_max_unsigned + 1) - kz_max_unsigned;
  int kxy = kxyz/(2 * kz_max_unsigned + 1);
  int kx = kxy/(2 * ky_max_unsigned + 1); int ky = kxy%(2 * ky_max_unsigned + 1) - ky_max_unsigned;
  //printf("threadID: %lu, kx: %d, ky: %d, kz: %d, AtomID: %lu\n", threadID, kx, ky, kz, AtomID);
  if(FF.noCharges) return; 
  double alpha = FF.FFParams[4]; double alpha_squared = alpha * alpha;
  double prefactor = FF.FFParams[3] * (2.0 * M_PI / Box.Volume); 
 
  double ax[3] = {Box.InverseCell[0], Box.InverseCell[3], Box.InverseCell[6]}; 
  double ay[3] = {Box.InverseCell[1], Box.InverseCell[4], Box.InverseCell[7]}; 
  double az[3] = {Box.InverseCell[2], Box.InverseCell[5], Box.InverseCell[8]}; 
  
  size_t nvec = kxyz; //Number of k_points in total
  //for debugging
  double kvec_x[3]; for(size_t j = 0; j < 3; j++) kvec_x[j] = 2.0 * M_PI * static_cast<double>(kx) * ax[j];
  // Only positive kx are used, the negative kx are taken into account by the factor of two
  double factor = (kx == 0) ? (1.0 * prefactor) : (2.0 * prefactor);
    
  double kvec_y[3]; for(size_t j = 0; j < 3; j++) kvec_y[j] = 2.0 * M_PI * static_cast<double>(ky) * ay[j];
  // Precompute and store eik_x * eik_y outside the kz-loop
  Complex eiky_temp = Box.eik_y[AtomID + NumberOfAtoms * static_cast<size_t>(std::abs(ky))];
  eiky_temp.imag = ky>=0 ? eiky_temp.imag : -eiky_temp.imag;
  Complex xy_temp = multiply(Box.eik_x[AtomID + NumberOfAtoms * static_cast<size_t>(kx)], eiky_temp);

  // Ommit kvec==0
  if((kx * kx + ky * ky + kz * kz) != 0)
  {
    double kvec_z[3]; for(size_t j = 0; j < 3; j++) kvec_z[j] = 2.0 * M_PI * static_cast<double>(kz) * az[j];
    Complex ck; 
    
    Complex eikz_temp = Box.eik_z[AtomID + NumberOfAtoms * static_cast<size_t>(std::abs(kz))];
    eikz_temp.imag = kz>=0 ? eikz_temp.imag : -eikz_temp.imag;
    size_t comp = 0; size_t posi = 0; GPU_get_component_thread(AtomID, System, NumberOfComponents, &comp, &posi);
    double charge = System[comp].charge[posi];
    double scaling = System[comp].scaleCoul[posi];
    ck = multiply(xy_temp, eikz_temp); ck.real*=scaling*charge; ck.imag*=scaling*charge;
          
    //double rksq = (kvec_x + kvec_y + kvec_z).length_squared();
    double tempkvec[3] = {kvec_x[0]+kvec_y[0]+kvec_z[0], kvec_x[1]+kvec_y[1]+kvec_z[1], kvec_x[2]+kvec_y[2]+kvec_z[2]};
    double rksq = tempkvec[0]*tempkvec[0] + tempkvec[1]*tempkvec[1] + tempkvec[2]*tempkvec[2];
    double temp = factor * std::exp((-0.25 / alpha_squared) * rksq) / rksq;
    //Take a square root of temp//
    double tempsqrt = sqrt(temp);
    //double tempsum = temp * (ck.real * ck.real + ck.imag * ck.imag);
    ewaldE[threadID].real = tempsqrt*ck.real; ewaldE[threadID].imag = tempsqrt*ck.imag;
    //printf("ThreadID: %lu, AtomID: %lu, outer sum: kxyz: %d, %d, %d, cksum: (%.10f, %.10f), %.10f, %.10f\n", threadID, AtomID, kx, ky, kz, ck.real, ck.imag, temp, tempsum);
  }
  }
}

__device__ void Determine_MoleculeID_for_Thread(Atoms* System, size_t ThreadID, size_t NumberOfComponents, size_t *ComponentID, size_t *MoleculeID, size_t *AtomID_i, size_t *AtomID_j)
{
  size_t TotalThreads = 0; size_t tempcomp = 0; size_t tempMolID = 0;
  size_t Atomi = 0; size_t Atomj = 0;
  for(size_t i = 0; i < NumberOfComponents; i++)
  {
    //Determine the number of molecules in this component
    size_t NumMol = System[i].size/System[i].Molsize;
    //For this component, determine the number of threads the molecule in this component needs//
    //For a molecule of size 5, it needs 5*4*3*2*1 threads//
    size_t MolThreads = System[i].Molsize * (System[i].Molsize + 1)/2;
    if((TotalThreads + MolThreads*NumMol) > ThreadID)
    {
      tempMolID = (ThreadID - TotalThreads) / MolThreads;
      tempcomp = i;
      //printf("threadID: %lu, total: %lu, comp: %lu, MolID: %lu\n", ThreadID, TotalThreads, i, tempMolID);
      break;
    }
    TotalThreads += MolThreads;
  }
  //https://stackoverflow.com/questions/4803180/mapping-elements-in-2d-upper-triangle-and-lower-triangle-to-linear-structure
  size_t index = ThreadID - TotalThreads;
  Atomj = (size_t)((-1+sqrt(8*index+1))/2);
  Atomi = index - Atomj*(Atomj+1)/2;
  *AtomID_i = Atomi + tempMolID * System[tempcomp].Molsize; *AtomID_j = Atomj + tempMolID * System[tempcomp].Molsize;
  *ComponentID = tempcomp; *MoleculeID = tempMolID;
}

__global__ void Ewald_Total_Exclusion_Energy(Boxsize Box, Atoms* System, size_t NumberOfComponents, size_t NumberOfAtoms, ForceField FF, double* EwaldE)
{
  size_t threadID = blockIdx.x * blockDim.x + threadIdx.x;
  EwaldE[threadID] = 0.0;
  double alpha = FF.FFParams[4];

  //Determine the indices: component, MoleculeID, AtomIDs//
  size_t comp = 0; size_t MolID = 0; size_t AtomID_i = 0; size_t AtomID_j = 0;
  Determine_MoleculeID_for_Thread(System, threadID, NumberOfComponents, &comp, &MolID, &AtomID_i, &AtomID_j);
  // Subtract exclusion-energy, Zhao's note: taking out the pairs of energies that belong to the same molecule
  double factorA = System[comp].scaleCoul[AtomID_i] * System[comp].charge[AtomID_i];
  double factorB = System[comp].scaleCoul[AtomID_j] * System[comp].charge[AtomID_j];

  if(AtomID_i == AtomID_j)
  {
    double prefactor_self = FF.FFParams[3] * alpha / std::sqrt(M_PI);
    EwaldE[threadID] = (-1.0) * prefactor_self * factorA * factorB;
  }
  else
  {
    double posA[3] = {System[comp].x[AtomID_i], System[comp].y[AtomID_i], System[comp].z[AtomID_i]};
    double posB[3] = {System[comp].x[AtomID_j], System[comp].y[AtomID_j], System[comp].z[AtomID_j]};
    double posvec[3] = {posA[0]-posB[0], posA[1]-posB[1], posA[2]-posB[2]};
    GPU_JUST_Ewald_PBC(posvec, Box.Cell, Box.InverseCell, FF.OtherParams);
    double rr_dot = posvec[0]*posvec[0] + posvec[1]*posvec[1] + posvec[2]*posvec[2];
    double r = std::sqrt(rr_dot);
    EwaldE[threadID] = (-1.0) * FF.FFParams[3] * factorA * factorB * std::erf(alpha * r) / r;
  }
  //printf("thread: %lu, component: %lu, Molecule: %lu, Atom_i: %lu, Atom_j: %lu\n", threadID, comp, MolID, AtomID_i, AtomID_j);
}
/*  // Handle net-charges
  for(size_t i = 0; i != components.size(); ++i)
  {
    for(size_t j = 0; j != components.size(); ++j)
    {
      energyStatus.ewald += CoulombicFourierEnergySingleIon * netCharge[i] * netCharge[j];
    }
  }
*/

__global__ void GPU_Ewald_Total_singlethread(Boxsize Box, Atoms* System, ForceField FF, size_t NumberOfComponents)
{
  if(FF.noCharges) return; 
  double alpha = FF.FFParams[4]; double alpha_squared = alpha * alpha;
  double prefactor = FF.FFParams[3] * (2.0 * M_PI / Box.Volume); printf("prefactor: %.10f\n", prefactor);
 
  double ewaldE = 0.0;
 
  double ax[3] = {Box.InverseCell[0], Box.InverseCell[3], Box.InverseCell[6]}; //printf("ax: %.10f, %.10f, %.10f\n", Box.InverseCell[0], Box.InverseCell[3], Box.InverseCell[6]);
  double ay[3] = {Box.InverseCell[1], Box.InverseCell[4], Box.InverseCell[7]}; //printf("ay: %.10f, %.10f, %.10f\n", Box.InverseCell[1], Box.InverseCell[4], Box.InverseCell[7]);
  double az[3] = {Box.InverseCell[2], Box.InverseCell[5], Box.InverseCell[8]}; //printf("az: %.10f, %.10f, %.10f\n", Box.InverseCell[2], Box.InverseCell[5], Box.InverseCell[8]);
  
  size_t numberOfAtoms = 0;
  for(size_t i=0; i < NumberOfComponents; i++) //Skip the first one(framework)
  {
    numberOfAtoms  += System[i].size;
  }
  //printf("numberOfAtoms: %lu\n", numberOfAtoms);
  size_t numberOfWaveVectors = (kx_max_unsigned + 1) * 2 * (ky_max_unsigned + 1) * 2 * (kz_max_unsigned + 1);
  size_t xsize = numberOfAtoms * (kx_max_unsigned + 1); Complex eik_x[xsize];
  size_t ysize = numberOfAtoms * (ky_max_unsigned + 1); Complex eik_y[ysize];
  size_t zsize = numberOfAtoms * (kz_max_unsigned + 1); Complex eik_z[zsize];
  Complex eik_xy[numberOfAtoms];
  Complex storedEik[numberOfWaveVectors];
  // Construct exp(ik.r) for atoms and k-vectors kx, ky, kz = 0, 1 explicitly
  for(size_t i = 0; i != numberOfAtoms; ++i)
  {
    //determine the component for i
    Complex tempcomplex; tempcomplex.real = 1.0; tempcomplex.imag = 0.0;
    size_t comp = 0; size_t posi = 0; GPU_get_component_thread(i, System, NumberOfComponents, &comp, &posi);
    double pos[3] = {System[comp].x[posi], System[comp].y[posi],System[comp].z[posi]};
    eik_x[i + 0 * numberOfAtoms] = tempcomplex;
    eik_y[i + 0 * numberOfAtoms] = tempcomplex;
    eik_z[i + 0 * numberOfAtoms] = tempcomplex;
    double s[3]; GPU_matrix_multiply_by_vector(Box.InverseCell, pos, s); for(size_t j = 0; j < 3; j++) s[j]*=2*M_PI;
    tempcomplex.real = std::cos(s[0]); tempcomplex.imag = std::sin(s[0]); eik_x[i + 1 * numberOfAtoms] = tempcomplex;
    tempcomplex.real = std::cos(s[1]); tempcomplex.imag = std::sin(s[1]); eik_y[i + 1 * numberOfAtoms] = tempcomplex;
    tempcomplex.real = std::cos(s[2]); tempcomplex.imag = std::sin(s[2]); eik_z[i + 1 * numberOfAtoms] = tempcomplex;
  }
  // Calculate remaining positive kx, ky and kz by recurrence
  for(size_t kx = 2; kx <= kx_max_unsigned; ++kx)
  {
    for(size_t i = 0; i != numberOfAtoms; ++i)
    {
      eik_x[i + kx * numberOfAtoms] = multiply(eik_x[i + (kx - 1) * numberOfAtoms], eik_x[i + 1 * numberOfAtoms]);
    }
  }
  for(size_t ky = 2; ky <= ky_max_unsigned; ++ky)
  {
    for(size_t i = 0; i != numberOfAtoms; ++i)
    {
      eik_y[i + ky * numberOfAtoms] = multiply(eik_y[i + (ky - 1) * numberOfAtoms], eik_y[i + 1 * numberOfAtoms]);
    }
  }
  for(size_t kz = 2; kz <= kz_max_unsigned; ++kz)
  {
    for(size_t i = 0; i != numberOfAtoms; ++i)
    {
      eik_z[i + kz * numberOfAtoms] = multiply(eik_z[i + (kz - 1) * numberOfAtoms], eik_z[i + 1 * numberOfAtoms]);
    }
  }
  //for(size_t i = 0; i < xsize; i++) printf("eik_x[%lu]: (%.10f, %.10f)\n", i, eik_x[i].real, eik_x[i].imag);
  //for(size_t i = 0; i < ysize; i++) printf("eik_y[%lu]: (%.10f, %.10f)\n", i, eik_y[i].real, eik_y[i].imag);
  //for(size_t i = 0; i < zsize; i++) printf("eik_z[%lu]: (%.10f, %.10f)\n", i, eik_z[i].real, eik_z[i].imag);

  size_t nvec = 0;
  //for debugging
  size_t kxcount = 0; size_t kycount = 0; size_t kzcount = 0; size_t kzinactive = 0;
  for(int kx = 0; kx <= kx_max; ++kx)
  {
    double kvec_x[3]; for(size_t j = 0; j < 3; j++) kvec_x[j] = 2.0 * M_PI * static_cast<double>(kx) * ax[j];
    // Only positive kx are used, the negative kx are taken into account by the factor of two
    double factor = (kx == 0) ? (1.0 * prefactor) : (2.0 * prefactor);
    //printf("kx: %zu, kvec_x is %.10f, %.10f, %.10f, factor: %.10f\n", kx, kvec_x[0], kvec_x[1], kvec_x[2], factor);
    
    for(int ky = -ky_max; ky <= ky_max; ++ky)
    {
      double kvec_y[3]; for(size_t j = 0; j < 3; j++) kvec_y[j] = 2.0 * M_PI * static_cast<double>(ky) * ay[j];
      // Precompute and store eik_x * eik_y outside the kz-loop
      for(size_t i = 0; i != numberOfAtoms; ++i)
      {
        Complex eiky_temp = eik_y[i + numberOfAtoms * static_cast<size_t>(std::abs(ky))];
        //eiky_temp.imag(ky>=0 ? eiky_temp.imag() : -eiky_temp.imag());
        eiky_temp.imag = ky>=0 ? eiky_temp.imag : -eiky_temp.imag;
        eik_xy[i] = multiply(eik_x[i + numberOfAtoms * static_cast<size_t>(kx)], eiky_temp);
      }

      for(int kz = -kz_max; kz <= kz_max; ++kz)
      {
        // Ommit kvec==0
        if((kx * kx + ky * ky + kz * kz) != 0)
        {
          double kvec_z[3]; for(size_t j = 0; j < 3; j++) kvec_z[j] = 2.0 * M_PI * static_cast<double>(kz) * az[j];
          Complex cksum; cksum.real = 0.0; cksum.imag = 0.0;
          for(size_t i = 0; i != numberOfAtoms; ++i)
          {
            Complex eikz_temp = eik_z[i + numberOfAtoms * static_cast<size_t>(std::abs(kz))];
            //eikz_temp.imag(kz>=0 ? eikz_temp.imag() : -eikz_temp.imag());
            eikz_temp.imag = kz>=0 ? eikz_temp.imag : -eikz_temp.imag;
            size_t comp = 0; size_t posi = 0; GPU_get_component_thread(i, System, NumberOfComponents, &comp, &posi);
            double charge = System[comp].charge[posi];
            double scaling = System[comp].scaleCoul[posi];
            //cksum += scaling * charge * (eik_xy[i] * eikz_temp);
            Complex tempval; tempval = multiply(eik_xy[i], eikz_temp); tempval.real*=scaling*charge; tempval.imag*=scaling*charge;
            cksum.real += tempval.real; cksum.imag += tempval.imag;
            //printf("kxyz: %d, %d, %d, i: %lu, val: (%.10f, %.10f)\n", (int) kx, (int) ky, (int) kz, i, tempval.real(), tempval.imag());
          }
          //double rksq = (kvec_x + kvec_y + kvec_z).length_squared();
          double tempkvec[3] = {kvec_x[0]+kvec_y[0]+kvec_z[0], kvec_x[1]+kvec_y[1]+kvec_z[1], kvec_x[2]+kvec_y[2]+kvec_z[2]};
          double rksq = tempkvec[0]*tempkvec[0] + tempkvec[1]*tempkvec[1] + tempkvec[2]*tempkvec[2];
          double temp = factor * std::exp((-0.25 / alpha_squared) * rksq) / rksq;
          ewaldE += temp * (cksum.real * cksum.real + cksum.imag * cksum.imag);
          double tempsum = temp * (cksum.real * cksum.real + cksum.imag * cksum.imag);
          //printf("outer sum: kxyz: %d, %d, %d, cksum: (%.10f, %.10f), %.10f, %.10f\n", static_cast<int> (kx), static_cast<int> (ky), static_cast<int> (kz),cksum.real, cksum.imag, temp, tempsum);
          storedEik[nvec] = cksum;
          ++nvec;
          kzcount++;
        }
        kzinactive++;
      }
      kycount++;
    }
    kxcount++;
  }

  //printf("nvec is %lu, kxyz count: %ld %ld %ld, inactive: %lu, GPU-Pre-subtract-Energy: %.10f\n", nvec, kxcount, kycount, kzcount, kzinactive, ewaldE);
/*
  // Subtract self-energy
  double prefactor_self = FF.FFParams[3] * alpha / std::sqrt(M_PI);
  for(size_t i = 0; i != numberOfAtoms; ++i)
  {
    size_t comp = 0; size_t posi = 0; get_component_thread(i, System, NumberOfComponents, &comp, &posi);
    double charge = System[comp].charge[posi];
    double scaling = System[comp].scaleCoul[posi];
    ewaldE -= prefactor_self * scaling * charge * scaling * charge;
  }
  printf("ewaldE: %.10f\n", ewaldE);


  // Subtract exclusion-energy, Zhao's note: taking out the pairs of energies that belong to the same molecule
  size_t j_count = 0;
  for(size_t l = 0; l != NumberOfComponents; ++l)
  {
      for(size_t i = 0; i != System[l].size - 1; i++)
      {
        double factorA = System[l].scaleCoul[i] * System[l].charge[i];
        double posA[3] = {System[l].x[i], System[l].y[i], System[l].z[i]};
        for(size_t j = i + 1; j != ystem[l].size; j++)
        {
          double factorB = System[l].scaleCoul[j] * System[l].charge[j];
          double posB[3] = {System[l].x[j], System[l].y[j], System[l].z[j]};

          double posvec[3] = {posA[0]-posB[0], posA[1]-posB[1], posA[2]-posB[2]};
          GPU_JUST_Ewald_PBC(posvec, Box.Cell, Box.InverseCell, FF.OtherParams);
          double rr_dot = posvec[0]*posvec[0] + posvec[1]*posvec[1] + posvec[2]*posvec[2];
          double r = std::sqrt(rr_dot);

          ewaldE -= FF.FFParams[3] * factorA * factorB * std::erf(alpha * r) / r;
          j_count++;
        }
      }
  }
  printf("j_count: %zu, Post-subtract-Energy: %.10f\n", j_count, ewaldE);

  // Handle net-charges
  for(size_t i = 0; i != components.size(); ++i)
  {
    for(size_t j = 0; j != components.size(); ++j)
    {
      energyStatus.ewald += CoulombicFourierEnergySingleIon * netCharge[i] * netCharge[j];
    }
  }
  return ewaldE;
*/
}
