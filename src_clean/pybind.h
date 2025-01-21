//PYBIND11
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <typeinfo>

namespace py = pybind11;

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
/////////////////////////////////
//SOME FUNCTIONS USED BY PYBIND//
/////////////////////////////////
double* get_epsilon_pointer(ForceField& FF)
{
  return FF.epsilon;
};
double* get_sigma_pointer(ForceField& FF)
{
  return FF.sigma;
};

double get_val_ptr(double* ptr, size_t location)
{
  return ptr[location];
}


////////////////////
//POINTER WRAPPERS//
////////////////////


template <class T> class ptr_wrapper
{
    public:
        ptr_wrapper() : ptr(nullptr) {}
        ptr_wrapper(T* ptr) : ptr(ptr) {}
        ptr_wrapper(const ptr_wrapper& other) : ptr(other.ptr) {}
        T& operator* () const { return *ptr; }
        T* operator->() const { return  ptr; }
        T* get() const { return ptr; }
        void destroy() { delete ptr; }
        T& operator[](std::size_t idx) const { return ptr[idx]; }
    private:
        T* ptr;
};

template <typename T>
py::array_t<T> ptr_to_pyarray(T* ptr, int size)
{   
  return py::array_t<T>(
        {size}, // shTpe
        {sizeof(T)}, // strides
        ptr // data pointer
    );
}

//void change_val_ptr(T* ptr, size_t location, T newval)
template <typename T>
void change_val_ptr(ptr_wrapper<T>& ptr, size_t location, T newval)
{
  ptr[location] = newval;
}

ptr_wrapper<double> get_ptr_double(Variables& Vars, std::string INPUT)
{ 
  if(INPUT == "FF.epsilon")    {return Vars.FF.epsilon;}
  else if(INPUT == "FF.sigma") {return Vars.FF.sigma;}
  else if(INPUT == "FF.shift") {return Vars.FF.shift;}
}

template <typename T>
py::array_t<T> get_arr(Variables& Vars, std::string INPUT)
{
  T* P;
  if(INPUT == "FF.epsilon")    {P = Vars.FF.epsilon;}
  else if(INPUT == "FF.sigma") {P = Vars.FF.sigma;}
  else if(INPUT == "FF.shift") {P = Vars.FF.shift;}
  //else if(INPUT == "FF.FFType") {P = Vars.FF.FFType;}
  else if(INPUT == "Vars.SystemComponents[0].HostSystem[0].charge") {P = Vars.SystemComponents[0].HostSystem[0].charge; }
  py::array_t<T> Array = ptr_to_pyarray(P, Vars.FF.size);
  return Array;
}

void CopyAtomDataFromGPU(Variables& Vars, size_t systemId, size_t component)
{
  Components& SystemComponents = Vars.SystemComponents[systemId];
  Atoms  device_System[SystemComponents.NComponents.x];
  Atoms& Host_System = SystemComponents.HostSystem[component];
  cudaMemcpy(device_System, Vars.Sims[systemId].d_a, SystemComponents.NComponents.x * sizeof(Atoms), cudaMemcpyDeviceToHost);
  // if the host allocate_size is different from the device, allocate more space on the host
  size_t current_allocated_size = device_System[component].Allocate_size;
  if(current_allocated_size != Host_System.Allocate_size) //Need to update host
  {
    Host_System.pos       = (double3*) malloc(device_System[component].Allocate_size*sizeof(double3));
    Host_System.scale     = (double*)  malloc(device_System[component].Allocate_size*sizeof(double));
    Host_System.charge    = (double*)  malloc(device_System[component].Allocate_size*sizeof(double));
    Host_System.scaleCoul = (double*)  malloc(device_System[component].Allocate_size*sizeof(double));
    Host_System.Type      = (size_t*)  malloc(device_System[component].Allocate_size*sizeof(size_t));
    Host_System.MolID     = (size_t*)  malloc(device_System[component].Allocate_size*sizeof(size_t));
    Host_System.Allocate_size = device_System[component].Allocate_size;
  }
  //Host_System[component].size      = device_System[component].size; //Zhao's note: no matter what, the size (not allocated size) needs to be updated

  cudaMemcpy(Host_System.pos, device_System[component].pos, \
             sizeof(double3)*device_System[component].size, cudaMemcpyDeviceToHost);
  cudaMemcpy(Host_System.scale, device_System[component].scale, \
             sizeof(double)*device_System[component].size, cudaMemcpyDeviceToHost);
  cudaMemcpy(Host_System.charge, device_System[component].charge, \
             sizeof(double)*device_System[component].size, cudaMemcpyDeviceToHost);
  cudaMemcpy(Host_System.scaleCoul, device_System[component].scaleCoul, \
             sizeof(double)*device_System[component].size, cudaMemcpyDeviceToHost);
  cudaMemcpy(Host_System.Type, device_System[component].Type, \
             sizeof(size_t)*device_System[component].size, cudaMemcpyDeviceToHost);
  cudaMemcpy(Host_System.MolID, device_System[component].MolID, \
             sizeof(size_t)*device_System[component].size, cudaMemcpyDeviceToHost);
  Host_System.size = device_System[component].size;
}

//try return a dictionary//
//get all atoms from a system (simulation box) and a component//
py::dict GetAllAtoms(Variables& Vars, size_t systemId, size_t component)
{
  py::dict AtomDict;
  CopyAtomDataFromGPU(Vars, systemId, component); //Copy the atom info for this component from GPU to CPU//
  Atoms& HostSystem = Vars.SystemComponents[systemId].HostSystem[component];
  size_t size = HostSystem.size;
  AtomDict["pos"] = ptr_to_pyarray(HostSystem.pos, size);
  AtomDict["charge"] = ptr_to_pyarray(HostSystem.charge, size);
  AtomDict["Type"]   = ptr_to_pyarray(HostSystem.Type, size);
  AtomDict["MolID"]  = ptr_to_pyarray(HostSystem.MolID, size);
  return AtomDict;
}

void UpdateAtomInfoHost(Atoms& HostSystem, Atoms& HostTrial, size_t Molsize, int& MoveType)
{
  for(size_t i = 0; i < Molsize; i++)
  switch(MoveType)
  {
    case TRANSLATION: case ROTATION: case SINGLE_INSERTION: case SPECIAL_ROTATION: case INSERTION: case REINSERTION:
    {
      size_t MolID = HostTrial.MolID[i];
      size_t index = Molsize * MolID + i;
      HostSystem.pos[index]    = HostTrial.pos[i];
      HostSystem.charge[index] = HostTrial.charge[i];
      //Type should not change!//
      HostSystem.MolID[index]  = HostTrial.MolID[i];
      break;
    }
    case SINGLE_DELETION: case DELETION: //Copy the last molecule to the deleted one's place
    {
      size_t MolID = HostTrial.MolID[i];
      size_t last_mol_index    = HostSystem.size - i - 1;
      size_t last_molID        = HostSystem.MolID[last_mol_index];
      //If the last molecule is the one deleted, then no action needed
      if(MolID != last_molID)
      {
        size_t index             = Molsize * (MolID + 1) - i - 1;
        HostSystem.pos[index]    = HostSystem.pos[last_mol_index];
        HostSystem.charge[index] = HostSystem.charge[last_mol_index];
        //Type should not change!//
      }
    }
  }
  if(MoveType == SINGLE_INSERTION || MoveType == INSERTION) HostSystem.size += Molsize;
  if(MoveType == SINGLE_DELETION  || MoveType == DELETION)  HostSystem.size -= Molsize;
}

py::dict GetTrialConfig(Variables& Vars, size_t systemId, size_t component, bool WholeConfig)
{
  py::dict AtomDict;
  //Atoms& Trial = Vars.Sims[systemId].Old;
  Atoms& Trial = Vars.Sims[systemId].New;
  size_t start = 0; //copy from position 0
  size_t size  = Vars.SystemComponents[systemId].Moleculesize[component];
  Vars.SystemComponents[systemId].Copy_GPU_Data_To_Temp(Trial, start, size); //copied GPU trial position to CPU//
  Atoms& HostTrial = Vars.SystemComponents[systemId].TempSystem;
  HostTrial.size = Trial.size;

  if(!WholeConfig) //Just transfer the trial configuration (just the moved parts)
  {
    AtomDict["pos"]    = ptr_to_pyarray(HostTrial.pos, size);
    AtomDict["charge"] = ptr_to_pyarray(HostTrial.charge, size);
    AtomDict["Type"]   = ptr_to_pyarray(HostTrial.Type, size);
    AtomDict["MolID"]  = ptr_to_pyarray(HostTrial.MolID, size);
  }
  else //Transfer the whole component configuration
  {
    CopyAtomDataFromGPU(Vars, systemId, component); //Copy the atom info for this component from GPU to CPU//
    Atoms& HostSystem = Vars.SystemComponents[systemId].HostSystem[component];
    UpdateAtomInfoHost(HostSystem, HostTrial, size, Vars.TempVal.MoveType);
    
    AtomDict["pos"]    = ptr_to_pyarray(HostSystem.pos, HostSystem.size);
    AtomDict["charge"] = ptr_to_pyarray(HostSystem.charge, HostSystem.size);
    AtomDict["Type"]   = ptr_to_pyarray(HostSystem.Type, HostSystem.size);
    AtomDict["MolID"]  = ptr_to_pyarray(HostSystem.MolID, HostSystem.size);
  }
  return AtomDict;
}

//get the pseudo_atom definitions, written to a dict//
py::dict GetPseudoAtomDefinitions(Variables& Vars)
{
  py::dict Dict;
  Dict["Name"]        = py::cast(Vars.PseudoAtoms.Name);
  Dict["Symbol"]      = py::cast(Vars.PseudoAtoms.Symbol);
  Dict["SymbolIndex"] = py::cast(Vars.PseudoAtoms.SymbolIndex);
  Dict["mass"]        = py::cast(Vars.PseudoAtoms.mass);
  return Dict;
}

py::dict GetBox(Variables& Vars, size_t systemId)
{
  py::dict Dict;
  Boxsize& Box = Vars.Box[systemId];
  Dict["Cell"]        = ptr_to_pyarray(Box.Cell, 9);
  Dict["InverseCell"] = ptr_to_pyarray(Box.InverseCell, 9);
  Dict["Volume"]      = Box.Volume;
  Dict["Cubic"]       = Box.Cubic;
  return Dict;
}

ptr_wrapper<int> get_ptr_int(Variables& Vars, std::string INPUT)
{ 
  if(INPUT == "FF.FFType"){return Vars.FF.FFType;}
}

template<typename T>
void print_ptr(ptr_wrapper<T> ptr)
{
  int  element = 1;
  bool regular = true;
  
  //if(std::is_same<T, int2>::value) element = 2;
  //if(std::is_same<T, int3>::value) element = 3;
  //if(std::is_same<T, double2>::value) element = 2;
  //if(std::is_same<T, double3>::value) element = 3;
  //if(std::is_same<T, Complex>::value)
  //{
  //  regular = false;
  //}
  for (int i = 0; i < 3; ++i)
  {
    if(regular)  //double.int.double2.double3//
    {
      if(element == 1) std::cout << ptr[i] << " ";
    }
    else
    {
      //if(std::is_same<T, Complex>::value) std::cout << " " << ptr[i].real << " " << ptr[i].imag << " ";
      //if(element == 2) std::cout << " " << ptr[i].x << " " << ptr[i].y << " ";
      //if(element == 3) std::cout << " " << ptr[i].x << " " << ptr[i].y << " " << ptr[i].z << " ";
    }
  }
  std::cout << "\n";
}

template <typename T>
T square(T x)
{
  return x * x;
}

//MoveEnergy add/substract//
void MoveEnergy_Add(MoveEnergy& A, MoveEnergy& B)
{
  A += B;
}

PYBIND11_MODULE(gRASPA, m)
{
  py::class_<double2>(m, "double2")
    .def(py::init<>())
    .def_readwrite("x", &double2::x)
    .def_readwrite("y", &double2::y);

  py::class_<double3>(m, "double3")
    .def(py::init<>())
    .def_readwrite("x", &double3::x)
    .def_readwrite("y", &double3::y)
    .def_readwrite("z", &double3::z);

  PYBIND11_NUMPY_DTYPE(double3, x, y, z);

  py::class_<Complex>(m, "Complex")
    .def(py::init<>())
    .def_readwrite("real", &Complex::real)
    .def_readwrite("imag", &Complex::imag);

  m.def("ptr_to_pyarray", &ptr_to_pyarray<double>, "ptr", "size");

  m.def("get_ptr_int",    &get_ptr_int,    "Vars", "INPUT");
  m.def("get_ptr_double", &get_ptr_double, "Vars", "INPUT");
  //m.def("wrap_get_ptr", &wrap_get_ptr<int>, "Vars", "INPUT");
  m.def("square", square<double>);
  m.def("square", square<int>);

  m.def("print_ptr", &print_ptr<int>);
  m.def("print_ptr", &print_ptr<double>);
  //m.def("print_ptr", &print_ptr<double3>);
  m.def("get_val_ptr", &get_val_ptr, "ptr", "location");
  m.def("change_val_ptr", &change_val_ptr<double>, "ptr", "location", "newvalue");
  m.def("change_val_ptr", &change_val_ptr<int>, "ptr", "location", "newvalue");
  m.def("change_val_ptr", &change_val_ptr<size_t>, "ptr", "location", "newvalue");
  m.def("change_val_ptr", &change_val_ptr<double2>, "ptr", "location", "newvalue");
  m.def("change_val_ptr", &change_val_ptr<double3>, "ptr", "location", "newvalue");

  py::class_<RandomNumber>(m, "RandomNumber")
    .def(py::init<>())
    .def_readwrite("host_random", &RandomNumber::host_random)
    .def("AllocateRandom", &RandomNumber::AllocateRandom, "Allocate Space for RandomNumber on CPU")
    .def("DeviceRandom", &RandomNumber::DeviceRandom, "Allocate Space for RandomNumber on GPU")
    .def_readwrite("randomsize", &RandomNumber::randomsize)
    .def_readwrite("offset", &RandomNumber::offset)
    .def_readwrite("Rounds", &RandomNumber::Rounds);

  py::class_<MoveEnergy>(m, "MoveEnergy")
    .def(py::init<>())
    .def_readwrite("HHVDW", &MoveEnergy::HHVDW)
    .def_readwrite("HHReal", &MoveEnergy::HHReal)
    .def_readwrite("HHEwaldE", &MoveEnergy::HHEwaldE)

    .def_readwrite("HGVDW", &MoveEnergy::HGVDW)
    .def_readwrite("HGReal", &MoveEnergy::HGReal)
    .def_readwrite("HGEwaldE", &MoveEnergy::HGEwaldE)
 
    .def_readwrite("GGVDW", &MoveEnergy::GGVDW)
    .def_readwrite("GGReal", &MoveEnergy::GGReal)
    .def_readwrite("GGEwaldE", &MoveEnergy::GGEwaldE) 

    .def_readwrite("TailE", &MoveEnergy::TailE)
    .def_readwrite("DNN_E", &MoveEnergy::DNN_E)

    .def("total", &MoveEnergy::total, "Calculate the total energy based on the component values")
    .def("take_negative", &MoveEnergy::take_negative, "Calculate the negated values")
    .def("zero", &MoveEnergy::zero, "zero the components of energies")
    .def("print", &MoveEnergy::print, "print the energies");
  //m.def("DeviceRandom", &DeviceRandom, "RandomNumber on GPU");
  //m.def("set_value", &set_value, "Set value");
 

  py::enum_<SIMULATION_MODE>(m, "SIMULATION_MODE")
    .value("CREATE_MOLECULE", CREATE_MOLECULE)
    .value("INITIALIZATION", INITIALIZATION)
    .value("EQUILIBRATION", EQUILIBRATION)
    .value("PRODUCTION", PRODUCTION)
    .export_values();

  py::class_<ForceField>(m, "ForceField")
    .def(py::init<>())
    .def_readwrite("VDWRealBias", &ForceField::VDWRealBias)
    .def_readwrite("noCharges", &ForceField::noCharges)
    .def_readwrite("size", &ForceField::size)
    .def_readwrite("OverlapCriteria", &ForceField::OverlapCriteria)
    .def_readwrite("CutOffVDW", &ForceField::CutOffVDW)
    .def_readwrite("CutOffCoul", &ForceField::CutOffCoul)
    .def_readonly("epsilon", &ForceField::epsilon);
  m.def("get_epsilon_pointer", &get_epsilon_pointer, py::return_value_policy::reference, "FF");
  m.def("get_sigma_pointer",   &get_sigma_pointer,   py::return_value_policy::reference, "FF");

  py::class_<Boxsize>(m, "Boxsize")
    .def(py::init<>())
    .def_readwrite("Cubic", &Boxsize::Cubic)
    .def_readwrite("Alpha", &Boxsize::Alpha);

  py::class_<Simulations>(m, "Simulations")
    .def(py::init<>())
    .def_readwrite("Nblocks", &Simulations::Nblocks)
    .def_readwrite("Box", &Simulations::Box);

  py::class_<ptr_wrapper<double>>(m,"double_ptr_wrapper");
  py::class_<ptr_wrapper<Simulations>>(m,"pSim");
  //m.def("get_ptr", &get_ptr<Simulations>, "Vars", "INPUT");

  py::class_<Atom_FF>(m, "Atom_FF")
    .def(py::init<>())
    .def_readwrite("Name",    &Atom_FF::Name)
    .def_readwrite("epsilon", &Atom_FF::epsilon)
    .def_readwrite("sigma",   &Atom_FF::sigma)
    .def_readwrite("shift",   &Atom_FF::shift)
    .def_readwrite("tail",    &Atom_FF::tail);

  py::class_<Tail>(m, "Tail")
    .def(py::init<>())
    .def_readwrite("UseTail", &Tail::UseTail)
    .def_readwrite("Energy",  &Tail::Energy);

  py::class_<Input_Container>(m, "Input_Container")
    .def(py::init<>())
    .def_readwrite("AtomFF", &Input_Container::AtomFF)
    .def_readwrite("Mix_Epsilon", &Input_Container::Mix_Epsilon)
    .def_readwrite("Mix_Sigma", &Input_Container::Mix_Sigma)
    .def_readwrite("Mix_Shift", &Input_Container::Mix_Shift)
    .def_readwrite("Mix_Tail", &Input_Container::Mix_Tail)
    .def_readwrite("CutOffVDW", &Input_Container::CutOffVDW)
    .def_readwrite("CutOffCoul", &Input_Container::CutOffCoul)
    .def_readwrite("VDWRealBias", &Input_Container::VDWRealBias)
    .def_readwrite("OverlapCriteria", &Input_Container::OverlapCriteria)
    .def_readwrite("noCharges", &Input_Container::noCharges);

  
  py::class_<MoveTempStorage>(m, "MoveTempStorage")
    .def(py::init<>())
    .def_readwrite("RandomNumber", &MoveTempStorage::RandomNumber)
    .def_readwrite("systemId",     &MoveTempStorage::systemId)
    .def_readwrite("component",    &MoveTempStorage::component)
    .def_readwrite("MoveType",     &MoveTempStorage::MoveType)
    .def_readwrite("Accept",       &MoveTempStorage::Accept)
    .def_readwrite("molecule",     &MoveTempStorage::molecule);

  py::class_<Move_Statistics>(m, "Move_Statistics")
    .def(py::init<>())
    .def_readwrite("TranslationProb", &Move_Statistics::TranslationProb)
    .def_readwrite("RotationProb",    &Move_Statistics::RotationProb)
    .def_readwrite("SwapProb",        &Move_Statistics::SwapProb)
    .def_readwrite("ReinsertionProb", &Move_Statistics::ReinsertionProb);

  py::class_<Components>(m, "Components")
    .def(py::init<>())
    .def_readwrite("Moves", &Components::Moves)
    .def_readwrite("deltaE", &Components::deltaE)
    .def_readwrite("NumberOfMolecules", &Components::NumberOfMolecule_for_Component);
    /*
    .def_readwrite("", &Components::Moves);
    .def_readwrite("Moves", &Components::Moves);
    .def_readwrite("Moves", &Components::Moves);
    .def_readwrite("Moves", &Components::Moves);
    */
   
  py::class_<PseudoAtomDefinitions>(m, "PseudoAtomDefinitions")
    .def(py::init<>())
    .def_readwrite("Name", &PseudoAtomDefinitions::Name)
    .def_readwrite("Symbol", &PseudoAtomDefinitions::Symbol)
    .def_readwrite("Mass", &PseudoAtomDefinitions::mass);
 
  py::class_<Variables>(m, "Variables")
    .def(py::init<>())
    .def("set_TEST", &Variables::set_TEST)
    .def("get_TEST", &Variables::get_TEST)
    .def_readwrite("Input", &Variables::Input)
    .def_readwrite("TEST", &Variables::TEST)
    .def_readwrite("Ttwo", &Variables::Ttwo)
    .def_readwrite("FF", &Variables::FF)
    .def_readwrite("Random", &Variables::Random)
    .def_readwrite("NumberOfInitializationCycles", &Variables::NumberOfInitializationCycles)
    .def_readwrite("NumberOfEquilibrationCycles", &Variables::NumberOfEquilibrationCycles)
    .def_readwrite("NumberOfProductionCycles", &Variables::NumberOfProductionCycles)
    .def_readwrite("SimulationMode",  &Variables::SimulationMode)
    .def_readwrite("SystemComponents",  &Variables::SystemComponents)
    .def_readwrite("PseudoAtoms",  &Variables::PseudoAtoms)
    .def_readwrite("MCMoveVariables", &Variables::TempVal);


  m.def("get_arr", &get_arr<double>, "Variable", "NAME");


  //Check final energy and energy drifts//
  m.def("get_total_energy", &check_energy_wrapper, "get total energy", py::arg("Var"), py::arg("SimulationIndex"));
  m.def("final_energy_summary", &ENERGY_SUMMARY, "summarize final energy and energy drifts", py::arg("Variable"));

  //m.def("get_total_energy", &check_energy_wrapper, "get total energy", "Var", "SimulationIndex"_a=0);

  ///////////
  // Enums //
  ///////////
  //MoveTypes//
  py::enum_<MoveTypes>(m, "MoveTypes")
    .value("TRANSLATION", MoveTypes::TRANSLATION)
    .value("ROTATION", MoveTypes::ROTATION)
    .value("SINGLE_INSERTION", MoveTypes::SINGLE_INSERTION)
    .value("SINGLE_DELETION", MoveTypes::SINGLE_DELETION);

  //////////////////////
  // HELPER FUNCTIONS //
  //////////////////////
  m.def("Get_Uniform_Random", &Get_Uniform_Random, "Get a random number from gRASPA's side");
  m.def("MoveEnergy_Add", &MoveEnergy_Add, py::arg("A"), py::arg("B"));
  m.def("omp_get_wtime", &omp_get_wtime);
  //Copy Data and get data from gRASPA to python//
  m.def("CopyAtomDataFromGPU", &CopyAtomDataFromGPU, py::arg("Variable"), py::arg("systemId"), py::arg("component"));
  m.def("GetAllAtoms", &GetAllAtoms, py::arg("Variable"), py::arg("systemId"), py::arg("component"));
  m.def("GetBox", &GetBox, py::arg("Variable"), py::arg("systemId"));
  m.def("GetPseudoAtomDefinitions", &GetPseudoAtomDefinitions, py::arg("Variable"));

  m.def("GetTrialConfig", &GetTrialConfig, py::arg("Vars"), py::arg("systemId"), py::arg("component"), py::arg("WholeConfig"));
 
  //////////////////////// 
  // GENERAL SIMULATION //
  ////////////////////////
  //The following three combines into a full, normal gRASPA simulation//
  m.def("Initialize", &Initialize, "Initialize SIMULATION");
  m.def("RUN", &RunSimulation, "Run SIMULATION");
  m.def("finalize", &EndOfSimulationWrapUp, "Final wrap up SIMULATION");
  ///////////////////
  // MOVES RELATED //
  ///////////////////
  m.def("InitializeMC", &InitialMCBeforeMoves, "InitializeMC", py::arg("Variables"), py::arg("systemId"));
  m.def("Determine_Number_Of_Steps", &Determine_Number_Of_Steps, py::arg("Variables"), py::arg("systemId"), py::arg("current_cycle"));

  //m.def("Select_Box_Component_Molecule", &Select_Box_Component_Molecule, "Select_Box_Component_Molecule", py::arg("Variables"), py::arg("systemId"), py::arg("component"), py::arg("molecule"), py::arg("RandomNumber"), py::return_value_policy::reference_internal);
  m.def("Select_Box_Component_Molecule", &Select_Box_Component_Molecule, "Select_Box_Component_Molecule", py::arg("Variables"), py::arg("systemId"));

  m.def("GatherStatisticsDuringSimulation", &GatherStatisticsDuringSimulation, "GatherStatisticsDuringSimulation", py::arg("Variables"), py::arg("systemId"), py::arg("current_cycle"));
  m.def("MCEndOfPhaseSummary", &MCEndOfPhaseSummary, "MCEndOfPhaseSummary", py::arg("Variables"));

  ///////////////
  // RUN MOVES //
  ///////////////
  m.def("Run_Simulation_ForOneBox", &Run_Simulation_ForOneBox, "run simulation for selected box", "Vars", "box_index", "SimulationMode");
  //SINGLE-BODY MOVE//
  m.def("SingleBody_Prepare", &SingleBody_Prepare, py::arg("Variables"), py::arg("systemId"));
  m.def("SingleBody_Calculation", &SingleBody_Calculation, py::arg("Vars"), py::arg("systemId"));
  m.def("SingleBody_Acceptance", &SingleBody_Acceptance, py::arg("Vars"), py::arg("Box_Index"), py::arg("delta_energy"));

  m.def("ForceField_Processing", &ForceField_Processing, py::arg("Input"));
  m.def("Copy_InputLoader_Data", &Copy_InputLoader_Data, py::arg("Vars"));
}
