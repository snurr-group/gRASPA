struct Allegro
{
  std::string ModelName;
  torch::jit::script::Module Model;
  Boxsize UCBox;
  Boxsize ReplicaBox;
  std::vector<Atoms> UCAtoms;
  std::vector<Atoms> ReplicaAtoms;
  double Cutoff = 6.0;
  double Cutoffsq = 0.0;
  NeighList NL;
  int3 NReplicacell = {1,1,1};

  size_t nstep = 0;

  void GetSQ_From_Cutoff()
  {
    Cutoffsq = Cutoff * Cutoff;
  }

  double dot(double3 a, double3 b)
  {
    return a.x * b.x + a.y * b.y + a.z * b.z;
  }
  double matrix_determinant(double* x) //9*1 array
  {
    double m11 = x[0*3+0]; double m21 = x[1*3+0]; double m31 = x[2*3+0];
    double m12 = x[0*3+1]; double m22 = x[1*3+1]; double m32 = x[2*3+1];
    double m13 = x[0*3+2]; double m23 = x[1*3+2]; double m33 = x[2*3+2];
    double determinant = +m11 * (m22 * m33 - m23 * m32) - m12 * (m21 * m33 - m23 * m31) + m13 * (m21 * m32 - m22 * m31);
    return determinant;
  }

  void inverse_matrix(double* x, double **inverse_x)
  {
    double m11 = x[0*3+0]; double m21 = x[1*3+0]; double m31 = x[2*3+0];
    double m12 = x[0*3+1]; double m22 = x[1*3+1]; double m32 = x[2*3+1];
    double m13 = x[0*3+2]; double m23 = x[1*3+2]; double m33 = x[2*3+2];
    double determinant = +m11 * (m22 * m33 - m23 * m32) - m12 * (m21 * m33 - m23 * m31) + m13 * (m21 * m32 - m22 * m31);
    double* result = (double*) malloc(9 * sizeof(double));
    result[0] = +(m22 * m33 - m32 * m23) / determinant;
    result[3] = -(m21 * m33 - m31 * m23) / determinant;
    result[6] = +(m21 * m32 - m31 * m22) / determinant;
    result[1] = -(m12 * m33 - m32 * m13) / determinant;
    result[4] = +(m11 * m33 - m31 * m13) / determinant;
    result[7] = -(m11 * m32 - m31 * m12) / determinant;
    result[2] = +(m12 * m23 - m22 * m13) / determinant;
    result[5] = -(m11 * m23 - m21 * m13) / determinant;
    result[8] = +(m11 * m22 - m21 * m12) / determinant;
    *inverse_x = result;
  }

  __host__ double3 GetFractionalCoord(double* InverseCell, bool Cubic, double3 posvec)
  {
    double3 s = {0.0, 0.0, 0.0};
    s.x=InverseCell[0*3+0]*posvec.x + InverseCell[1*3+0]*posvec.y + InverseCell[2*3+0]*posvec.z;
    s.y=InverseCell[0*3+1]*posvec.x + InverseCell[1*3+1]*posvec.y + InverseCell[2*3+1]*posvec.z;
    s.z=InverseCell[0*3+2]*posvec.x + InverseCell[1*3+2]*posvec.y + InverseCell[2*3+2]*posvec.z;
    return s;
  }

  __host__ double3 GetRealCoordFromFractional(double* Cell, bool Cubic, double3 s)
  {
    double3 posvec = {0.0, 0.0, 0.0};
    posvec.x=Cell[0*3+0]*s.x+Cell[1*3+0]*s.y+Cell[2*3+0]*s.z;
    posvec.y=Cell[0*3+1]*s.x+Cell[1*3+1]*s.y+Cell[2*3+1]*s.z;
    posvec.z=Cell[0*3+2]*s.x+Cell[1*3+2]*s.y+Cell[2*3+2]*s.z;
    return posvec;
  }

  std::vector<std::string> split(const std::string txt, char ch)
  {
    size_t pos = txt.find(ch);
    size_t initialPos = 0;
    std::vector<std::string> strs{};

    // Decompose statement
    while (pos != std::string::npos) {

        std::string s = txt.substr(initialPos, pos - initialPos);
        if (!s.empty())
        {
            strs.push_back(s);
        }
        initialPos = pos + 1;

        pos = txt.find(ch, initialPos);
    }

    // Add the last one
    std::string s = txt.substr(initialPos, std::min(pos, txt.size()) - initialPos + 1);
    if (!s.empty())
    {
        strs.push_back(s);
    }

    return strs;
  }

  bool caseInSensStringCompare(const std::string& str1, const std::string& str2)
  {
    return str1.size() == str2.size() && std::equal(str1.begin(), str1.end(), str2.begin(), [](auto a, auto b) {return std::tolower(a) == std::tolower(b); });
  }

  void Split_Tab_Space(std::vector<std::string>& termsScannedLined, std::string& str)
  {
    if (str.find("\t", 0) != std::string::npos) //if the delimiter is tab
    {
      termsScannedLined = split(str, '\t');
    }
    else
    {
      termsScannedLined = split(str, ' ');
    }
  }
  /*
  void write_ReplicaPos(auto& pos, auto& ij2type, size_t ntotal, size_t nstep)
  {
    std::ofstream textrestartFile{};
    std::string fname = "Pos_Replica_" + std::to_string(nstep) + ".txt";
    std::filesystem::path cwd = std::filesystem::current_path();
    std::filesystem::path fileName = cwd /fname;
    textrestartFile = std::ofstream(fileName, std::ios::out);

    textrestartFile << "i x y z type\n";
    for(size_t i = 0; i < ntotal; i++)
      textrestartFile << i << " " << pos[i][0] << " " << pos[i][1] << " " << pos[i][2] << " " << ij2type[i] << "\n";
    textrestartFile.close();
  }
  void write_edges(auto& edges, auto& ij2type, size_t nedges, size_t nstep)
  {
    std::ofstream textrestartFile{};
    std::string fname = "Neighbor_List_" + std::to_string(nstep) + ".txt";
    std::filesystem::path cwd = std::filesystem::current_path();
    std::filesystem::path fileName = cwd /fname;
    textrestartFile = std::ofstream(fileName, std::ios::out);

    printf("There are %zu edges\n", nedges);

    textrestartFile << "i j ij2type edge_counter\n";
    for(size_t i = 0; i < nedges; i++)
        textrestartFile << edges[0][i] << " " << edges[1][i] << " " << ij2type[edges[0][i]] << " " << i << "\n";
    textrestartFile.close();
  }
  */
  void AllocateUCSpace(size_t comp)
  {
    UCAtoms[comp].pos   = (double3*) malloc(UCAtoms[comp].size * sizeof(double3));
    UCAtoms[comp].Type  = (size_t*)  malloc(UCAtoms[comp].size * sizeof(size_t));
  }

  void GenerateUCBox(double* SuperCell, int3 Ncell)
  {
    UCBox.Cell        = (double*) malloc(9 * sizeof(double));
    UCBox.InverseCell = (double*) malloc(9 * sizeof(double));
    for(size_t i = 0; i < 9; i++) UCBox.Cell[i] = SuperCell[i];
    UCBox.Cell[0] /= Ncell.x; UCBox.Cell[1]  = 0.0;     UCBox.Cell[2]  = 0.0;
    UCBox.Cell[3] /= Ncell.y; UCBox.Cell[4] /= Ncell.y; UCBox.Cell[5]  = 0.0;
    UCBox.Cell[6] /= Ncell.z; UCBox.Cell[7] /= Ncell.z; UCBox.Cell[8] /= Ncell.z;
    inverse_matrix(UCBox.Cell, &UCBox.InverseCell);
  }

  //Assuming the framework atoms are reproduced, and the order of atoms in a unit cell matches the order in the cif//
  //This assumes rigid framework//
  void CopyAtomsFromFirstUnitcell(Atoms& HostAtoms, size_t comp, int3 NSupercell, PseudoAtomDefinitions& PseudoAtoms)
  {
    size_t NAtoms = HostAtoms.Molsize / (NSupercell.x * NSupercell.y * NSupercell.z);
    if(HostAtoms.size % NAtoms != 0) throw std::runtime_error("SuperCell size cannot be divided by number of supercell atoms!!!!");
    UCAtoms[comp].size = NAtoms;
    AllocateUCSpace(comp);
    for(size_t i = 0; i < NAtoms; i++)
    {
      UCAtoms[comp].pos[i]  = HostAtoms.pos[i];
      size_t SymbolIdx = PseudoAtoms.GetSymbolIdxFromPseudoAtomTypeIdx(HostAtoms.Type[i]);
      UCAtoms[comp].Type[i] = SymbolIdx;
      //printf("Component %zu, Atom %zu, xyz %f %f %f, Type %zu, SymbolIndex %zu\n", comp, i, UCAtoms[comp].pos[i].x, UCAtoms[comp].pos[i].y, UCAtoms[comp].pos[i].z, HostAtoms.Type[i], UCAtoms[comp].Type[i]);
    }
  }
  /*
  void ReadCutOffFromModel(std::string Name)
  {
    std::vector<std::string> termsScannedLined{};
    std::ifstream simfile(Name);
    std::filesystem::path pathfile = std::filesystem::path(Name);
    std::string str;
    size_t skip = 1;
    size_t count= 0;
    while (std::getline(simfile, str))
    {
      //if(count <= skip) continue;
      if(str.find("r_max", 0) != std::string::npos)
      {
        Split_Tab_Space(termsScannedLined, str);
        Cutoff = std::stod(termsScannedLined[1]);
        return;
      }
    }
  }
  */
  void ReadModel(std::string Name)
  {
    ModelName = Name;
    //https://stackoverflow.com/questions/18199728/is-it-possible-to-have-an-auto-member-variable
    //auto device = torch::kCPU;
    auto device = torch::kCUDA;//c10::Device(torch::kCUDA,1);
    //Added new lines from pair_allegro.cpp to see if it fixes the issue//
    std::unordered_map<std::string, std::string> metadata =
    {
      {"config", ""},
      {"nequip_version", ""},
      {"r_max", ""},
      {"n_species", ""},
      {"type_names", ""},
      {"_jit_bailout_depth", ""},
      {"_jit_fusion_strategy", ""},
      {"allow_tf32", ""}
    };
    Model = torch::jit::load(ModelName, device, metadata);
    Model.eval();
    //Freeze the model
    Model = torch::jit::freeze(Model);
    //ReadCutOffFromModel(ModelName);
  }
  
  double Predict()
  {
    size_t nAtoms = 0; for(size_t i = 0; i < UCAtoms.size(); i++) nAtoms += UCAtoms[i].size;
    size_t ntotal = 0; for(size_t i = 0; i < ReplicaAtoms.size(); i++) ntotal += ReplicaAtoms[i].size;

    torch::Tensor edges_tensor = torch::zeros({2,NL.nedges}, torch::TensorOptions().dtype(torch::kInt64));
    auto edges = edges_tensor.accessor<long, 2>();

    torch::Tensor pos_tensor = torch::zeros({ntotal, 3});
    auto pos = pos_tensor.accessor<float, 2>();

    torch::Tensor ij2type_tensor = torch::zeros({ntotal}, torch::TensorOptions().dtype(torch::kInt64));
    auto ij2type = ij2type_tensor.accessor<long, 1>();
   
    size_t counter = 0; 
    for(size_t comp = 0; comp < ReplicaAtoms.size(); comp++)
    for(size_t i = 0; i < ReplicaAtoms[comp].size; i++)
    {
      pos[counter][0] = ReplicaAtoms[comp].pos[i].x;
      pos[counter][1] = ReplicaAtoms[comp].pos[i].y;
      pos[counter][2] = ReplicaAtoms[comp].pos[i].z;
      ij2type[counter]= ReplicaAtoms[comp].Type[i];
      //if(comp != 0)
      //  printf("comp %zu, counter %zu, ij2type %zu\n", comp, counter, ij2type[counter]);
      counter ++;
    }
    size_t N_Replica_FrameworkAtoms = 0; size_t NFrameworkAtoms = 0;
    N_Replica_FrameworkAtoms = ReplicaAtoms[0].size;
    NFrameworkAtoms = UCAtoms[0].size;
    for(size_t i = 0; i < nAtoms; i++)
    {
      for(size_t j = 0; j < NL.List[i].size(); j++)
      {
        size_t edge_counter = NL.List[i][j].z;
        edges[0][edge_counter] = NL.List[i][j].x; //i
        //Try to map the adsorbate i index to the replica-cell index
        //So the replicacell-adsorbate for cell 0 (center box) is located after the framework atoms//
        if(NL.List[i][j].x >= NFrameworkAtoms) //adsorbate molecule, shift i by number of framework replica atoms//
        {
          edges[0][edge_counter] -= NFrameworkAtoms;
          edges[0][edge_counter] += (N_Replica_FrameworkAtoms);
        }
        edges[1][edge_counter] = NL.List[i][j].y; //j
        //printf("i %d, j %d, edge_counter %d\n", NL.List[i][j].x, NL.List[i][j].y, NL.List[i][j].z);
      }
    }
    //write_ReplicaPos(pos, ij2type, ntotal, nstep);
    //write_edges(edges, ij2type, NL.nedges, nstep);

    //auto device = torch::kCPU;
    auto device = torch::kCUDA;//c10::Device(torch::kCUDA,1);
    c10::Dict<std::string, torch::Tensor> input;
    input.insert("pos", pos_tensor.to(device));
    input.insert("edge_index", edges_tensor.to(device));
    input.insert("atom_types", ij2type_tensor.to(device));
    std::vector<torch::IValue> input_vector(1, input);

    auto output = Model.forward(input_vector).toGenericDict();

    torch::Tensor atomic_energy_tensor = output.at("atomic_energy").toTensor().cpu();
    auto atomic_energies = atomic_energy_tensor.accessor<float, 2>();

    float atomic_energy_sum = atomic_energy_tensor.sum().data_ptr<float>()[0];

    float nAtomSum = 0.0;
    for(size_t i = 0; i < nAtoms; i++)
    {
      size_t AtomIndex = i;
      if(i >= NFrameworkAtoms) //adsorbate molecule, shift i by number of framework replica atoms//
      {
        AtomIndex -= NFrameworkAtoms;
        AtomIndex += N_Replica_FrameworkAtoms;
      }
      nAtomSum += atomic_energies[AtomIndex][0];
      //printf("atom %zu, AtomIndex %zu, energy: %.5f\n", i, AtomIndex, atomic_energies[AtomIndex][0]);
    }
    nstep ++;
    return static_cast<double>(nAtomSum);
  }
 
  void ReplicateAtomsPerComponent(size_t comp, bool Allocate)
  {
    //Get Fractional positions//
    std::vector<double3>fpos;
    for(size_t i = 0; i < UCAtoms[comp].size; i++)
    {
      fpos.push_back(GetFractionalCoord(UCBox.InverseCell, UCBox.Cubic, UCAtoms[comp].pos[i]));
    }

    size_t NTotalCell = static_cast<size_t>(NReplicacell.x * NReplicacell.y * NReplicacell.z);
    double3 Shift = {(double)1/NReplicacell.x, (double)1/NReplicacell.y, (double)1/NReplicacell.z};
    if(NReplicacell.x % 2 == 0) throw std::runtime_error("Ncell in x needs to be an odd number (so that original unit cell sits in the center\n");
    if(NReplicacell.y % 2 == 0) throw std::runtime_error("Ncell in y needs to be an odd number (so that original unit cell sits in the center\n");
    if(NReplicacell.z % 2 == 0) throw std::runtime_error("Ncell in z needs to be an odd number (so that original unit cell sits in the center\n");
    int Minx = (NReplicacell.x - 1)/2 * -1; int Maxx = (NReplicacell.x - 1)/2;
    int Miny = (NReplicacell.y - 1)/2 * -1; int Maxy = (NReplicacell.y - 1)/2;
    int Minz = (NReplicacell.z - 1)/2 * -1; int Maxz = (NReplicacell.z - 1)/2;
    std::vector<int>xs; std::vector<int>ys; std::vector<int>zs;
    xs.push_back(0);
    ys.push_back(0);
    zs.push_back(0);
    for(int i = Minx; i <= Maxx; i++)
      if(i != 0)
        xs.push_back(i);
    for(int i = Miny; i <= Maxy; i++)
      if(i != 0)
        ys.push_back(i);
    for(int i = Minz; i <= Maxz; i++)
      if(i != 0)
        zs.push_back(i);

    if(Allocate)
    {
      ReplicaAtoms[comp].pos   = (double3*) malloc(NTotalCell * UCAtoms[comp].size * sizeof(double3));
      ReplicaAtoms[comp].Type  = (size_t*)  malloc(NTotalCell * UCAtoms[comp].size * sizeof(size_t));
    }
    size_t counter = 0;
    for(size_t a = 0; a < static_cast<size_t>(NReplicacell.x); a++)
      for(size_t b = 0; b < static_cast<size_t>(NReplicacell.y); b++)
        for(size_t c = 0; c < static_cast<size_t>(NReplicacell.z); c++)
        {
          int ix = xs[a];
          int jy = ys[b];
          int kz = zs[c];
          //printf("a: %zu, ix: %d, b: %zu, jy: %d, c: %zu, kz: %d\n", a, ix, b, jy, c, kz);
          double3 NCellID = {(double) ix, (double) jy, (double) kz};
          for(size_t i = 0; i < UCAtoms[comp].size; i++)
          {
            double3 temp = {fpos[i].x + NCellID.x, 
                            fpos[i].y + NCellID.y,
                            fpos[i].z + NCellID.z};
            double3 super_fpos = {temp.x * Shift.x, 
                                  temp.y * Shift.y, 
                                  temp.z * Shift.z};
            // Get real xyz from fractional xyz //
            double3 Replica_pos;
            Replica_pos.x = super_fpos.x*ReplicaBox.Cell[0]+super_fpos.y*ReplicaBox.Cell[3]+super_fpos.z*ReplicaBox.Cell[6];
            Replica_pos.y = super_fpos.x*ReplicaBox.Cell[1]+super_fpos.y*ReplicaBox.Cell[4]+super_fpos.z*ReplicaBox.Cell[7];
            Replica_pos.z = super_fpos.x*ReplicaBox.Cell[2]+super_fpos.y*ReplicaBox.Cell[5]+super_fpos.z*ReplicaBox.Cell[8];
            ReplicaAtoms[comp].pos[counter]   = Replica_pos;
            ReplicaAtoms[comp].Type[counter]  = UCAtoms[comp].Type[i];
            counter ++;
          }
        }
    ReplicaAtoms[comp].size = NTotalCell * UCAtoms[comp].size;
  }
  
  //For Single Unit cell atoms, we separate them into different components//
  //For replica, we also do that//
  void GenerateReplicaCells(bool Allocate)
  {
    size_t NComp = UCAtoms.size();
    size_t N_UCAtom = 0; for(size_t comp = 0; comp < NComp; comp++) N_UCAtom += UCAtoms[comp].size;
    if(Allocate)
    {
      ReplicaBox.Cell = (double*) malloc(9 * sizeof(double));
      ReplicaBox.InverseCell = (double*) malloc(9 * sizeof(double));
      for(size_t i = 0; i < 9; i++) ReplicaBox.Cell[i] = UCBox.Cell[i];
 
      ReplicaBox.Cell[0] *= NReplicacell.x; ReplicaBox.Cell[1] *= 0.0;            ReplicaBox.Cell[2] *= 0.0;
      ReplicaBox.Cell[3] *= NReplicacell.y; ReplicaBox.Cell[4] *= NReplicacell.y; ReplicaBox.Cell[5] *= 0.0;
      ReplicaBox.Cell[6] *= NReplicacell.z; ReplicaBox.Cell[7] *= NReplicacell.z; ReplicaBox.Cell[8] *= NReplicacell.z;

      printf("a: %f, b: %f, c: %f\n", ReplicaBox.Cell[0], ReplicaBox.Cell[4], ReplicaBox.Cell[8]);
      inverse_matrix(ReplicaBox.Cell, &ReplicaBox.InverseCell);
    }
    //Assuming Framework fixed//
    for(size_t comp = 0; comp < UCAtoms.size(); comp++)
      if(Allocate || comp != 0)
        ReplicateAtomsPerComponent(comp, Allocate);
  }
  void Get_Neighbor_List_Replica(bool Initialize)
  {
    if(Initialize) GetSQ_From_Cutoff();
    size_t AtomCount = 0;
    for(size_t comp = 0; comp < UCAtoms.size(); comp++)
    {
      for(size_t j = 0; j < UCAtoms[comp].size; j++)
      {
        std::vector<int3>Neigh_per_atom;
        if(Initialize)
        {
          NL.List.push_back(Neigh_per_atom);
          NL.cumsum_neigh_per_atom.push_back(0);
        }
        else
        {
          NL.List[AtomCount] = Neigh_per_atom;
          NL.cumsum_neigh_per_atom[AtomCount] = 0;
        }
        AtomCount ++; //printf("AtomCount: %zu\n", AtomCount);
      }
    }
    NL.nedges = 0; //rezero
    if(!Initialize) Copy_From_Rigid_Framework_Edges(UCAtoms[0].size); //When compi = 0 and compj = 0//
 
    size_t i_start = 0;
    for(size_t compi = 0; compi < UCAtoms.size(); compi++)
    {
      size_t j_start = 0;
      for(size_t compj = 0; compj < ReplicaAtoms.size(); compj++)
      {
        if(Initialize || (compi != 0 || compj != 0))
        {
          //printf("compi %zu, compj %zu, nedges %zu\n", compi, compj, NL.nedges);
          bool SameComponent = false; if(compi == compj) SameComponent = true;
          Count_Edges_Replica(compi, compj, i_start, j_start, SameComponent);
          if(Initialize && compi == 0 && compj == 0) Copy_To_Rigid_Framework_Edges(UCAtoms[0].size);
        }
        j_start += ReplicaAtoms[compj].size;
      }
      i_start += UCAtoms[compi].size;
    }
    int prev = 0;
    for(size_t i = 0; i < NL.List.size(); i++)
    {
      size_t nedge_perAtom = NL.List[i].size();
      if(i != 0) prev = NL.cumsum_neigh_per_atom[i-1];
      NL.cumsum_neigh_per_atom[i] = prev + nedge_perAtom;
      //printf("Atom %zu, cumsum: %zu\n", i, NL.cumsum_neigh_per_atom[i]);
    }
  }

  void Count_Edges_Replica(size_t compi, size_t compj, size_t i_start, size_t j_start, bool SameComponent)
  {
    //printf("Atom size %zu, ReplicaAtom size %zu, i_start %zu, j_start %zu\n", UCAtoms[compi].size, ReplicaAtoms[compj].size, i_start, j_start);
    for(size_t i = 0; i < UCAtoms[compi].size; i++)
    {
      size_t i_idx = i_start + i;
      for(size_t j = 0; j < ReplicaAtoms[compj].size; j++)
      {
        size_t j_idx = j_start + j;
        if(i == j && SameComponent) continue; //Becareful with this line, we are separating the atoms into different components, CO2 atom for replica-cell and for unitcell should have the same i/j, so we just compare i/j (relative index), not i_idx/j_idx (absolute index).
        //Also need to make sure that they are the same component when comparing//
        double3 dist = {UCAtoms[compi].pos[i].x - ReplicaAtoms[compj].pos[j].x, 
                        UCAtoms[compi].pos[i].y - ReplicaAtoms[compj].pos[j].y,
                        UCAtoms[compi].pos[i].z - ReplicaAtoms[compj].pos[j].z};

        double dsq = dot(dist, dist);
        if(dsq <= Cutoffsq)
        { 
          NL.List[i_idx].push_back({(int)i_idx, (int)j_idx, (int) NL.nedges}); //nedge here = edge_counter//
          NL.nedges ++;
        }
      }
    }
    //printf("i_start %zu, j_start %zu, nedges %zu\n", i_start, j_start, NL.nedges);
  }

  //Record framework-framework neighbor list//
  void Copy_To_Rigid_Framework_Edges(size_t NFrameworkAtom)
  {
    for(size_t i = 0; i < NFrameworkAtom; i++)
    {
      NL.FrameworkList.push_back(NL.List[i]);
    }
  }
  void Copy_From_Rigid_Framework_Edges(size_t NFrameworkAtom)
  {
    for(size_t i = 0; i < NFrameworkAtom; i++)
    {
      NL.List[i] = NL.FrameworkList[i];
      NL.nedges += NL.FrameworkList[i].size();
    }
  }
  void WrapSuperCellAtomIntoUCBox(size_t comp)
  {
    std::vector<double>Bonds; //For checking bond distances if the molecule is wrapped onto different sides of the box//
    std::vector<double3>newpos;
    for(size_t i = 0; i < UCAtoms[comp].size; i++)
    {
      double3 FPOS = GetFractionalCoord(UCBox.InverseCell, UCBox.Cubic, UCAtoms[comp].pos[i]);
      //printf("New Atom %zu, fxyz: %f %f %f\n", i, FPOS.x, FPOS.y, FPOS.z);
      double3 FLOOR = {(double)floor(FPOS.x), (double)floor(FPOS.y), (double)floor(FPOS.z)};
      double3 New_fpos = {FPOS.x - FLOOR.x,
                          FPOS.y - FLOOR.y,
                          FPOS.z - FLOOR.z};

      newpos.push_back(New_fpos);
      
      if(i == 0) continue;
      double3 dist = {UCAtoms[comp].pos[i].x - UCAtoms[comp].pos[i - 1].x, 
                      UCAtoms[comp].pos[i].y - UCAtoms[comp].pos[i - 1].y,
                      UCAtoms[comp].pos[i].z - UCAtoms[comp].pos[i - 1].z};
      double dsq = dot(dist, dist);
      Bonds.push_back(dsq);     
    }
    for(size_t i = 0; i < UCAtoms[comp].size; i++)
    {
      double3 Real_Pos = GetRealCoordFromFractional(UCBox.Cell, UCBox.Cubic, newpos[i]);
      UCAtoms[comp].pos[i] = Real_Pos;
      /*
      //printf("NEW POS %zu, xyz: %f %f %f\n", Real_Pos.x, Real_Pos.y, Real_Pos.z);
      if(i == 0) continue;
      double3 dist = {UCAtoms[comp].pos[i].x - UCAtoms[comp].pos[i - 1].x,
                      UCAtoms[comp].pos[i].y - UCAtoms[comp].pos[i - 1].y,
                      UCAtoms[comp].pos[i].z - UCAtoms[comp].pos[i - 1].z};
      double dsq = dot(dist, dist);
      if(abs(dsq - Bonds[i]) > 1e-10)
        printf("THERE IS ERROR IN BOND LENGTH AFTER WRAPPING for ATOM %zu\n", i);
      */
    }
  }
  double MCEnergyWrapper(size_t comp, bool Initialize, double DNNEnergyConversion)
  {
    WrapSuperCellAtomIntoUCBox(comp);
    GenerateReplicaCells(Initialize);
    Get_Neighbor_List_Replica(Initialize);
    double DNN_E = Predict();
    //This generates the unit of eV, convert to 10J/mol.
    //https://www.weizmann.ac.il/oc/martin/tools/hartree.html
    return DNN_E * DNNEnergyConversion;
  }
  /*
  void CopyHostToUCAtoms(double3* pos, size_t comp, size_t Molsize)
  {
    for(size_t i = 0; i < Molsize; i++)
      UCAtoms[comp].pos[i] = pos[i];
  }
  */
};
