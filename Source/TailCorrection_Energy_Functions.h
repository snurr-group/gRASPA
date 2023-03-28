//Need to consider if we use CBCF//
double TotalTailCorrection(Components& SystemComponents, size_t FFsize, double Volume)
{
  double TailE = 0.0;
  if(!SystemComponents.HasTailCorrection) return TailE;
  for(size_t i = 0; i < SystemComponents.NumberOfPseudoAtoms.size(); i++)
    for(size_t j = i; j < SystemComponents.NumberOfPseudoAtoms.size(); j++)
    {
      size_t IJ_Forward = i * FFsize + j;
      if(SystemComponents.TailCorrection[IJ_Forward].UseTail)
      {
        size_t Ni = SystemComponents.NumberOfPseudoAtoms[i];
        size_t Nj = SystemComponents.NumberOfPseudoAtoms[j];
        //printf("i: %zu, j: %zu, Ni: %zu, Nj: %zu, E: %.5f\n", i,j,Ni,Nj, SystemComponents.TailCorrection[IJ_Forward].Energy);
        TailE += SystemComponents.TailCorrection[IJ_Forward].Energy * static_cast<double>(Ni * Nj);
      }
    }
  return TailE / Volume;
}

size_t get_change_in_pseudoAtoms(Components& SystemComponents, size_t SelectedComponent, size_t Type)
{
  size_t d = 0;
  for(size_t i = 0; i < SystemComponents.NumberOfPseudoAtomsForSpecies[SelectedComponent].size(); i++)
  {
    if(Type == SystemComponents.NumberOfPseudoAtomsForSpecies[SelectedComponent][i].x)
    {
      d = SystemComponents.NumberOfPseudoAtomsForSpecies[SelectedComponent][i].y; break;
    }
  }
  return d;
}

double TailCorrectionDifference(Components& SystemComponents, size_t SelectedComponent, size_t FFsize, double Volume, int MoveType)
{
  double TailE = 0.0;
  if(!SystemComponents.HasTailCorrection) return TailE;
  int sign = 1;
  switch(MoveType)
  {
    case INSERTION:
    {
      sign = 1;
      break;
    }
    case CBCF_INSERTION:
    {
      throw std::runtime_error("TAIL CORRECTIONS NOT READY FOR CBCF MOVES!");
    }
    case CBCF_DELETION:
    {
      throw std::runtime_error("TAIL CORRECTIONS NOT READY FOR CBCF MOVES!");
    }
    case DELETION:
    {
      sign = -1;
      break;
    }
  }
  for(size_t i = 0; i < SystemComponents.NumberOfPseudoAtoms.size(); i++)
  {
    size_t di = get_change_in_pseudoAtoms(SystemComponents, SelectedComponent, i); di *= sign;
    for(size_t j = i; j < SystemComponents.NumberOfPseudoAtoms.size(); j++)
    {
      int dj = get_change_in_pseudoAtoms(SystemComponents, SelectedComponent, j); dj *= sign;
      size_t IJ_Forward = i * FFsize + j;
      if(SystemComponents.TailCorrection[IJ_Forward].UseTail)
      {
        int    Ni = SystemComponents.NumberOfPseudoAtoms[i];
        int    Nj = SystemComponents.NumberOfPseudoAtoms[j];
        double E  = SystemComponents.TailCorrection[IJ_Forward].Energy;
        int    dN = Ni * dj + Nj * di + di * dj; //Zhao's note: need to define a temporary variable for this first//
        TailE    += E * static_cast<double>(dN);
        //printf("Ni: %zu, di: %d, Nj: %zu, dj: %d, dN: %.10f\n", Ni, di, Nj, dj, static_cast<double>(dN));
      }
    }
  }
  return TailE / Volume;
}
