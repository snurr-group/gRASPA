void Initialize_WangLandauIteration(LAMBDA& lambda);
void Sample_WangLandauIteration(LAMBDA& lambda);
void Finalize_WangLandauIteration(LAMBDA& lambda);
void Adjust_WangLandauIteration(LAMBDA& lambda);

double get_lambda(LAMBDA& lambda);
int    selectNewBin(LAMBDA& lambda);

void Initialize_WangLandauIteration(LAMBDA& lambda)
{
  lambda.WangLandauScalingFactor = 1.0;
  std::fill(lambda.Histogram.begin(),  lambda.Histogram.end(),  0.0);
  std::fill(lambda.biasFactor.begin(), lambda.biasFactor.end(), 0.0);
  printf("Initialize WL\n");
}

void Sample_WangLandauIteration(LAMBDA& lambda)
{
  //printf("Before Update: bin %zu, Factor: %.5f\n", lambda.currentBin, lambda.biasFactor[lambda.currentBin]);
  lambda.biasFactor[lambda.currentBin] -= lambda.WangLandauScalingFactor;
  //printf("Current Bias for %zu bin: %.5f, WL Factor: %.5f\n", lambda.currentBin, lambda.biasFactor[lambda.currentBin], lambda.WangLandauScalingFactor);
  lambda.Histogram[lambda.currentBin] += 1.0;
  
  //printf("Total Bin: %zu\n", lambda.biasFactor.size());
  //for(size_t i = 0; i < lambda.biasFactor.size(); i++)
  //  printf("Bin %zu, Factor: %.5f\n", i, lambda.biasFactor[i]);
}

void Adjust_WangLandauIteration(LAMBDA& lambda)
{
  //Checking for flatness criteria//
  std::vector<double>::iterator minValueIterator = std::min_element(lambda.Histogram.begin(), lambda.Histogram.end());
  double minimumValue = *minValueIterator;
  //printf("Min Histogram Value: %.5f\n", minimumValue);
  if(minimumValue>0.01 && lambda.WangLandauScalingFactor > 1e-3)
  {
    //printf("Adjusting WL Factor!\n");
    double sumOfHistogram = std::accumulate(lambda.Histogram.begin(), lambda.Histogram.end(), 0.0);
    if(minimumValue / sumOfHistogram > 0.01)
    {
      lambda.WangLandauScalingFactor *= 0.5;
    }
  }
  std::fill(lambda.Histogram.begin(), lambda.Histogram.end(), 0.0);
  //printf("Adjusting WL Factor: %.5f\n", lambda.WangLandauScalingFactor);
}

void Finalize_WangLandauIteration(LAMBDA& lambda)
{
  //std::fill(lambda.Histogram.begin(), lambda.Histogram.end(), 0.0);

  double normalize = lambda.biasFactor[0];
  for(double &bias : lambda.biasFactor)
  {
    bias -= normalize;
  }
  printf("Finalize WL\n");
}

inline double get_lambda(LAMBDA& lambda)
{
  return static_cast<double>(lambda.currentBin) * lambda.delta;
}

inline int selectNewBin(LAMBDA& lambda)
{
  return lambda.currentBin + static_cast<int>(static_cast<double>(lambda.binsize) * (Get_Uniform_Random() - 0.5));
}

inline int selectNewBinTMMC(LAMBDA& lambda)
{ 
  //Think about whether the code can stay at the same bin//
  //Staying at the same bin may help the WL biasing//
  int delta = 1; if(Get_Uniform_Random() < 0.5) delta = -1;
  return lambda.currentBin + delta;
}
