#include <stdio.h>

#include <array>

using namespace std;

/*
static inline std::array<double,2> GPUpotentialVDW(std::array<double,4>& FFarg, const double scaling, const double rr, int potentialType)
{
    //LennardJones = 0,
    //BuckingHam = 1,
    //Morse = 2,
    //FeynmannHibbs = 3,
    //MM3 = 4,
    //BornHugginsMeyer = 5
    //VDWParameters::Type potentialType = forcefield(typeA, typeB).type;
    std::array<double,2> result;
    switch (potentialType)
    {
    case 0:
    {
        double arg1 = 4.0 * FFarg[0];
        double arg2 = FFarg[1] * FFarg[1];
        double arg3 = FFarg[3]; //the third element of the 3rd dimension of the array
        double temp = (rr / arg2);
        double temp3 = temp * temp * temp;
        double rri3 = 1.0 / (temp3 + 0.5 * (1.0 - scaling) * (1.0 - scaling));
        double rri6 = rri3 * rri3;
        double term = arg1 * (rri3 * (rri3 - 1.0)) - arg3;
        double dlambda_term = scaling * arg1 * (rri6 * (2.0 * rri3 - 1.0));
        //printf("DOING LJ\n");
        result[0] = scaling * term; result[1] = scaling < 1.0 ? term + (1.0 - scaling) * dlambda_term : 0.0;
        break;
        //return result;
    }
    case 1:
    {
        double arg1 = 4.0 * FFarg[0];
        double arg2 = FFarg[1] * FFarg[1];
        double arg3 = FFarg[3];
        double temp = (rr / arg2);
        double rri3 = 1.0 / ((temp * temp * temp) + 0.5 * (1.0 - scaling) * (1.0 - scaling));
        result[0] = scaling * (4.0 * arg1 * (rri3 * (rri3 - 1.0)) - arg3); result[1] = 0.0;
        //printf("DOING BUCKINGHAM\n");
        //return result;
    }
    case 2:
    {
        double arg1 = 4.0 * FFarg[0];
        double arg2 = FFarg[1] * FFarg[1];
        double arg3 = FFarg[3];
        double temp = (rr / arg2);
        double rri3 = 1.0 / ((temp * temp * temp) + 0.5 * (1.0 - scaling) * (1.0 - scaling));
        result[0] = scaling * (4.0 * arg1 * (rri3 * (rri3 - 1.0)) - arg3); result[1] = 0.0;
        //printf("DOING MORSE\n");
        //return result; 
    }
    case 3:
    {
        double arg1 = 4.0 * FFarg[0];
        double arg2 = FFarg[1] * FFarg[1];
        double arg3 = FFarg[3];
        double temp = (rr / arg2);
        double rri3 = 1.0 / ((temp * temp * temp) + 0.5 * (1.0 - scaling) * (1.0 - scaling));
        result[0] = scaling * (4.0 * arg1 * (rri3 * (rri3 - 1.0)) - arg3); result[1] = 0.0;
        //return result;
    }
    case 4:
    {
        double arg1 = 4.0 * FFarg[0];
        double arg2 = FFarg[1] * FFarg[1];
        double arg3 = FFarg[3];
        double temp = (rr / arg2);
        double rri3 = 1.0 / ((temp * temp * temp) + 0.5 * (1.0 - scaling) * (1.0 - scaling));
        result[0] = scaling * (4.0 * arg1 * (rri3 * (rri3 - 1.0)) - arg3); result[1] = 0.0;
        //return result;
    }
    case 5:
    {
        double arg1 = 4.0 * FFarg[0];
        double arg2 = FFarg[1] * FFarg[1];
        double arg3 = FFarg[3];
        double temp = (rr / arg2);
        double rri3 = 1.0 / ((temp * temp * temp) + 0.5 * (1.0 - scaling) * (1.0 - scaling));
        result[0] = scaling * (4.0 * arg1 * (rri3 * (rri3 - 1.0)) - arg3); result[1] = 0.0;
        //return result;
    }
    default: 
    {
        printf("You are screwed");
        //return result;
    }
    }
    return result;
}
*/
static inline std::array<double,2> GPUpotentialVDW(std::array<double,4>& FFarg, const double scaling, const double rr)
{
    std::array<double,2> result;
    double arg1 = 4.0 * FFarg[0];
    double arg2 = FFarg[1] * FFarg[1];
    double arg3 = FFarg[3]; //the third element of the 3rd dimension of the array
    double temp = (rr / arg2);
    double temp3 = temp * temp * temp;
    double rri3 = 1.0 / (temp3 + 0.5 * (1.0 - scaling) * (1.0 - scaling));
    double rri6 = rri3 * rri3;
    double term = arg1 * (rri3 * (rri3 - 1.0)) - arg3;
    double dlambda_term = scaling * arg1 * (rri6 * (2.0 * rri3 - 1.0));
    //printf("DOING LJ\n");
    result[0] = scaling * term; result[1] = 0.0;
    return result;
}
