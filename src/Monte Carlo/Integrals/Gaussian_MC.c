
//  Created on August 31, 2022
//  Author: Jailson Santana
//  E-mail: Jailson.Oliveira.Fisica@outlook.com
//  Coded in C w std:c99
//  Compiled w gcc-10


/*||========================================================================================||*/
/*||                                                                                        ||*/
/*||     This code integrates part of computational production developed as a test in       ||*/
/*||     the course "computational methods in physics" in my undergraduation in Physics     ||*/
/*||                                                                                        ||*/
/*||     -- Professor: Felipe Mondaini, D. Sc.                                              ||*/
/*||     -- Student: Jailson Oliveira Santana.                                              ||*/
/*||                                                                                        ||*/
/*||     Monte Carlo computational implementation for calculating the improper integral     ||*/
/*||     of (e^-x^2) over the interval (-inf, inf). This integral has as a solution the     ||*/
/*||     value sqrt(pi) which is approximately 1.772453851 in its exact solution.           ||*/
/*||                                                                                        ||*/
/*||========================================================================================||*/


#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include <omp.h>

#define N 2E6                   // Number of MC steps.
#define xrange 8.0              // x-range domain to provide values for independent variable.
#define ymax 2.0                // y-range set of values ​​that can be accepted for approximation.

double x = 0.0, y = 0.0;
long int SEED = 10;


/*----------------------------------------------------*/
#define IM1 2147483563
#define IM2 2147483399
#define AM (1.0/IM1)
#define IMM1 (IM1-1)
#define IA1 40014
#define IA2 40692
#define IQ1 53668
#define IQ2 52774
#define IR1 12211
#define IR2 3791
#define NTAB 32
#define NDIV (1+IMM1/NTAB)
#define EPS 1.2e-7
#define RNMX (1.0-EPS)


float ran2(long *idum){
    int j; long k;
    static long idum2 = 123456789;
    static long iy = 0;
    static long iv[NTAB];
    double temp;

    if (*idum <= 0) {
        if (-(*idum) < 1) *idum=1;
        else *idum = -(*idum);
        idum2 = (*idum);
        for (j=NTAB+7; j>=0; j--){
            k = (*idum)/IQ1;
            *idum = IA1*(*idum - k*IQ1) - k*IR1;
            if (*idum < 0) *idum += IM1;
            if (j < NTAB) iv[j] = *idum;
        }
        iy = iv[0];
    }
    k = (*idum)/IQ1;
    *idum = IA1 * (*idum - k*IQ1) - k*IR1;
    if (*idum < 0) *idum += IM1;
    k = idum2/IQ2;
    idum2 = IA2 * (idum2 - k*IQ2) - k*IR2;
    if (idum2 < 0) idum2 += IM2;
    j = iy/NDIV;
    iy = iv[j] - idum2;
    iv[j] = *idum;
    if (iy < 1) iy += IMM1;
    if ((temp = AM*iy) > RNMX) return RNMX;
    else return temp;
}

#undef IM1
#undef IM2
#undef AM
#undef IMM1
#undef IA1
#undef IA2
#undef IQ1
#undef IQ2
#undef IR1
#undef IR2
#undef NTAB
#undef NDIV
#undef EPS
#undef RNMX
/*----------------------------------------------------*/


int main(int argc, char *argv[]){
    long int seed = SEED;
    double x_sample, y_sample, P_A, P_B;
    double Area_Under_Function, Area_Outside_Function, N_acceptances, N_rejects;

    static double *threshold = NULL;
    if (threshold == NULL) { threshold = (double *) malloc((size_t) (N*sizeof(double))); }

    // #pragma omp parallel
    for (int i=0; i<N; i++){
        Area_Under_Function = 0.0; Area_Outside_Function = 0.0;
        x = xrange*(ran2(&seed));  y = ymax*(ran2(&seed));
        x_sample = x; y_sample = y;
        threshold[i] += exp(-x_sample*x_sample);
        if (y <= threshold[i]) { N_acceptances++; }
        else { N_rejects++; }
        P_A = N_acceptances/N;
        P_B = N_rejects/N;
        Area_Under_Function += 2*(xrange*ymax)*P_A;
        Area_Outside_Function += 2*(xrange*ymax)*P_B; // It must be (xrange*ymax)*2. In the current configuration, (8x2)x2, so 32 is the total area of the square.
        threshold[i] = 0.0;
    }
    printf("\n\n");
    printf("\x1B[38;5;26mP(A) =   %1.8lf  ----> The probability of the chosen point being in the Area under the function.\n",P_A);
    printf("\x1B[38;5;26mP(B) =   %1.8lf  ----> The probability of the chosen point is outside the Area under the function.\n",P_B);
    printf("\033[1;4;92;118mArea(funct.) Aprox.:  %1.8lf\033[0;0m      \033[1;1;92;118m---->\033[0;0m \033[1;4;5;92;118mTHE VALUE OF"
            " THE INTEGRAL!""\n\033[0;0m",Area_Under_Function);
    printf("\033[36;118mArea(Square) Aprox.:  %1.7lf      ----> Area inside the box and outside the function""\n",Area_Outside_Function);
    printf("\x1B[38;35;26mP(A) + P(B)            =  %1.8lf  ----> The total (normalized) probability.""\n",P_B + P_A);
    printf("\x1B[38;35;26mArea(f) + Area(Square) =  %1.7lf  ----> Area inside the box (outside and inside the function)""\033[0;0m\n\n\n",
            (Area_Outside_Function + Area_Under_Function));
    return 0;
}
