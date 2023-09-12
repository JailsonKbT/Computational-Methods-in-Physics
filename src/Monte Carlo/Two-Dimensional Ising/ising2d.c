

//  Created on December 27, 2022
//  Revised on Feb, 2023
//  Author: Jailson Santana
//  E-mail: Jailson.Oliveira.Fisica@outlook.com
//  Coded in C w std:c99
//  Compiled w open-mpi --version 4.1.2 linked to gcc(11.2.0)


/*|========================================================================================|*/
/*|                                                                                        |*/
/*|     This code integrates part of computational production developed during my          |*/
/*|     undergraduation research project.                                                  |*/
/*|                                                                                        |*/
/*|     -- Advisor: Felipe Mondaini, D. Sc.                                                |*/
/*|     -- Student: Jailson Oliveira Santana.                                              |*/
/*|                                                                                        |*/
/*|     Computational simulation of Bidimensional Ising Model for the Ferromagnetism       |*/
/*|     On a Two-dimensional Square Lattice. The simulation was performed in the           |*/
/*|     Microcanonical (NVE) and Canonical Ensemble (NVT) and is described by              |*/
/*|     Heisemberg model for the Ferromagnetism using the Metropolis algorithm.            |*/
/*|                                                                                        |*/
/*|========================================================================================|*/



#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <mpi.h>
#include <stdbool.h>
#include <unistd.h>
#include <omp.h>
#include <complex.h>


#define PARAMAG 1                   // boolean flag used to allow the system to generate a paramagnectic state for the initial frame.
#define RANDOM  0                   // boolean flag used to allow the system to generate a random state for the initial frame.
#define ALIGNED 0                   // boolean flag used to allow the system to generate a ordenated(up or down) state for the initial frame.
#define NVE 0                       // boolean flag used to choose the Micrcanonical Ensemble for the simulation.
#define NVT 1                       // boolean flag used to choose the Canonical Ensemble for the simulation.
#define USE_PARALL 1
#define USE_SERIAL 0
#define WITH_FIELD 0
#define ZERO_FIELD 1

long int SEED = -1;                             // A variable used to provide a 'seed' for random number generation.
double m_temp;                                  // Temporaly magnetization.
double mag;                                     // Total Magnectic Momentum by lattice point.
double E_temp;

double norm;
double Cv,Xmag;
double Eavg,E2avg,Evar;
double Mavg,M2avg,Mvar,Mabsavg;
double S_per_MCS,Savg;
int acceptance;
int cycle,count;
// E_temp = 0.0, m_temp = 0.0;
// double mag = 0.0;
// double E_temp = 0.0;

double E;
double T;              // Some variables to be used in a global scope.

#define J (int) 1.0                               // Coupling constant J_{ij} for exchange interactions between spin(i) and spin(j) on a chain (d=1 lattice).
#define N (int) 500                             // Number of spins of the linear dimension of the Lattice. The total Number of spin over the Lattice is N*N.
#define MCS (int) 1E3                           // Number of Monte Carlo Cycles used to provide independent realizations for the system.
#define Kb (double) 1.0//38E-23                         // Unit energy used in some definitions of statistical physics.
#define equilibration (int)  1E2                // The # of steps that will be take into account for discard before the MC steps for averages composition.
#define dT (double) 2E-1                        // Temperature step to be used in the Microcanonical Ensemble (NVE).
#define sigma (double) 1.0                      // Lattice parameter for the lattice frames generation in the Lattice.dump file that contains the set of frames.
#define T_init (double) 2.0                     // The initial value for the system temperature (ground state temperature) in the Micrcanonical ensemble.
#define T_final (double) 10.0                    // The final value for the system temperature (for NVE ensemble).
#define N_allocation (int) (N+1)                // Defining the space that will be allocated in virtual memory (Heap, specifically) for the Lattice use.
#define Flip_table_alloc (int) 10               // Defining the space that will be allocated in virtual memory (Heap, specifically) for the Spin-flip Table use.
#define beta (double) 1.0/(Kb*T)                // A usual variable that is considered into some theoretical definitions.
#define rnd_ab(min, max) (((int)rand() % (int)(((max) + 1) - (min))) + (min)) // By simplicity, we are defining these function in the Macro.


// Defining the value for the steps according with the choosen ensemble.
#if NVE
    #define steps ((T_final-T_init)/dT)*MCS
#endif

#if NVT
    #define steps MCS
#endif


// Defining the value for extern magnetic field.
#if WITH_FIELD
    double H = 1.0;
#endif

#if ZERO_FIELD
    double H = 0.0;
#endif


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

/*  Que vetor é o que
    **spin      -> valor do spin.
    *w          -> função de transição.
    *avg        -> vetor dos valores estatisticos.
    **ij_pbc    -> vetor para as condições periódicas de contorno.
*/

/*  what is each vector?
    **spin      -> spin value.
    *w          -> transition function.
    *avg        -> vector of statistical values.
    **ij_pbc    -> vector for boundary conditions.
*/


void Init_Configuration(int **spin, int **ij_pbc, double *avg);
void Periodic_Conditions(int **ij_pbc, int rnd_index_x, int rnd_index_y);
void NVE_Ensemble(int **spin, double *w, int **ij_pbc, double *avg);
void NVE_Ensemble(int **spin, double *w, int **ij_pbc, double *avg);
void Metropolis(int **spin, double *w, int **ij_pbc, double *avg);
int Neigh_Update(int **spin, int rnd_index_x, int rnd_index_y);
void prog_bar(char value[], int print_step, int N_iterations);
void Lattice_frames(int **spin, double *w, int timestep);
double w_transition(int **spin, double *w, int dE_flip);
void Output(double H, double T, double *avg);
void spin_dump(int **spin, double *w, int timestep);
double gauss(double mean, double d);
float ran2(long *idum);


void Periodic_Conditions(int **ij_pbc, int rnd_index_x, int rnd_index_y){
    int pbc_index = sizeof(N);

    do {
        ij_pbc[0][pbc_index] = pbc_index + 1; //ip --> S3
        ij_pbc[1][pbc_index] = pbc_index - 1; //im --> S1
        ij_pbc[2][pbc_index] = pbc_index + 1; //jp --> S2
        ij_pbc[3][pbc_index] = pbc_index - 1; //jm --> S4
    } while(pbc_index--);
    ij_pbc[0][N-1] = 0;
    ij_pbc[1][0] = (N-1);
    ij_pbc[2][N-1] = 0;
    ij_pbc[3][0] = (N-1);
}

int Neigh_Update(int **spin, int rnd_index_x, int rnd_index_y){
    int s1 = 0,s2 = 0,s3 = 0,s4 = 0,neigh = 0;

    if (rnd_index_x == 0){
        s1 = spin[N-1][rnd_index_y];
        s3 = spin[1][rnd_index_y];
    }
    else if (rnd_index_x == N-1){
        s1 = spin[N-2][rnd_index_y];
        s3 = spin[0][rnd_index_y];
    }
    else{
        s1 = spin[rnd_index_x-1][rnd_index_y];
        s3 = spin[rnd_index_x+1][rnd_index_y];
    }

    if (rnd_index_y == 0) {
        s2 = spin[rnd_index_x][1];
        s4 = spin[rnd_index_x][N-1];
    }
    else if (rnd_index_y == N-1) {
        s2 = spin[rnd_index_x][0];
        s4 = spin[rnd_index_x][N-2];
    }
    else{
        s2 = spin[rnd_index_x][rnd_index_y+1];
        s4 = spin[rnd_index_x][rnd_index_y-1];
    }

    neigh = (s1 + s2 + s3 + s4);
    return neigh;

}

double w_transition(int **spin, double *w, int dE_flip){

    if (**spin == 1){
        switch(dE_flip){
            case 1: if (dE_flip <= 0){ *w  = 1.0; }
                break;
            default: if (dE_flip > 0){ *w = exp(-beta*(dE_flip + H/J)); }
                break;
        }
        return *w;
    }
    
    if (**spin == -1){
        switch(dE_flip){
            case 1: if (dE_flip <= 0){ *w  = 1.0; }
                break;
            default: if (dE_flip > 0){ *w = exp(-beta*(dE_flip - H/J)); }
                break;
        }
        return *w;
    }

}


void Init_Configuration(int **spin, int **ij_pbc, double *avg){
    int i,j,rnd_index_x,rnd_index_y;
    long int seed = 0; seed = SEED;
    int E_si_sj_coupling=0, E_Hcoupling=0;
    // E_temp=0.0, m_temp=0.0;
    // Periodic_Conditions(ij_pbc,rnd_index_x,rnd_index_y);
    // Neigh_Update(spin,rnd_index_x,rnd_index_y);
    
    if (**spin == NULL || **spin == 0) {

        #if PARAMAG
            for (i=0; i<N; i++){
                for (j=0; j<N; j++){
                    if (i%2 == 0) {
                        spin[i][j] = 1;
                    }
                    else {
                        spin[i][j] = -1;
                    }
                    if (j%2 == 0 && i%2 != 0 || j%2 != 0 && i%2 == 0) {
                        spin[i][j] = 1;
                    }
                    else {
                        spin[i][j] = -1;
                    }
                    m_temp += spin[i][j];
                    E_Hcoupling += -H*(spin[i][j]);
                    E_si_sj_coupling += -J*spin[i][j] * (spin[ij_pbc[0][i]][j] + spin[i][ij_pbc[2][j]]);
                    E_temp += (E_si_sj_coupling + E_Hcoupling);
                    avg[0] += E_temp;
    	  		avg[1] += m_temp;
                }
            }
        #endif
    
        #if RANDOM
            for (i=0; i<N; i++){
                for (j=0; j<N; j++){
                    if (ran2(&seed) < 0.5) {
                        spin[i][j] = -1;
                    }
                    else {
                        spin[i][j] = 1;
                    }
                    m_temp += spin[i][j];
                    E_Hcoupling += -H*(spin[i][j]);
                    E_si_sj_coupling += -J*spin[i][j] * (spin[ij_pbc[0][i]][j] + spin[i][ij_pbc[2][j]]);
                    E_temp += (E_si_sj_coupling + E_Hcoupling);
                }
            }
        #endif
    
        #if ALIGNED
            for (i=0; i<N; i++){
                for (j=0; j<N; j++){
                    spin[i][j] = -1;
                    m_temp += spin[i][j];
                    E_Hcoupling += -H*(spin[i][j]);
                    E_si_sj_coupling += -J*spin[i][j] * (spin[ij_pbc[0][i]][j] + spin[i][ij_pbc[2][j]]);
                    E_temp += (E_si_sj_coupling + E_Hcoupling);
                }
            }
        #endif     

    }
}

void Metropolis(int **spin, double *w, int **ij_pbc, double *avg){
    double w_collect = 0.0;
    long int seed = 0; seed = SEED; double H_Coupling;
    int rnd_index_x = 0, rnd_index_y = 0, dE_flip = 0, timestep;
    // E_temp = 0.0, m_temp = 0.0;

    Periodic_Conditions(ij_pbc,rnd_index_x,rnd_index_y);
    Neigh_Update(spin,rnd_index_x,rnd_index_y);
    w_collect = w_transition(spin,w,dE_flip);
    
    #pragma omp_in_parallel num_threads(N_THREADS)
    for (int rnd_index_x=0; rnd_index_x<N; rnd_index_x++){
        for (int rnd_index_y=0; rnd_index_y<N; rnd_index_y++){

            rnd_index_x = (int) (ran2(&seed)*(double)N);
            rnd_index_y = (int) (ran2(&seed)*(double)N);

            H_Coupling = -H*(spin[rnd_index_x][rnd_index_y]);

            dE_flip = 2*J*spin[rnd_index_x][rnd_index_y]*Neigh_Update(spin,rnd_index_x,rnd_index_y) + H_Coupling;

            if (dE_flip <= 0 || (dE_flip > 0 && ran2(&seed) < w_collect)){
                spin[rnd_index_x][rnd_index_y] *= -1;
                avg[5] += spin[rnd_index_x][rnd_index_y];
                m_temp += 2*spin[rnd_index_x][rnd_index_y];
                E_temp += dE_flip;
                acceptance++;
                avg[0] += E_temp;
    	  		avg[1] += m_temp;
                avg[2] += E_temp*E_temp;
                avg[3] += m_temp*m_temp;
                avg[4] += m_temp;
                if (cycle >= equilibration) { count++; }
            }
            else{
                spin[rnd_index_x][rnd_index_y] = spin[rnd_index_x][rnd_index_y];
                // continue;
            }
        }
    }
    // Eavg  = avg[0]*norm;
	// E2avg = avg[2]*norm;
    // Mavg  = avg[1]*norm;
	// M2avg = avg[3]*norm;
    // Mabsavg = fabs(avg[4])*norm;
    // S_per_MCS = avg[5]*norm;
    // Savg = avg[5]/N/N;
    // acceptance = acceptance/N/N;
    // Cv = beta*beta*(E2avg - Eavg*Eavg);
	// Xmag = beta*(M2avg - Mavg*Mavg);
}

void NVE_Ensemble(int **spin, double *w, int **ij_pbc, double *avg){
    int dE_flip,timestep = 0;
    int eq_steps,print_step,i_avg;

    Init_Configuration(spin,ij_pbc,avg);
    Lattice_frames(spin,w,timestep);
    // Output(H,T,avg);
    for (T=T_init;  T<=T_final; T+=dT){

        prog_bar("\033[1;39m Executando: ", timestep, steps);
        for (eq_steps=0; eq_steps<equilibration; eq_steps++){
            Metropolis(spin,w,ij_pbc,avg);
        }
    
        for(i_avg=0; i_avg<5; i_avg++) { avg[i_avg] = 0.0; }

        count = 0.0, print_step += 1; mag = 0.0, E_temp = 0.0;

        #pragma omp_in_parallel num_threads(N_THREADS)
        for (cycle = 0; cycle < MCS; cycle++){
            SEED=rnd_ab(-10,10);
            #if WITH_FIELD
                H += dT;//*sin(cycle);
            #endif
            timestep += 1;
            Metropolis(spin,w,ij_pbc,avg);
            if (count = N*N) {
                Eavg  = avg[0]*norm;
            	E2avg = avg[2]*norm;
                Mavg  = avg[1]*norm;
    	        M2avg = avg[3]*norm;
                Mabsavg = fabs(avg[4])*norm;
                S_per_MCS = avg[5]*norm;
                Savg = avg[5]/N/N;
                Evar = (E2avg - Eavg*Eavg)/N/N;
    	        Mvar = (M2avg - Mabsavg*Mabsavg)/N/N;
                acceptance = acceptance/N/N;
                Cv = beta*beta*(E2avg - Eavg*Eavg);
	            Xmag = beta*(M2avg - Mavg*Mavg);
            }
            // avg[0] += E_temp;
	  		// avg[1] += m_temp;
            // avg[2] += E_temp*E_temp;
            // avg[3] += m_temp*m_temp;
            // avg[4] += fabs(m_temp);
            // Cv = beta*beta*(E2avg - Eavg*Eavg);
	        // Xmag = beta*(M2avg - Mavg*Mavg);
            // avg[5] += spin[i][j];
            // Lattice_frames(spin,w,timestep);
            if (cycle%1 == 100) { Lattice_frames(spin,w,timestep); }
        }
        Output(H,T,avg);
        **spin = 0;
        *w = 0;
    }

}

void NVT_Ensemble(int **spin, double *w, int **ij_pbc, double *avg){
    int dE_flip,timestep = 0;
    int eq_steps,print_step,i_avg;
    T = 3.0;

    Init_Configuration(spin,ij_pbc,avg);
    Lattice_frames(spin,w,timestep);
    // Output(H,T,avg);

    for(i_avg = 0; i_avg < 5; i_avg++) { avg[i_avg] = 0.0; }

    count = 0.0, print_step += 1;
    // mag = 0.0, E_temp = 0.0;

    #pragma omp_in_parallel num_threads(N_THREADS)
    for (cycle = 0; (cycle < MCS); cycle++){
        SEED+=rnd_ab(-1,1);
        #if WITH_FIELD
            H += sin(cycle);
        #endif
        timestep += 1;
        Metropolis(spin,w,ij_pbc,avg);
        if (count = N*N) {
            Eavg  = avg[0]*norm;
        	E2avg = avg[2]*norm;
            Mavg  = avg[1]*norm;
	        M2avg = avg[3]*norm;
            Mabsavg = fabs(avg[4])*norm;
            S_per_MCS = avg[5]*norm;
            Savg = avg[5]/N/N;
            Evar = (E2avg - Eavg*Eavg)/N/N;
    	    Mvar = (M2avg - Mabsavg*Mabsavg)/N/N;
            acceptance = acceptance/N/N;
            Cv = beta*beta*(E2avg - Eavg*Eavg);
            Xmag = beta*(M2avg - Mavg*Mavg);
        }
        // avg[0] += E_temp;
	  	// avg[1] += m_temp;
        // avg[2] += E_temp*E_temp;
        // avg[3] += m_temp*m_temp;
        // avg[4] += fabs(m_temp);
        // Cv = beta*beta*(E2avg - Eavg*Eavg);
	    // Xmag = beta*(M2avg - Mavg*Mavg);
        // Lattice_frames(spin,w,timestep);
        if (cycle%1 == 100) { Lattice_frames(spin,w,timestep); }
        Output(H,T,avg);
        **spin = 0;
        *w = 0;
        prog_bar("\033[1;39m Executando: ", timestep, steps);
    }
}

void Output(double H, double T, double *avg){
    static FILE *output1;
    // int acceptance = acceptance/N/N;

    norm = 1/((double) (MCS))/(N*N);  // divided by total number of cycles 
    // Eavg = avg[0]*norm;
	// E2avg = avg[2]*norm;
	// Mavg = avg[1]*norm;
	// M2avg = avg[3]*norm;
	// Mabsavg = avg[4]*norm;
	// S_per_MCS = avg[5]*norm;
	// all expectation values are per spin, divide by 1/n_spins/n_spins
	// Savg = avg[5]/N/N;
	// Evar = (E2avg - Eavg*Eavg)/N/N;
	// Mvar = (M2avg - Mabsavg*Mabsavg)/N/N;

    if (output1 == NULL){
        output1 = fopen("/Users/jailsonoliveira/Desktop/Monte_Carlo/"
        "IsingModelFerromag/Ising2DFerromag/ising.txt", "w");
        // #pragma omp single
        fprintf(output1, "E\tM\tE2\tM2\tMabs\tSsum\tEavg\tMavg\tE2avg"
        "\tM2avg\tMabsavg\tS_per_MCS\tSavg\tEvar\tMvar\tH\tT\tAccep\tCv\tXmag\n");
    }
    // #pragma omp single
    fprintf(output1,"%.4lf\t%.4lf\t%.4lf\t%.4lf\t%.4lf\t%.4lf\t%.4lf\t%.4lf"
            "\t%.4lf\t%.4lf\t%.4lf\t%.4lf\t%.4lf\t%.4lf\t%.4lf\t%.4lf\t%.4lf"
            "\t%i\t%.4lf\t%.4lf\n", avg[0], avg[1], avg[2], avg[3], avg[4], avg[5], Eavg,
            Mavg, E2avg, M2avg, Mabsavg, S_per_MCS, Savg, Evar, Mvar, H, T,
            acceptance,Cv,Xmag);

}

void Lattice_frames(int **spin, double *w, int timestep){
    
    int box_xlim = (double)N, box_ylim = (double)N, box_zlim = 1.0;
    int i, j, k, spin_id, Nspins = N*N; double Radius = 0.5;
    static double *x_lattice = NULL, *y_lattice = NULL;
    static FILE *output2;

    if(x_lattice == NULL && y_lattice == NULL){
        x_lattice = (double *) malloc((size_t) (Nspins*sizeof(double)));
        y_lattice = (double *) malloc((size_t) (Nspins*sizeof(double)));
    }

    if (output2 == NULL){
        output2 = fopen("/Users/jailsonoliveira/Desktop/Monte_Carlo/"
        "IsingModelFerromag/Ising2DFerromag/Lattice.dump", "w");
    }
    #pragma omp single
    fprintf(output2,"%s","ITEM: TIMESTEP\n");
    fprintf(output2,"%i\n",timestep);
    fprintf(output2,"%s","ITEM: NUMBER OF ATOMS\n");
    fprintf(output2,"%i\n",Nspins);
    fprintf(output2,"%s","ITEM: BOX BOUNDS f f f\n");
    fprintf(output2,"%i",(-1));               //    box_xlim
    fprintf(output2,"%s"," ");                //    box_xlim
    fprintf(output2,"%i",(box_xlim));         //    box_xlim
    fprintf(output2,"%s","\n");               //    box_xlim linebreak
    fprintf(output2,"%i",(-1));               //    box_ylim
    fprintf(output2,"%s"," ");                //    box_ylim
    fprintf(output2,"%i",(box_ylim));         //    box_ylim
    fprintf(output2,"%s","\n");               //    box_ylim linebreak
    fprintf(output2,"%i",(-1));               //    box_zlim
    fprintf(output2,"%s"," ");                //    box_zlim
    fprintf(output2,"%i",(box_zlim));         //    box_zlim
    fprintf(output2,"%s","\n");               //    box_zlim linebreak
    fprintf(output2,"%s","ITEM: ATOMS type id x y z fz spin\n");
    spin_id = 0;
    #pragma omp_in_parallel
    for(i=0; i<N; i++){
        x_lattice[i] = i*sigma;
        for(j=0; j<N; j++){
            y_lattice[j] = j*sigma;
            spin_id++;
            if (spin[i][j] == 1){
                #pragma omp single
                fprintf(output2,"%*s ",1, "1");
            }
            else{
                #pragma omp single
                fprintf(output2,"%*s ",1, "0");
            }
            if (spin[i][j] == 1){
                #pragma omp single
                fprintf(output2,"\t%i\t%.2e\t%.2e\t%.2e\t%.2e\t%.2e\t",
                        spin_id-1,x_lattice[i],y_lattice[j],0.0,1.0,1.0);
                fprintf(output2,"\n");
            }
            else {
                #pragma omp single
                fprintf(output2,"\t%i\t%.2e\t%.2e\t%.2e\t%.2e\t%.2e\t",
                        spin_id-1,x_lattice[i],y_lattice[j],0.0,-1.0,-1.0);
                fprintf(output2,"\n");
            }
        }
    }
}

void spin_dump(int **spin, double *w, int timestep){
    
    int i, j, spin_up = 0, Nspins = N*N, spin_down = 0;
    static FILE *output3;

    if (output3 == NULL){
        output3 = fopen("/Users/jailsonoliveira/Desktop/Monte_Carlo/"
        "IsingModelFerromag/Ising2DFerromag/spindump.txt", "w");
    }

    // fprintf(output3,"%s","NEW CONFIGURATION\n");
    #pragma omp_in_parallel
    for(i=0; i<N; i++){
        for(j=0; j<N; j++){
            if (spin[i][j] == 1){
                #pragma omp single
                fprintf(output3,"%*s\n",1, "1");
            }
            else{
                #pragma omp single
                fprintf(output3,"%*s\n",1, "-1");
            }
            if (spin[i][j] == 1){
                spin_up++;
                #pragma omp single
                fprintf(output3,"\t%i\t",spin_up);
            }
            else{
                spin_down++;
                #pragma omp single
                fprintf(output3,"\t%i\t",spin_down);
            }
        }
    }
}

double gauss(double mean, double d){
    static double t = 0.0;
    double x_gauss, v1, v2, r_gauss2, r_gauss;
    long int seed = SEED;

    if (t == 0.0) {
        do {
            v1 = 2.0 * ran2(&seed) - 1.0;
            v2 = 2.0 * ran2(&seed) - 1.0;
            r_gauss2 = v1 * v1 + v2 * v2;
            t = 0.0;
        }
        while (r_gauss2 >= 1.0);
        r_gauss = sqrt((-2.0*log(r_gauss2))/r_gauss2);
        t = v2 * r_gauss;
        return(mean + v1 * r_gauss * d);
    }
    else {
        x_gauss = t;
        t = 0.0;
        return(mean + x_gauss * d);
    }
}

void prog_bar(char value[], int print_step, int N_iterations){

    const int barsize = 72;                                     // The #(-8) of columns defined in the terminal (currently 80 is the default value).
    int bar_width = barsize - strlen(value);                    // This automatically adjusts the width of the progress-bar according to the value parameter passed.
    int N_character = (print_step * bar_width)/N_iterations;    // Number of characters that will be used to fill the progress-bar.
    int percent = (print_step * 100)/N_iterations;              // Percentage values used to indicate the progress of the run.

    fflush(stdout);                                             // To avoid blink the cursor during the progress of the bar.
    printf("\033[1;95m %s|", value);
    for (int i = 0; i < N_character; i++){                          // For every S in the main function this for loop fills de progress-bar with spaces and # of characters.
        // printf("\033[1;95m%c", "#");
        printf("\x1B[38;5;118m%s", "\u2588");                   //54 is purple, 199 is pink. \u2588 are filled block bar unicode character. \u258A are discrete block bar unicode character.
        // printf("%s", "\x1b[35m\u258A\033[0m");
    }
    printf("\033[1;39m %*c", bar_width - N_character + 1, '|');
    printf(" ");
    printf("\033[1;39m%3d %%\r", percent);                      // The percentual value printed every S value in range (0,steps).
}


float ran2(long *idum){
    int j_rnd;
    long k_rnd;
    static long idum2 = 123456789;
    static long iy = 0;
    static long iv[NTAB];
    double temp;

    if (*idum <= 0) {
        if (-(*idum) < 1) *idum=1;
        else *idum = -(*idum);
        idum2 = (*idum);
        for (j_rnd=NTAB+7; j_rnd>=0; j_rnd--){
            k_rnd = (*idum)/IQ1;
            *idum = IA1*(*idum - k_rnd*IQ1) - k_rnd*IR1;
            if (*idum < 0) *idum += IM1;
            if (j_rnd < NTAB) iv[j_rnd] = *idum;
        }
        iy = iv[0];
    }
    k_rnd = (*idum)/IQ1;
    *idum = IA1 * (*idum - k_rnd*IQ1) - k_rnd*IR1;
    if (*idum < 0) *idum += IM1;
    k_rnd = idum2/IQ2;
    idum2 = IA2 * (idum2 - k_rnd*IQ2) - k_rnd*IR2;
    if (idum2 < 0) idum2 += IM2;
    j_rnd = iy/NDIV;
    iy = iv[j_rnd] - idum2;
    iv[j_rnd] = *idum;
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



int main(int argc, char *argv[]){

    int tag = 1;
    double avg[6];
    int timestep = 0;
    int count,dE_flip;
    double t_inicial, t_final;
    int versao, subversao,  aux;
    int rank, n_procs, thread_level;
    double barsize = 50, max = steps;
    char maquina[MPI_MAX_PROCESSOR_NAME];
    int MASTER_THREAD_INIT, MASTER_THREAD_FINAL = 0;
    int row,col,cycle,eq_steps,print_step,ij_pbc_index;
    int percent = (cycle / T_final * 100), chras = (cycle * barsize / max);

    static double *w  = NULL;
    static int *ip = NULL, *im = NULL;
    static int *jp = NULL, *jm = NULL;
    static int **spin = NULL, **ij_pbc = NULL;

    if (spin == NULL && ij_pbc == NULL){
        spin   = (int **) malloc((size_t) (N_allocation*sizeof(int *)));                  // for allocate lines
        ij_pbc = (int **) malloc((size_t) (N_allocation*sizeof(int *)));                  // for allocate lines
        for (col=0; col<N; col++){
            spin[col] = (int *) malloc((size_t) (N_allocation*sizeof(int)));            // for allocate columns.
        }
        for (ij_pbc_index=0; ij_pbc_index<4; ij_pbc_index++){
            ij_pbc[ij_pbc_index] = (int *) malloc((size_t) (N_allocation*sizeof(int)));            // for allocate columns.
        }
    }

    if (ip == NULL && im == NULL && jp == NULL && jm == NULL){
        ip = (int *) malloc((size_t) (N_allocation*sizeof(int)));
        jp = (int *) malloc((size_t) (N_allocation*sizeof(int)));
        im = (int *) malloc((size_t) (N_allocation*sizeof(int)));
        jm = (int *) malloc((size_t) (N_allocation*sizeof(int)));
    }

    if (w == NULL){
        w = (double *) malloc((size_t) (Flip_table_alloc*sizeof(double)));            // for allocate lines
    }

    /* T R I N G  T O  A U T O M A T E  T H E  O P E N M P I  W I N D O W  T O  A L L O W  T H E  R E Q U I R E D  C O N N E C T I O N. */
    
    // system("osascript -e 'tell application \"System Events\" to click button \"Permitir\" of window 1 of application process \"Firewall\"'");
    // system("osascript -e 'tell application\"System Events\" to get name of every application process whose role description of window 1 is\"alert\\n'");
    // system("osascript -e 'tell application\"System Events\" to tell process\"System Preferences\" to get button button\"Permitir\"of sheet 1 of window\"Alert\"\n'");
    // system("osascript -e 'tell application\"System Events\" to get button button\"Permitir\"of sheet 1 of window\"Alert\"\n'");
    // system("osascript -e 'tell application \"System Events\" to click button \"Permitir\" of front window of application process \"UserNotificationCenter\n'");
    // system("osascript -e 'tell application\"System Events\" to tell process\"CoreServicesUIAgent\"if exists (button\"Permitir\"of front window) then click (button \"Permitir\"of front window) end if end tell\n'");

    #if USE_SERIAL

        #if NVE
            NVE_Ensemble(spin,w,ij_pbc,avg);
        #endif

        #if NVT
            NVT_Ensemble(spin,w,ij_pbc,avg);
        #endif

    #endif


    #if USE_PARALL

        // MPI_Init(&argc,&argv);
        MPI_Init_thread(&argc,&argv,MPI_THREAD_MULTIPLE,&thread_level);
        MPI_Get_version(&versao,&subversao);
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Get_processor_name(maquina, &aux);
        MPI_Comm_size(MPI_COMM_WORLD, &n_procs);
        t_inicial = MPI_Wtime();
        // system("osascript -e 'tell application \"System Events\" to click button \"Permitir\" of sheet 1 in window \"<nil>\"'");
        // system("osascript -e 'tell application \"System Events\" to keystroke \"Permitir\"'");
        // system("osascript -e 'tell application \"System Events\" to click button \"Permitir\" in front window of application process \"CoreServicesUIAgent\"'");
        // system("osascript -e 'tell application \"System Events\" to tell application process\"NSPanel\" to tell button\"Permitir\" of sheet 1 in window \"<nil>\" to perform action \"AXPress\"'");

        // system("osascript -e 'tell application \"UserNotificationCenter\" to tell application process\"<NSPanel>\" to tell button\"Permitir\" of sheet 1 in window \"<nil>\" to perform action \"AXPress\"'");
        // system("osascript -e 'tell application \"System Events\" to click button \"Permitir\" of window \"<nil>\" of application process \"UserNotificationCenter\"'");
        // system("osascript -e 'tell application \"System Events\" to click button \"Permitir\" of window 1 of application process \"SystemUIServer\"'");

        // system("osascript -e 'tell application \"System Events\" to tell process \"com.apple.AccountPolicyHelper\" to click button \"Permitir\" of sheet 1 in window \"<nil>\n\"'");
        // system("osascript -e 'tell application \"System Events\" to tell process \"com.apple.AccountPolicyHelper\" to click button \"Permitir\" of sheet 1 in window \"<nil>\n\"'");


        if (rank == 0){
    
            #if NVE
                NVE_Ensemble(spin,w,ij_pbc,avg);
            #endif

            #if NVT
                NVT_Ensemble(spin,w,ij_pbc,avg);
            #endif
        
            if (rank == 0){t_final = MPI_Wtime();}
                #pragma omp single
                printf("\n\n");
                printf(" ");
                printf("\e[38;5;27m MPI Version:               %d.%d \n", versao, subversao);
                printf(" ");
                printf("\e[38;5;27m Number of Tasks:           %d\n", n_procs);
                printf(" ");
                printf("\e[38;5;27m Rank:                      %d\n", rank);
                printf(" ");
                printf("\e[38;5;27m Executing on the Machine:  %s\n", maquina);
                printf("\n");
                printf(" ");
                printf("\e[38;5;118m Task Finished in %3.5f seconds\n",t_final-t_inicial);
                printf("\033[0;0m\n\n\n");
                // printf("\033[1;39m -------------------------------------------------------------------------"); should I use this one? or the one below? ;-D
                printf("\033[1;39m ============================================================================="); /* -- splitting the execution of each configuration to different SEED values */
                printf("\033[0;0m\n");
                printf("                    \x1B[38;5;147m\u2554\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550"
                        "\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550"
                        "\u2550\u2550\u2550\u2557\033[0;0m\n");
                printf("                    \x1B[38;5;147m\u2551                                     \u2551\n");
                printf("                    \x1B[38;5;147m\u2551 \x1B[1;35m PROGRAM EXECUTED SUCCESSFULLY !!! \x1B[0m \x1B[38;5;147m\u2551\n");
                printf("                    \x1B[38;5;147m\u2551                                     \u2551\n");
                printf("                    \x1B[38;5;147m\u255A\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550"
                        "\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550"
                        "\u2550\u2550\u2550\u255D\033[0;0m\n");

                // print_step=0;       /* -- resetting print step parameter to keep prog_bar working properly -- */
                printf("\n");

        }

        if (rank == 0){
            // #pragma omp parallel for num_threads(N_THREADS)
            for (MASTER_THREAD_INIT = 1; MASTER_THREAD_INIT < n_procs; MASTER_THREAD_INIT++){
                MPI_Recv(&Periodic_Conditions, 1, MPI_DOUBLE, MASTER_THREAD_INIT, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Recv(&Init_Configuration, 1, MPI_DOUBLE, MASTER_THREAD_INIT, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Recv(&Metropolis, 1, MPI_DOUBLE, MASTER_THREAD_INIT, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        }

        else{
            t_final = MPI_Wtime();
            MPI_Send(&Periodic_Conditions, 1, MPI_DOUBLE, MASTER_THREAD_FINAL, tag, MPI_COMM_WORLD);
            MPI_Send(&Init_Configuration, 1, MPI_DOUBLE, MASTER_THREAD_FINAL, tag, MPI_COMM_WORLD);
            MPI_Send(&Metropolis, 1, MPI_DOUBLE, MASTER_THREAD_FINAL, tag, MPI_COMM_WORLD);
        }

        MPI_Finalize();

    #endif


    printf("\n\n");
    // system("osascript -e 'tell application\"System Events\" to get name of every application process whose role description of window 1 is\"alert\\n'");
    // system("osascript -e 'tell application\"System Events\" to tell process\"System Preferences\" to get button button\"Permitir\"of sheet 1 of window\"Alert\"\n'");
    // system("osascript -e 'tell application\"System Events\" to click button\"Permitir\"of front window of application\"UserNotificationCenter\n'");
    // system("osascript -e 'tell application\"System Events\" to tell process\"CoreServicesUIAgent\"if exists (button\"Permitir\"of front window) then click (button \"Permitir\"of front window) end if end tell\n'");
    // printf("Press \x1B[1;4mENTER\x1B[0m to close window...\x1B[0m");
    // getchar();
    // #ifdef __APPLE__
    // system("osascript -e 'tell application\"Terminal\" to close windows 0\n'"); // It works only if you are in the macOS Terminal.
    // #elif _WIN32 | __linux__

    return 0;

    // #endif
// tell application "System Events" to tell process "System Preferences" ¬
//         to get button "OK" of sheet 1 of window "Built-in Retina Display"

}
