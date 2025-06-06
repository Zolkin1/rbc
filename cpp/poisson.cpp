#include <memory>
#include <iostream>
#include <stdio.h>
#include <fstream>
#include <vector>
#include <string>
#include <math.h>
#include <algorithm>
#include <cstring>

#include "poisson.h"

// Poisson Variables
float *hgrid;
int h_iters;
const float h0 = 0.0f; // Set boundary level set value
const float dh0 = 1.0f; // Set dh Value  

/* Perform a bilinear interpolation on a 2-D grid */
float bilinear_interpolation(const float *grid, const float i, const float j){

    const float i1f = floorf(i);
    const float j1f = floorf(j);
    const float i2f = ceilf(i);
    const float j2f = ceilf(j);

    const int i1 = (int)i1f;
    const int j1 = (int)j1f;
    const int i2 = (int)i2f;
    const int j2 = (int)j2f;

    if((i1 != i2) && (j1 != j2)){
        const float f1 = (i2f - i) * grid[i1*JMAX+j1] + (i - i1f) * grid[i2*JMAX+j1];
        const float f2 = (i2f - i) * grid[i1*JMAX+j2] + (i - i1f) * grid[i2*JMAX+j2];
        return (j2f - j) * f1 + (j - j1f) * f2;
    }
    else if(i1 != i2){
        return (i2f - i) * grid[i1*JMAX+(int)j] + (i - i1f) * grid[i2*JMAX+(int)j];
    }
    else if(j1 != j2){
        return (j2f - j) * grid[(int)i*JMAX+j1] + (j - j1f) * grid[(int)i*JMAX+j2];
    }
    else{
        return grid[(int)i*JMAX+(int)j];
    }

};

/* Find Boundaries (Any Unoccupied Point that Borders an Occupied Point) */
void find_boundary(float *bound){
    
    // Set Border
    for(int i = 0; i < IMAX; i++){
        for(int j = 0; j < JMAX; j++){
            if(i==0 || i==(IMAX-1) || j==0 || j==(JMAX-1)) bound[i*JMAX+j] = 0.0f;
        }
    }

    float b0[IMAX*JMAX];
    memcpy(b0, bound, IMAX*JMAX*sizeof(float));
    for(int n = 0; n < IMAX*JMAX; n++){
        if(b0[n]==1.0f){
            if(b0[n+1]==-1.0f || 
                b0[n-1]==-1.0f || 
                b0[n+JMAX]==-1.0f || 
                b0[n-JMAX]==-1.0f || 
                b0[n+JMAX+1]==-1.0f || 
                b0[n-JMAX+1]==-1.0f || 
                b0[n+JMAX-1]==-1.0f || 
                b0[n-JMAX-1]==-1.0f) bound[n] = 0.0f;
        }
    }

};

/* Find Boundaries (Any Unoccupied Point that Borders an Occupied Point) */
void find_and_fix_boundary(float *grid, float *bound){
    
    // Set Border
    for(int i = 0; i < IMAX; i++){
        for(int j = 0; j < JMAX; j++){
            if(i==0 || i==(IMAX-1) || j==0 || j==(JMAX-1)) bound[i*JMAX+j] = 0.0f;
        }
    }

    float b0[IMAX*JMAX];
    memcpy(b0, bound, IMAX*JMAX*sizeof(float));
    for(int n = 0; n < IMAX*JMAX; n++){
        if(b0[n]==1.0f){
            if(b0[n+1]==-1.0f || 
                b0[n-1]==-1.0f || 
                b0[n+JMAX]==-1.0f || 
                b0[n-JMAX]==-1.0f || 
                b0[n+JMAX+1]==-1.0f || 
                b0[n-JMAX+1]==-1.0f || 
                b0[n+JMAX-1]==-1.0f || 
                b0[n-JMAX-1]==-1.0f) bound[n] = 0.0f;
        }
        if(!bound[n]) grid[n] = h0;
    }

};

/* Buffer Occupancy Grid with 2-D Robot Shape */
void inflate_occupancy_grid(float *bound, const float yawk){

    /* Step 1: Create Robot Kernel */
    const float length = 0.13f; // Crazyflie Drone
    const float width = 0.13f;
    
    const float D = sqrtf(length*length + width*width); // Max Robot Dimension to Define Kernel Size
    const int dim = ceilf((ceilf(D / DS) + 1.0f) / 2.0f) * 2.0f - 1.0f;
    float robot_grid[dim*dim];

    const float MOS = 1.2f;
    const float ar = MOS * length / 2.0f;
    const float br = MOS * width / 2.0f;

    const float expo = 2.0f;
    for(int i = 0; i < dim; i++){
        const float yi = (float)i*DS - D/2.0f;
        for(int j = 0; j < dim; j++){
            robot_grid[i*dim+j] = 0.0;
            const float xi = (float)j*DS - D/2.0f;
            const float xb = cosf(yawk)*xi + sinf(yawk)*yi;
            const float yb = -sinf(yawk)*xi + cosf(yawk)*yi;
            const float dist = powf(fabsf(xb/ar), expo) + powf(fabsf(yb/br), expo);
            if(dist <= 1.0f) robot_grid[i*dim+j] = -1.0f;
            //if(fabsf(xb/ar) <= 1.0f && fabsf(yb/br) <= 1.0f) robot_grid[i*dim+j] = -1.0f;
        }
    }

    /* Step 2: Convolve Robot Kernel with Occupancy Grid, Along the Boundary */
    float b0[IMAX*JMAX];
    memcpy(b0, bound, IMAX*JMAX*sizeof(float));

    int lim = (dim - 1)/2;
    for(int i = 1; i < IMAX-1; i++){
        int ilow = std::max(i - lim, 0);
        int itop = std::min(i + lim, IMAX);
        for(int j = 1; j < JMAX-1; j++){
            int jlow = std::max(j - lim, 0);
            int jtop = std::min(j + lim, JMAX);
            if(!b0[i*JMAX+j]){
                for(int p = ilow; p < itop; p++){
                    for(int q = jlow; q < jtop; q++){
                        bound[p*JMAX+q] += robot_grid[(p-i+lim)*dim+(q-j+lim)];
                    }
                }
            }
        }
    }
    for(int n = 0; n < IMAX*JMAX; n++){
        if(bound[n] < -1.0f) bound[n] = -1.0f;
    }

};

/* Compute Forcing Function for Average Flux */
void compute_fast_forcing_function(float *force, const float *bound){

    float perimeter_c = 0.0f;
    float area_c = 0.0f;
    
    for(int i = 1; i < IMAX-1; i++){
        for(int j = 1; j < JMAX-1; j++){
            if(bound[i*JMAX+j] == 0.0f) perimeter_c += DS;
            else if(bound[i*JMAX+j] < 0.0f) area_c += DS*DS;
        }
    }
    
    float perimeter_o = 2.0f*(float)IMAX*DS + 2.0f*(float)JMAX*DS + perimeter_c;
    float area_o = (float)IMAX*(float)JMAX*DS*DS - area_c;
    float force_o = -dh0 * perimeter_o / area_o * DS*DS;
    float force_c = 0.0f;
    if(area_c != 0.0f) force_c = dh0 * perimeter_c / area_c * DS*DS;
    
    for(int n = 0; n < IMAX*JMAX; n++){
        if(bound[n] > 0.0f){
            force[n] = force_o;
        }
        else if(bound[n] < 0.0f){
            force[n] = force_c;
        }
        else{
            force[n] = 0.0f;
        }
    }

};

/* Solve Poisson's Equation -- Checkerboard Successive Overrelaxation (SOR) Method */
int poisson(float *grid, const float *force, const float *bound, const float relTol = 1.0e-4f, const float N = 25.0f){
    
    const float w_SOR = 2.0f/(1.0f+sinf(M_PI/(N+1))); // This is the "optimal" value from Strikwerda, Chapter 13.5

    int iters = 0;
    const int max_iters = 10000;
    for(int n = 0; n < max_iters; n++){

        float rss = 0.0f;
       
        // Red Pass
        for(int i = 1; i < IMAX-1; i++){
            for(int j = 1; j < JMAX-1; j++){
                const bool red = (((i%2)+(j%2))%2) == 0;
                if(bound[i*JMAX+j] && red){
                    float dg = 0.0f;
                    dg += (grid[(i+1)*JMAX+j] + grid[(i-1)*JMAX+j]);
                    dg += (grid[i*JMAX+(j+1)] + grid[i*JMAX+(j-1)]);
                    dg -= force[i*JMAX+j];
                    dg /= 4.0f;
                    dg -= grid[i*JMAX+j];
                    grid[i*JMAX+j] += w_SOR * dg;
                    rss += dg * dg;
                }
            }
        }
        // Black Pass
        for(int i = 1; i < IMAX-1; i++){
            for(int j = 1; j < JMAX-1; j++){
                const bool black = (((i%2)+(j%2))%2) == 1;
                if(bound[i*JMAX+j] && black){
                    float dg = 0.0f;
                    dg += (grid[(i+1)*JMAX+j] + grid[(i-1)*JMAX+j]);
                    dg += (grid[i*JMAX+(j+1)] + grid[i*JMAX+(j-1)]);
                    dg -= force[i*JMAX+j];
                    dg /= 4.0f;
                    dg -= grid[i*JMAX+j];
                    grid[i*JMAX+j] += w_SOR * dg;
                    rss += dg * dg;
                }
            }
        }

        rss = sqrtf(rss) * DS;
        iters++;
        if(rss < relTol) break;

    }

    return iters;

};

/* Compute the Poisson Safety Function */
void solve_poisson_safety_function(float *grid, const float *occ){
    
    float *bound, *force;
    bound = (float *)malloc(IMAX*JMAX*sizeof(float));
    force = (float *)malloc(IMAX*JMAX*sizeof(float));

    memcpy(bound, occ, IMAX*JMAX*sizeof(float));
    find_boundary(bound);
    float yaw = 0;
    inflate_occupancy_grid(bound, yaw);
    find_and_fix_boundary(grid, bound);
    compute_fast_forcing_function(force, bound);

    const float h_RelTol = 1.0e-4f;
    h_iters = poisson(grid, force, bound, h_RelTol, 25.0f);

    free(bound);
    free(force);
    
};

float get_h0(const float *grid, const float rx, const float ry){

    // Fractional Index Corresponding to Current Position
    const float ir = ry / DS;
    const float jr = rx / DS;
    const float ic = fminf(fmaxf(0.0f, ir), (float)(IMAX-1)); // Saturated Because of Finite Grid Size
    const float jc = fminf(fmaxf(0.0f, jr), (float)(JMAX-1)); // Numerical Derivatives Shrink Effective Grid Size
    
    return bilinear_interpolation(grid, ic, jc);

};

// int main(void){
//
//     hgrid = (float *)malloc(IMAX*JMAX*sizeof(float));
//     for(int n = 0; n < IMAX*JMAX; n++) hgrid[n] = h0;
//
//     solve_poisson_safety_function(hgrid, occ);
//     float h0 = get_h0(hgrid, rx, ry);
//
//     free(hgrid);
//
//     return 0;
//
// }