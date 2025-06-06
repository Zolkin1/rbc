//
// Created by zolkin on 6/6/25.
//

#ifndef POISSON_H
#define POISSON_H

#define IMAX 101 // Grid X Size
#define JMAX 101 // Grid Y Size
#define DS 0.05f // X-Y Grid Resolution

// TODO: Template
float bilinear_interpolation(const float *grid, const float i, const float j);
void solve_poisson_safety_function(float *grid, const float *occ);

#endif //POISSON_H
