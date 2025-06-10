//
// Created by zolkin on 6/6/25.
//

#ifndef POISSON_H
#define POISSON_H

#define IMAX 121 // Grid X Size
#define JMAX 121 // Grid Y Size
#define DS 0.05f // X-Y Grid Resolution

// TODO: Template
float bilinear_interpolation(const float *grid, const float i, const float j);
void solve_poisson_safety_function(float *grid, const float *occ);

#endif //POISSON_H
