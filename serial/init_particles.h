#ifndef INIT_PARTICLES_H
#define INIT_PARTICLES_H

typedef struct {
    double x, y;
    long vx, vy;
    long long m;
} particle_t;

// Function prototype
void init_particles(long seed, double side, long ncside, long long n_part, particle_t *par);

#endif // INIT_PARTICLES_H