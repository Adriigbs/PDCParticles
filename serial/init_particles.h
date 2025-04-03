#ifndef INIT_PARTICLES_H
#define INIT_PARTICLES_H

typedef struct {
    double x, y;
    double vx, vy;
    double m;
    double gravity_x, gravity_y;
    int collided;
} particle_t;

// Function prototype
void init_particles(long seed, double side, long ncside, long long n_part, particle_t *par);

#endif // INIT_PARTICLES_H