#ifndef INIT_PARTICLES_H
#define INIT_PARTICLES_H

typedef struct cell_t cell_t;

typedef struct {
    char id;
    double x, y;
    double vx, vy;
    double m;
    double gravity_x, gravity_y;
} particle_t;

// Function prototype
void init_particles(long seed, double side, long ncside, long long n_part, particle_t *par);

void init_process_particles(long seed, double side, long ncside, long long n_part, cell_t **grid, int process_id, int num_processes);

#endif // INIT_PARTICLES_H