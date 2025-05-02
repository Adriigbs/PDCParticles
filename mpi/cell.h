#ifndef CELL_H
#define CELL_H

#include "init_particles.h"


#define ROW_LOW(id, p, n) ((id < n % p) ? id * (n / p + 1) : id * (n / p) + n % p)
#define ROW_HIGH(id, p, n) ((id < n % p) ? (id + 1) * (n / p + 1) : (id + 1) * (n / p) + n % p)
#define ROW_SIZE(id, p, n) ((id < n % p) ? (n / p + 1) : (n / p))

// used to create an MPI datatype
typedef struct center_of_mass {
    double x, y;
    double m;
} center_of_mass;

typedef struct cell_t {
    double x, y;
    double m;
    long long n_particles;
    long long index;
    particle_t *particles;
} cell_t;

void add_particle_to_cell(cell_t *cell, particle_t particle);

void remove_particle_from_cell(cell_t *cell, long long index);

void init_grid(long ncside, cell_t **grid, long initial_size, int rows);

#endif