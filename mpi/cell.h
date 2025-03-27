#ifndef CELL_H
#define CELL_H

#include "init_particles.h"


#define ROW_LOW(id, p, n) ((id < n % p) ? id * (n / p + 1) : id * (n / p) + n % p)
#define ROW_HIGH(id, p, n) ((id < n % p) ? (id + 1) * (n / p + 1) : (id + 1) * (n / p) + n % p)
#define ROW_SIZE(id, p, n) ((id < n % p) ? (n / p + 1) : (n / p))

typedef struct {
    double x, y;
    double m;
    long long n_particles;
    long long index;
    particle_t *particles;
} cell_t;

void add_particle_to_cell(cell_t *cell, particle_t particle);

void remove_particle_from_cell(cell_t *cell, long index);


#endif