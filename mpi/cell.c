#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <string.h>
#include <math.h>
#include "cell.h"



void init_grid(long ncside, cell_t **grid, long initial_size, int rows) {
    long i, j;

    for (i = 0; i < rows; i++) {
        for (j = 0; j < ncside; j++) {
            grid[i][j].m = 0.0;
            grid[i][j].x = 0.0;
            grid[i][j].y = 0.0;
            grid[i][j].n_particles = initial_size;
            grid[i][j].index = 0;
            grid[i][j].particles = (particle_t*) malloc(initial_size * sizeof(particle_t));
        
        }
    }
}

void add_particle_to_cell(cell_t *cell, particle_t particle) {
    if (cell->index >= cell->n_particles) {
        cell->particles = (particle_t*) realloc(cell->particles, 2 * cell->n_particles * sizeof(particle_t));
        cell->n_particles *= 2;
    }

    cell->particles[cell->index] = particle;
    cell->index++;
}

void remove_particle_from_cell(cell_t *cell, long index) {

    cell->particles[index] = cell->particles[cell->index - 1];
    cell->index--;
}
