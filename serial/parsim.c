#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <string.h>
#include <math.h>
#include "init_particles.h"


#define G 6.67408e-11  
#define DELTA 0.1
#define EPSILON 0.005

#define GRAV_FORCE(m1, m2, d) ((G * (m1) * (m2)) / ((d) * (d)))
#define SPEED(v_i, a, t) ((v_i) + (a) * (t))
#define POSITION(x_i, v_i, a, t) ((x_i) + (v_i) * (t) + 0.5 * (a) * (t) * (t))


typedef struct {
    double x, y;
    double m;
    long long n_particles;
    long long index;
    particle_t **particles;
} cell_t;


void parse_args(int argc, char **argv, long *seed, double *side, long *ncside, long long *n_part, long *n_steps) {
    if(argc != 6) {
        fprintf(stderr, "Usage: %s <seed> <side> <ncside> <n_part> <n_steps>\n", argv[0]);
        exit(1);
    }
    *seed = atol(argv[1]);
    *side = atof(argv[2]);
    *ncside = atol(argv[3]);
    *n_part = atoll(argv[4]);
    *n_steps = atol(argv[5]);
    
    

}



// Calculate the center of mass of each cell
void calculate_center_of_mass(particle_t *particles, long long n_part, long ncside, cell_t grid[][ncside], double cell_side, long seed) {

    for(long long i = 0; i < n_part; i++) {

        // Calculate the cell in which the particle is located
        long x = particles[i].x / cell_side;
        long y = particles[i].y / cell_side;

        // Sum the mass and position of the particle to the cell
        grid[y][x].m += particles[i].m;
        grid[y][x].x += particles[i].x * particles[i].m;
        grid[y][x].y += particles[i].y * particles[i].m;
        grid[y][x].n_particles++;
        
    }

    // Divide by the number of particles in the cell to get the center of mass
    for (long i = 0; i < ncside; i++) {
        for (long j = 0; j < ncside; j++) {

            if (grid[i][j].n_particles == 0) {
                grid[i][j].m = 0;
                grid[i][j].x = 0;
                grid[i][j].y = 0;
                continue;
            }

            grid[i][j].index = 0;
            grid[i][j].x /= grid[i][j].m;
            grid[i][j].y /= grid[i][j].m;
            grid[i][j].particles = (particle_t**) malloc(grid[i][j].n_particles * sizeof(particle_t*));
        }
    }
    
}


void split_particles_by_cell(particle_t *particles, long long n_part, long ncside, cell_t grid[][ncside], double cell_side, particle_t **particles_per_cell) {

    for (long long i = 0; i < n_part; i++) {

        long x = particles[i].x / cell_side;
        long y = particles[i].y / cell_side;

        grid[y][x].particles[grid[y][x].index] = &particles[i];
        grid[y][x].index++;
        
    }

}

void print_particles_and_cells(particle_t *particles, long long n_part, long ncside, cell_t grid[][ncside]) {
    // Print particles
    for (long long i = 0; i < n_part; i++) {
        printf("Particle %lld: mass=%.3f x=%.3f y=%.3f vx=%.3f vy=%.3f\n",
               i, particles[i].m, particles[i].x, particles[i].y, particles[i].vx, particles[i].vy);
    }

    // Print cells
    long cell_index = 0;
    for (long i = 0; i < ncside; i++) {
        for (long j = 0; j < ncside; j++) {
            printf("Cell %ld x: %.3f y: %.3f m: %.3f\n",
                   cell_index, grid[i][j].x, grid[i][j].y, grid[i][j].m);
            cell_index++;
        }
    }
}


int main(int argc, char **argv)
{
    double exec_time;
    long seed;
    double side;
    long ncside;
    long long n_part;
    long time_steps;
    
    double cell_side;

    // Parse arguments from command line
    parse_args(argc, argv, &seed, &side, &ncside, &n_part, &time_steps);

    
    // Calculate the side size of each cell
    cell_side = (double) side / ncside;

    // Allocate memory for particles 
    // Initialize particles
    particle_t *particles = (particle_t*) malloc(n_part * sizeof(particle_t));
    init_particles(seed, side, ncside, n_part, particles);

    cell_t grid[ncside][ncside];

    exec_time = -omp_get_wtime();
    
    // Calculate the center of mass of each cell
    calculate_center_of_mass(particles, n_part, ncside, grid, cell_side, seed);
    print_particles_and_cells(particles, n_part, ncside, grid);


    
    exec_time += omp_get_wtime();
    fprintf(stderr, "%.1fs\n", exec_time);
   

    return 0;
}