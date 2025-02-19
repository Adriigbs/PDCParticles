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



void update_particles(particle_t *particles, long long n_part, cell_t *grid_center_of_mass, particle_t **particles_per_cell, long *cell_offsets, long ncside, long cell_side) {

    for (long long i = 0; i < n_part; i++) {

        long x = particles[i].x / cell_side;
        long y = particles[i].y / cell_side;

        long particle_cell_index = y * ncside + x;

        double force_x = 0.0;
        double force_y = 0.0;

        // Add force from other cells
        for (long i = 0; i < ncside; i++) {
            for (long j = 0; j < ncside; j++) {

                long cell_index = i * ncside + j;

                if (particle_cell_index != cell_index) {
                    
                    long distance_x = grid_center_of_mass[cell_index].x - particles[i].x;
                    long distance_y = grid_center_of_mass[cell_index].y - particles[i].y;
                    long distance = sqrt(distance_x * distance_x + distance_y * distance_y);

                    force_x += GRAV_FORCE(particles[i].m, grid_center_of_mass[cell_index].m, distance) * (distance_x / distance);
                    force_y += GRAV_FORCE(particles[i].m, grid_center_of_mass[cell_index].m, distance) * (distance_y / distance);

                }
                
            }
        }

        // Add force from particles in the same cell
        long starting_index = cell_offsets[particle_cell_index] - grid_center_of_mass[particle_cell_index].n_particles;

        for (starting_index; starting_index < grid_center_of_mass[particle_cell_index].n_particles; starting_index++) {

            long distance_x = particles_per_cell[starting_index]->x - particles[i].x;
            long distance_y = particles_per_cell[starting_index]->y - particles[i].y;
            long distance = sqrt(distance_x * distance_x + distance_y * distance_y);

            force_x += GRAV_FORCE(particles[i].m, particles_per_cell[starting_index]->m, distance) * (distance_x / distance);
            force_y += GRAV_FORCE(particles[i].m, particles_per_cell[starting_index]->m, distance) * (distance_y / distance);

        }


        particles[i].gravity_x = force_x;
        particles[i].gravity_y = force_y;

    }


    // Update position and speed of particles
    for (long long i = 0; i < n_part; i++) {

        long acceleration_x = particles[i].gravity_x / particles[i].m;
        long acceleration_y = particles[i].gravity_y / particles[i].m;


        // Update position
        particles[i].x = POSITION(particles[i].x, particles[i].vx, acceleration_x, DELTA);
        particles[i].y = POSITION(particles[i].y, particles[i].vy, acceleration_y, DELTA);

        // Update speed
        particles[i].vx = SPEED(particles[i].vx, acceleration_x, DELTA);
        particles[i].vy = SPEED(particles[i].vy, acceleration_y, DELTA);
    }

}



// Calculate the center of mass of each cell
void calculate_center_of_mass(particle_t *particles, long long n_part, cell_t *grid_center_of_mass, long ncside, double cell_side, long seed) {

    printf("Cell side: %ld\n", cell_side);

    for(long long i = 0; i < n_part; i++) {

        // Calculate the cell in which the particle is located
        long x = particles[i].x / cell_side;
        long y = particles[i].y / cell_side;

        // Sum the mass and position of the particle to the cell
        printf("Particle is on cell %ld \n", y * ncside + x);
        grid_center_of_mass[y*ncside + x].x += particles[i].x * particles[i].m;
        grid_center_of_mass[y*ncside + x].y += particles[i].y * particles[i].m;
        grid_center_of_mass[y*ncside + x].m += particles[i].m;
        grid_center_of_mass[y*ncside + x].n_particles++;
        
    }

    // Divide by the number of particles in the cell to get the center of mass
    for (long i = 0; i < ncside; i++) {
        for (long j = 0; j < ncside; j++) {

            if (grid_center_of_mass[i*ncside + j].m != 0) {
                grid_center_of_mass[i*ncside + j].x /= grid_center_of_mass[i*ncside + j].m;
                grid_center_of_mass[i*ncside + j].y /= grid_center_of_mass[i*ncside + j].m;
            }
        }
    }
    
}


void particles_by_cell(particle_t *particles, long long n_part, cell_t *grid_center_of_mass, long ncside, long cell_side, particle_t **particles_per_cell, long *cell_offsets) {

    cell_offsets[0] = 0;
    for (long i = 1; i < ncside * ncside; i++) {
        cell_offsets[i] = cell_offsets[i-1] + grid_center_of_mass[i-1].n_particles;
    }

    // Sort particles by cell
    for (long long i = 0; i < n_part; i++) {

        long x = particles[i].x / cell_side;
        long y = particles[i].y / cell_side;

        long cell_index = y * ncside + x;
        long offset = cell_offsets[cell_index];
        particles_per_cell[offset] = &particles[i];
        cell_offsets[cell_index]++;

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

    // Allocate memory for grid to store center of mass of each cell
    cell_t *grid_center_of_mass = (cell_t*) calloc(ncside * ncside, sizeof(cell_t));

    // Calculate the side size of each cell
    cell_side = (double) side / ncside;

    // print args  REMOVE BEFORE SUBMISSION
    printf("%ld %lf %ld %lld %ld\n", seed, side, ncside, n_part, time_steps);


    // Allocate memory for particles 
    // Initialize particles
    particle_t *particles = (particle_t*) malloc(n_part * sizeof(particle_t));
    init_particles(seed, side, ncside, n_part, particles);

    for (long long i = 0; i < n_part; i++) {
        printf("%lf %lf %lf %lf %lf\n", particles[i].x, particles[i].y, particles[i].vx, particles[i].vy, particles[i].m);
    }


    exec_time = -omp_get_wtime();

    particle_t **particles_per_cell = (particle_t**) malloc(n_part * sizeof(particle_t*));
    long *cell_offsets = (long*) malloc(ncside * ncside * sizeof(long));


    //for (long i = 0; i < time_steps; i++) {


        calculate_center_of_mass(particles, n_part, grid_center_of_mass, ncside, cell_side, seed);
        particles_by_cell(particles, n_part, grid_center_of_mass, ncside, cell_side, particles_per_cell, cell_offsets);
        // Print center of mass of each cell
        for (long i = 0; i < ncside * ncside; i++) {
            // print Cell i x: y: m:
            printf("Cell %ld x: %.3lf y: %.3lf m: %.3lf\n", i, grid_center_of_mass[i].x, grid_center_of_mass[i].y, grid_center_of_mass[i].m);
        }

    //}
    
    free(particles);
    free(grid_center_of_mass);
    free(particles_per_cell);
    free(cell_offsets);
    exec_time += omp_get_wtime();
    fprintf(stderr, "%.1fs\n", exec_time);
   

    return 0;
}