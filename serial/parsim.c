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
    long long m;
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

    for(long long i = 0; i < n_part; i++) {

        // Calculate the cell in which the particle is located
        long x = particles[i].x / cell_side;
        long y = particles[i].y / cell_side;

        // Sum the mass and position of the particle to the cell
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


void particles_by_cell(particle_t *particles, long long n_part, 
    cell_t *grid_center_of_mass, long ncside, double cell_side, 
    particle_t **particles_per_cell, long *cell_offsets) {

    // Compute initial cell offsets (starting index in particles_per_cell)
    cell_offsets[0] = 0;
    for (long i = 1; i < ncside * ncside; i++) {
        cell_offsets[i] = cell_offsets[i - 1] + grid_center_of_mass[i - 1].n_particles;
    }

    // Store a copy of initial offsets for later reference
    long *initial_offsets = (long *)malloc(ncside * ncside * sizeof(long));
    if (!initial_offsets) {
        printf("Memory allocation failed\n");
        return;
    }
    memcpy(initial_offsets, cell_offsets, ncside * ncside * sizeof(long));

    // Sort particles into particles_per_cell and update cell_offsets
    for (long long i = 0; i < n_part; i++) {
        long x = (long)(particles[i].x / cell_side);
        long y = (long)(particles[i].y / cell_side);
        long cell_index = y * ncside + x;

        printf("row = %ld, col = %ld, cell index = %ld\n", y, x, cell_index);

        long offset = cell_offsets[cell_index];
        particles_per_cell[offset] = &particles[i];

        cell_offsets[cell_index]++;
    }

    // Restore cell_offsets to point to **starting positions**
    memcpy(cell_offsets, initial_offsets, ncside * ncside * sizeof(long));

    free(initial_offsets);
}


void remove_particle(particle_t *particles, long long *n_part, 
    particle_t **particles_per_cell, long *cell_offsets, 
    long ncside, double cell_side, long index_to_remove) {

    if (index_to_remove < 0 || index_to_remove >= *n_part) {
        printf("Invalid index for removal\n");
        return;
    }

    // Get the particle to remove
    particle_t *particle_to_remove = &particles[index_to_remove];
    printf("particle's coordinates: %lf %lf\n", particle_to_remove->x, particle_to_remove->y);

    // Find the cell of the particle
    long x = particle_to_remove->x / cell_side;
    long y = particle_to_remove->y / cell_side;
    long cell_index = y * ncside + x;
    printf("cell_index = %ld\n", cell_index);

    // Find the particle's position in particles_per_cell
    long start_offset = cell_offsets[cell_index];
    long end_offset = (cell_index + 1 < ncside * ncside) ? cell_offsets[cell_index + 1] : *n_part;
    printf("start_offset = %ld, end_offset = %ld\n", start_offset, end_offset);

    long pos_in_particles_per_cell = -1;
    for (long i = start_offset; i < end_offset; i++) {
        if (particles_per_cell[i] == particle_to_remove) {
            pos_in_particles_per_cell = i;
            break;
        }
    }

    if (pos_in_particles_per_cell == -1) {
        printf("Particle not found in particles_per_cell\n");
        return;
    }

    // Remove from particles_per_cell by shifting left
    for (long i = pos_in_particles_per_cell; i < end_offset - 1; i++) {
        particles_per_cell[i] = particles_per_cell[i + 1];
    }

    // Update cell_offsets for affected cells
    for (long i = cell_index + 1; i < ncside * ncside; i++) {
        cell_offsets[i]--;
    }

    // Swap with last particle in particles array (if not last already)
    if (index_to_remove != *n_part - 1) {
        particles[index_to_remove] = particles[*n_part - 1];

        // Update the swapped particle's reference in particles_per_cell
        particle_t *moved_particle = &particles[index_to_remove];
        long moved_x = moved_particle->x / cell_side;
        long moved_y = moved_particle->y / cell_side;
        long moved_cell_index = moved_y * ncside + moved_x;

        // Find the old reference in particles_per_cell and update it
        long moved_start_offset = cell_offsets[moved_cell_index];
        long moved_end_offset = (moved_cell_index + 1 < ncside * ncside) ? cell_offsets[moved_cell_index + 1] : *n_part;
        for (long i = moved_start_offset; i < moved_end_offset; i++) {
            if (particles_per_cell[i] == &particles[*n_part - 1]) {
                particles_per_cell[i] = moved_particle;
                break;
            }
        }
    }

    // Reduce the particle count
    (*n_part)--;
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
    cell_side = side / ncside;

    // print args  REMOVE BEFORE SUBMISSION
    printf("%ld %lf %ld %lld %ld %lf\n", seed, side, ncside, n_part, time_steps, cell_side);


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
        printf("---\n");
        for (long long i = 0; i < n_part; i++) {
            printf("%lf %lf %lf %lf %lf\n", particles_per_cell[i]->x, particles_per_cell[i]->y, particles_per_cell[i]->vx, particles_per_cell[i]->vy, particles_per_cell[i]->m);
        }
        remove_particle(particles, &n_part, particles_per_cell, cell_offsets, ncside, cell_side, 4);
        // TODO: Iterate over each particle and calculate the force for x and y of each particle with the particles in the same cell and the 8 adjacent cells
        // TODO: Update the position of each particle, maybe this part can be done in the same loop as the previous one
        // TODO: Check collisions and remove particles if necessary
        // TODO: Update cells

    //}
    printf("---\n");
    for (long long i = 0; i < n_part; i++) {
        printf("%lf %lf %lf %lf %lf\n", particles[i].x, particles[i].y, particles[i].vx, particles[i].vy, particles[i].m);
    }
    
    free(particles);
    free(grid_center_of_mass);
    free(particles_per_cell);
    free(cell_offsets);
    exec_time += omp_get_wtime();
    fprintf(stderr, "%.1fs\n", exec_time);
   

    return 0;
}