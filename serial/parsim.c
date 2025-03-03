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


void reset_grid(long ncside, cell_t grid[][ncside]) {
    for (long i = 0; i < ncside; i++) {
        for (long j = 0; j < ncside; j++) {
            grid[i][j].m = 0.0;
            grid[i][j].x = 0.0;
            grid[i][j].y = 0.0;
            grid[i][j].n_particles = 0;
            grid[i][j].index = 0;
            grid[i][j].particles = NULL;
        }
    }
}


void split_particles_by_cell(particle_t *particles, long long n_part, long ncside, cell_t grid[][ncside], double cell_side) {

    for (long long i = 0; i < n_part; i++) {

        long x = particles[i].x / cell_side;
        long y = particles[i].y / cell_side;

        if (x >= ncside) x = ncside - 1;
        if (y >= ncside) y = ncside - 1;

        grid[y][x].particles[grid[y][x].index] = &particles[i];
        grid[y][x].index++;
        
    }

}


// Calculate the center of mass of each cell
void calculate_center_of_mass(particle_t *particles, long long n_part, long ncside, cell_t grid[][ncside], double cell_side, long seed) {

    reset_grid(ncside, grid); // not sure if this will be necessary at the end

    for(long long i = 0; i < n_part; i++) {

        // Calculate the cell in which the particle is located
        long x = particles[i].x / cell_side;
        long y = particles[i].y / cell_side;

        if (x >= ncside) x = ncside - 1;
        if (y >= ncside) y = ncside - 1;

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
            } else {
                grid[i][j].index = 0;
                grid[i][j].x /= grid[i][j].m;
                grid[i][j].y /= grid[i][j].m;
                // malloc or realloc?
                grid[i][j].particles = (particle_t**) realloc(grid[i][j].particles, grid[i][j].n_particles * sizeof(particle_t*));
                // set grid[i][j].particles to NULL?
            }
        }
    }

    split_particles_by_cell(particles, n_part, ncside, grid, cell_side);
}

void update_particles(particle_t *particles, long long n_part, long ncside, cell_t grid[][ncside], double cell_side, double side) {

    for (long long i = 0; i < n_part; i++) {

        long x = particles[i].x / cell_side;
        long y = particles[i].y / cell_side;

        double force_x = 0.0;
        double force_y = 0.0;

        // Add force from neighboring cells
        for (long dx = -1; dx <= 1; dx++) {
            for (long dy = -1; dy <= 1; dy++) {

                if (dx == 0 && dy == 0) continue; // skip own cell

                long nx = (x + dx + ncside) % ncside; // Wrap around edges
                long ny = (y + dy + ncside) % ncside;

                //long cell_index = ny * ncside + nx;

                cell_t *neighbor_cell = &grid[ny][nx];

                if (neighbor_cell->m == 0) continue; // Skip empty cells

                // Compute force from center of mass of the cell
                double distance_x = neighbor_cell->x - particles[i].x;
                double distance_y = neighbor_cell->y - particles[i].y;

                if (x + dx >= ncside) distance_x += side;
                if (x + dx < 0) distance_x -= side;
                if (y + dy >= ncside) distance_y += side;
                if (y + dy < 0) distance_y -= side;

                double distance = sqrt(distance_x * distance_x + distance_y * distance_y) + 1e-10; // avoid division by zero

                force_x += GRAV_FORCE(particles[i].m, neighbor_cell->m, distance) * (distance_x / distance);
                force_y += GRAV_FORCE(particles[i].m, neighbor_cell->m, distance) * (distance_y / distance);

                if (i == 0) {
                    //printf("P%lld/C%ld mag: %.6lf fx: %.6lf fy: %.6lf\n", i, cell_index, distance, force_x, force_y);
                }
            }
        }
        
        // Add force from particles in the same cell
        cell_t *current_cell = &grid[y][x];
        
        for (long j = 0; j < current_cell->n_particles; j++) {
            if (current_cell->particles[j] == &particles[i]) continue;

            double distance_x = current_cell->particles[j]->x - particles[i].x;
            double distance_y = current_cell->particles[j]->y - particles[i].y;
            double distance = sqrt(distance_x * distance_x + distance_y * distance_y) + 1e-10; // not sure if small number is necessary

            force_x += GRAV_FORCE(particles[i].m, current_cell->particles[j]->m, distance) * (distance_x / distance);
            force_y += GRAV_FORCE(particles[i].m, current_cell->particles[j]->m, distance) * (distance_y / distance);

            if (i == 0) {
                //printf("P%lld/P%ld mag: %.3lf fx: %.3lf fy: %.3lf\n", i, j, distance, force_x, force_y);
            }
        }

        particles[i].gravity_x = force_x;
        particles[i].gravity_y = force_y;
    }

    // Update position and speed of particles
    for (long long i = 0; i < n_part; i++) {

        double acceleration_x = particles[i].gravity_x / particles[i].m;
        double acceleration_y = particles[i].gravity_y / particles[i].m;

        // Update position
        particles[i].x = POSITION(particles[i].x, particles[i].vx, acceleration_x, DELTA);
        particles[i].y = POSITION(particles[i].y, particles[i].vy, acceleration_y, DELTA);

        // Update speed
        particles[i].vx = SPEED(particles[i].vx, acceleration_x, DELTA);
        particles[i].vy = SPEED(particles[i].vy, acceleration_y, DELTA);

        // Wrap-around on x
        if (particles[i].x < 0) {
            particles[i].x += side;
        }
        else if (particles[i].x >= side) {
            particles[i].x -= side;
        }
    
        // Apply wraparound on y
        if (particles[i].y < 0) {
            particles[i].y += side;
        }
        else if (particles[i].y >= side) {
            particles[i].y -= side;
        }
    }

}


void remove_particle(particle_t *particles, long long *n_part, 
    long index_to_remove) {

    if (index_to_remove < 0 || index_to_remove >= *n_part) {
        printf("Invalid index for removal\n");
        return;
    }

    // Swap with last particle in the array (if not last already)
    if (index_to_remove != *n_part - 1) {
        particles[index_to_remove] = particles[*n_part - 1];
    }

    // Reduce the total particle count
    (*n_part)--;
}


long detect_collisions(particle_t *particles, long long *n_part, long ncside, cell_t grid[][ncside]) {
    long collisions = 0;

    for (long i = 0; i < ncside; i++) {
        for (long j = 0; j < ncside; j++) {

            cell_t *cell = &grid[i][j];

            if (cell->n_particles < 2) continue;  // No collisions if only 0 or 1 particle
            long p1 = 0;
            while (p1 < cell->n_particles) {
                int removed = 0;  // track if a particle is removed

                for (long p2 = p1 + 1; p2 < cell->n_particles; p2++) {
                    particle_t *particle1 = cell->particles[p1];
                    particle_t *particle2 = cell->particles[p2];

                    double dx = particle1->x - particle2->x;
                    double dy = particle1->y - particle2->y;
                    double distance_squared = dx * dx + dy * dy;

                    if (distance_squared < EPSILON * EPSILON) {  // Collision detected
                        double distance = sqrt(distance_squared);
                        //printf("[Collision] P%lld/P%lld Distance: %lf\n", (long long)(particle1 - particles), (long long)(particle2 - particles), distance);
                        collisions++;

                        // Remove both particles
                        remove_particle(particles, n_part, (long long)(particle2 - particles));
                        remove_particle(particles, n_part, (long long)(particle1 - particles));

                        // Update particle count in the cell
                        cell->n_particles -= 2;
                        removed = 1;
                        break;  // Restart loop after removal
                    }
                }

                if (!removed) {
                    p1++;  // Move to the next particle only if none were removed
                }
            }
        }
    }
    return collisions;
}


void print_particles_and_cells(particle_t *particles, long long n_part, long ncside, cell_t grid[][ncside]) {
    // Print particles
    for (long long i = 0; i < n_part; i++) {
        printf("Particle %lld: mass=%.6f x=%.6f y=%.6f vx=%.6f vy=%.6f\n",
               i, particles[i].m, particles[i].x, particles[i].y, particles[i].vx, particles[i].vy);
    }

    // Print cells
    long cell_index = 0;
    for (long i = 0; i < ncside; i++) {
        for (long j = 0; j < ncside; j++) {
            printf("Cell %ld x: %.6f y: %.6f m: %.6f\n",
                   cell_index, grid[i][j].x, grid[i][j].y, grid[i][j].m);
            cell_index++;
        }
    }
}


int main(int argc, char **argv)
{
    double exec_time;
    long total_num_collisions = 0;
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
    
    calculate_center_of_mass(particles, n_part, ncside, grid, cell_side, seed);

    for (long i = 0; i < time_steps; i++) {
        //printf("t=%ld\n", i);

        // Calculate the center of mass of each cell
        //print_particles_and_cells(particles, n_part, ncside, grid);
        update_particles(particles, n_part, ncside, grid, cell_side, side);
        calculate_center_of_mass(particles, n_part, ncside, grid, cell_side, seed);
        total_num_collisions += detect_collisions(particles, &n_part, ncside, grid);
    }

    printf("%.3lf %.3lf\n%ld\n", particles[0].x, particles[0].y, total_num_collisions);

    free(particles);
    for (long i = 0; i < ncside; i++) {
        for (long j = 0; j < ncside; j++) {
            free(grid[i][j].particles);
        }
    }
    
    exec_time += omp_get_wtime();
    //fprintf(stderr, "%.1fs\n", exec_time);

   
    return 0;
}
