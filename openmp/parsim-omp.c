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


void init_grid(long ncside, cell_t grid[][ncside], long initial_size) {
    long i, j;

    for (i = 0; i < ncside; i++) {
        for (j = 0; j < ncside; j++) {
            grid[i][j].m = 0.0;
            grid[i][j].x = 0.0;
            grid[i][j].y = 0.0;
            grid[i][j].n_particles = initial_size;
            grid[i][j].index = 0;
            grid[i][j].particles = (particle_t**) malloc(initial_size * sizeof(particle_t*));
        
        }
    }
}

void add_particle_to_cell(cell_t *cell, particle_t *particle) {
    if (cell->index >= cell->n_particles) {
        printf("Number of particles in cell: %ld\n", cell->index);
        cell->particles = (particle_t**) realloc(cell->particles, 2 * cell->n_particles * sizeof(particle_t*));
        cell->n_particles *= 2;
        printf("Reallocating cell\n");
    }

    cell->particles[cell->index] = particle;
    cell->index++;
}

void remove_particle_from_cell(cell_t *cell, long index) {

    
    cell->particles[index] = cell->particles[cell->index - 1];
    cell->index--;
}


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

        if (x >= ncside) x = ncside - 1;
        if (y >= ncside) y = ncside - 1;

        // Sum the mass and position of the particle to the cell

        grid[y][x].m += particles[i].m;
        grid[y][x].x += particles[i].x * particles[i].m;
        grid[y][x].y += particles[i].y * particles[i].m;

        // Add the particle to the cell
        add_particle_to_cell(&grid[x][y], &particles[i]);
    
    }

    // Divide by the number of particles in the cell to get the center of mass
    for (long i = 0; i < ncside; i++) {
        for (long j = 0; j < ncside; j++) {

            if (grid[i][j].n_particles == 0) {
                grid[i][j].m = 0;
                grid[i][j].x = 0;
                grid[i][j].y = 0;
            } else {
                grid[i][j].x /= grid[i][j].m;
                grid[i][j].y /= grid[i][j].m;

            }
        }
    }

}


void move_particle(cell_t *previous_cell, cell_t *new_cell, particle_t *particle, long index) {
    add_particle_to_cell(new_cell, particle);
    remove_particle_from_cell(previous_cell, index);


    previous_cell->m -= particle->m;
    new_cell->m += particle->m;

}


void update_center_of_mass(long ncside, cell_t grid[][ncside], particle_t *particle, double cell_side) {
    
    for (long i = 0; i < ncside; i++) {
        for (long j = 0; j < ncside; j++) {
            if (grid[i][j].index == 0) continue;

            grid[i][j].x = 0.0;
            grid[i][j].y = 0.0;

            for (long k = 0; k < grid[i][j].index; k++) {
                grid[i][j].x += grid[i][j].particles[k]->x * grid[i][j].particles[k]->m;
                grid[i][j].y += grid[i][j].particles[k]->y * grid[i][j].particles[k]->m;
            }

            grid[i][j].x /= grid[i][j].m;
            grid[i][j].y /= grid[i][j].m;
        }
    }
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

              
            }
        }
        
        // Add force from particles in the same cell
        cell_t *current_cell = &grid[y][x];
        
        for (long j = 0; j < current_cell->index; j++) {
            if (current_cell->particles[j] == &particles[i]) continue;

            double distance_x = current_cell->particles[j]->x - particles[i].x;
            double distance_y = current_cell->particles[j]->y - particles[i].y;
            double distance = sqrt(distance_x * distance_x + distance_y * distance_y) + 1e-10; // not sure if small number is necessary

            force_x += GRAV_FORCE(particles[i].m, current_cell->particles[j]->m, distance) * (distance_x / distance);

            force_y += GRAV_FORCE(particles[i].m, current_cell->particles[j]->m, distance) * (distance_y / distance);

        }

        particles[i].gravity_x = force_x;
        particles[i].gravity_y = force_y;
    }

    

    // Update position and speed of particles
    for (long i = 0; i < ncside; i++) {
        for (long j = 0; j < ncside; j++) {
            for (long k = 0; k < grid[i][j].index; k++) {
                particle_t *particle = grid[i][j].particles[k];

                double acceleration_x = particle->gravity_x / particle->m;
                double acceleration_y = particle->gravity_y / particle->m;

                // Update position
                particle->x = POSITION(particle->x, particle->vx, acceleration_x, DELTA);
                particle->y = POSITION(particle->y, particle->vy, acceleration_y, DELTA);

                // Update speed
                particle->vx = SPEED(particle->vx, acceleration_x, DELTA);
                particle->vy = SPEED(particle->vy, acceleration_y, DELTA);

                // Wrap-around on x
                if (particle->x < 0) {
                    particle->x += side;
                }
                else if (particle->x >= side) {
                    particle->x -= side;
                }

                // Apply wraparound on y
                if (particle->y < 0) {
                    particle->y += side;
                }
                else if (particle->y >= side) {
                    particle->y -= side;
                }

                // check if cell has changed
                long new_x = particle->x / cell_side;
                long new_y = particle->y / cell_side;

                if (new_x != j || new_y != i) {
                    move_particle(&grid[i][j], &grid[new_x][new_y], particle, k);
                }


            }
        }
    }

    update_center_of_mass(ncside, grid, particles, cell_side);


}


void remove_particle(particle_t *particles, long long *n_part, 
    long index_to_remove, long cell_index, cell_t *cell) {


    // remove particle from its cell
    remove_particle_from_cell(cell, cell_index);
    cell->m -= particles[index_to_remove].m;

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
    long long *collision_group = (long long *)malloc(*n_part * sizeof(long long));  // Track collision groups
    for (long long i = 0; i < *n_part; i++) collision_group[i] = -1;  // Initialize all as ungrouped

    for (long i = 0; i < ncside; i++) {
        for (long j = 0; j < ncside; j++) {
            cell_t *cell = &grid[i][j];


            if (cell->index < 2) continue;  // No collisions possible if only 0 or 1 particle

            for (long p1 = 0; p1 < cell->index; p1++) {
                for (long p2 = p1 + 1; p2 < cell->index; p2++) {
                    particle_t *particle1 = cell->particles[p1];
                    particle_t *particle2 = cell->particles[p2];

                    
                    double dx = particle1->x - particle2->x;
                    double dy = particle1->y - particle2->y;
                    double distance= sqrt(dx * dx + dy * dy) + 1e-10;
                    
                    
                    
                    if (distance < EPSILON) {  // Collision detected
                        long long id1 = (long long)(particle1 - particles);
                        long long id2 = (long long)(particle2 - particles);

                        // If neither particle is assigned to a group, create a new one
                        if (collision_group[id1] == -1 && collision_group[id2] == -1) {
                            collision_group[id1] = collision_group[id2] = collisions;
                            collisions++;  
                        } 
                        // If only one particle has a group, add the particle without group to the group
                        else if (collision_group[id1] == -1) {
                            collision_group[id1] = collision_group[id2];
                        } 
                        else if (collision_group[id2] == -1) {
                            collision_group[id2] = collision_group[id1];
                        } 
                        // If both particles have different groups, join them into the same group
                        else if (collision_group[id1] != collision_group[id2]) {
                            long old_group = collision_group[id2];
                            for (long long k = 0; k < *n_part; k++) {
                                if (collision_group[k] == old_group) {
                                    collision_group[k] = collision_group[id1];
                                }
                            }
                        }

                        // Remove both particles
                        remove_particle(particles, n_part, id2, p2, &grid[i][j]);
                        remove_particle(particles, n_part, id1, p1, &grid[i][j]);

                        
                    }
                }
            }
        }
    }

    free(collision_group);  // Clean up memory
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

    long initial_cell_size = n_part / (ncside * ncside) * 1.5; // might add some factor later
    if (initial_cell_size == 0) initial_cell_size = 1;

    cell_t grid[ncside][ncside];

    exec_time = -omp_get_wtime();

    init_grid(ncside, grid, initial_cell_size);
    
    calculate_center_of_mass(particles, n_part, ncside, grid, cell_side, seed);

    for (long i = 0; i < time_steps; i++) {
        // Calculate the center of mass of each cell
        //print_particles_and_cells(particles, n_part, ncside, grid);
        update_particles(particles, n_part, ncside, grid, cell_side, side);
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
    fprintf(stderr, "%.1fs\n", exec_time);
    return 0;
}
