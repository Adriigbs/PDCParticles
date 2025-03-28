#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <string.h>
#include <math.h>
#include <mpi.h>
#include "init_particles.h"
#include "cell.h"


#define G 6.67408e-11  
#define DELTA 0.1
#define EPSILON 0.005

#define GRAV_FORCE(m1, m2, d) ((G * (m1) * (m2)) / ((d) * (d)))
#define SPEED(v_i, a, t) ((v_i) + (a) * (t))
#define POSITION(x_i, v_i, a, t) ((x_i) + (v_i) * (t) + 0.5 * (a) * (t) * (t))


// Macros for cell indexing
#define ROW_LOW(id, p, n) ((id < n % p) ? id * (n / p + 1) : id * (n / p) + n % p)
#define ROW_HIGH(id, p, n) ((id < n % p) ? (id + 1) * (n / p + 1) : (id + 1) * (n / p) + n % p)
#define ROW_SIZE(id, p, n) ((id < n % p) ? (n / p + 1) : (n / p))


MPI_Datatype MPI_particle_t;
MPI_Datatype MPI_cell_t;

void create_mpi_cell() {

    int block_lengths[3] = {1, 1, 1};
    MPI_Aint displacements[3];
    MPI_Datatype types[3] = {MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE};

    displacements[0] = offsetof(cell_t, x);
    displacements[1] = offsetof(cell_t, y);
    displacements[2] = offsetof(cell_t, m);

    MPI_Type_create_struct(3, block_lengths, displacements, types, &MPI_cell_t);
    MPI_Type_commit(&MPI_cell_t);
}

void create_mpi_particle() {
    int block_lengths[5] = {1, 1, 1, 1, 1};
    MPI_Aint displacements[5];
    MPI_Datatype types[5] = {MPI_INT, MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE};

    displacements[0] = offsetof(particle_t, id);
    displacements[1] = offsetof(particle_t, x);
    displacements[2] = offsetof(particle_t, y);
    displacements[3] = offsetof(particle_t, vx);
    displacements[4] = offsetof(particle_t, vy);

    MPI_Type_create_struct(5, block_lengths, displacements, types, &MPI_particle_t);
    MPI_Type_commit(&MPI_particle_t);
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

        // Skip disabled particles
        if (particles[i].m == 0) continue;

        long x = particles[i].x / cell_side;
        long y = particles[i].y / cell_side;

        if (x >= ncside) x = ncside - 1;
        if (y >= ncside) y = ncside - 1;

        // maybe usar um lock por cell?
        {
            grid[y][x].particles[grid[y][x].index] = particles[i];
            grid[y][x].index++;
        }
        
    }

}

/*
// Calculates center of mass, allocates grid particles and reallocates main particle array
long long divide_by_processes(particle_t *particles, long long n_part, long ncside,
                              cell_t grid[][ncside], double cell_side, int id, int p) {
    
    long long new_n_part = 0;
    reset_grid(ncside, grid);
    
    for (long long i = 0; i < n_part; i++) {

        // Skip disabled particles
        if (particles[i].m == 0) continue;

        long x = particles[i].x / cell_side;
        long y = particles[i].y / cell_side;

        if (x >= ncside) x = ncside - 1;
        if (y >= ncside) y = ncside - 1;

        // If process is responsible for this grid's row
        if (y % p == id) {
            grid[y][x].m += particles[i].m;
            grid[y][x].x += particles[i].x * particles[i].m;
            grid[y][x].y += particles[i].y * particles[i].m;
            grid[y][x].n_particles++;
        }
    }

    // Compute center of mass for each cell in the rows handled by this process
    for (long i = id; i < ncside; i += p) {
        for (long j = 0; j < ncside; j++) {

            if (grid[i][j].n_particles == 0) {
                grid[i][j].m = 0;
                grid[i][j].x = 0;
                grid[i][j].y = 0;
            } else {
                grid[i][j].index = 0;
                grid[i][j].x /= grid[i][j].m;
                grid[i][j].y /= grid[i][j].m;
                grid[i][j].particles = (particle_t**) realloc(grid[i][j].particles, grid[i][j].n_particles * sizeof(particle_t*));
            }
        }
    }

    // Buffers for sending and receiving row data
    double *current_row_data = (double*) malloc(ncside * 3 * sizeof(double));
    double *recv_up = (double*) malloc(ncside * 3 * sizeof(double));
    double *recv_down = (double*) malloc(ncside * 3 * sizeof(double));

    for (long i = id; i < ncside; i += p) {
        for (long j = 0; j < ncside; j++) {
            current_row_data[j * 3 + 0] = grid[i][j].m;
            current_row_data[j * 3 + 1] = grid[i][j].x;
            current_row_data[j * 3 + 2] = grid[i][j].y;
        }

        int above = (i - 1 + ncside) % ncside;
        int below = (i + 1 + ncside) % ncside;

        int rank_above = above % p;
        int rank_below = below % p;

        MPI_Status status;
        // if they don't belong to the same process, sendrecv
        if (rank_above != id) {
            MPI_Sendrecv(current_row_data, ncside * 3, MPI_DOUBLE, rank_above, 0,
                        recv_down, ncside * 3, MPI_DOUBLE, rank_below, 0,
                        MPI_COMM_WORLD, &status);
        }
        if (rank_below != id) {
            MPI_Sendrecv(current_row_data, ncside * 3, MPI_DOUBLE, rank_below, 0,
                        recv_up, ncside * 3, MPI_DOUBLE, rank_above, 0,
                        MPI_COMM_WORLD, &status);
        }
        
        for (long j = 0; j < ncside; j++) {
            grid[above][j].m = recv_up[j * 3 + 0];
            grid[above][j].x = recv_up[j * 3 + 1];
            grid[above][j].y = recv_up[j * 3 + 2];

            grid[below][j].m = recv_down[j * 3 + 0];
            grid[below][j].x = recv_down[j * 3 + 1];
            grid[below][j].y = recv_down[j * 3 + 2];
        }
    }

    // Free allocated memory
    free(current_row_data);
    free(recv_up);
    free(recv_down);

    // Constructs particles array for each cell that belongs to this process
    for (long long i = 0; i < n_part; i++) {

        // Skip disabled particles
        if (particles[i].m == 0) continue;

        long x = particles[i].x / cell_side;
        long y = particles[i].y / cell_side;

        if (x >= ncside) x = ncside - 1;
        if (y >= ncside) y = ncside - 1;

        // If process is responsible for this grid's row
        if (y % p == id) {
            particles[new_n_part++] = particles[i];

            grid[y][x].particles[grid[y][x].index] = &particles[new_n_part - 1];
            grid[y][x].index++;
        }
    }

    // reallocates the necessary memory for the particles array
    particles = (particle_t*) realloc(particles, new_n_part * sizeof(particle_t));
    return new_n_part; 
}

// Calculate the center of mass of each cell
void calculate_center_of_mass(particle_t *particles, long long n_part, long ncside, cell_t grid[][ncside], double cell_side, long seed) {

    reset_grid(ncside, grid); // not sure if this will be necessary at the end

    for(long long i = 0; i < n_part; i++) {

        // Skip disabled particles
        if (particles[i].m == 0) continue;

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
                grid[i][j].particles = (particle_t**) realloc(grid[i][j].particles, grid[i][j].n_particles * sizeof(particle_t*));
            }
        }
    }

    split_particles_by_cell(particles, n_part, ncside, grid, cell_side);
}

void update_particles(particle_t *particles, long long n_part, long ncside,
                      cell_t grid[][ncside], double cell_side, double side, int id, int p) {

    // Iterate over every cell belonging to this process
    for (long x = 0; x < ncside; x++) {
        for (long y = id; y < ncside; y += p) {
            
            cell_t *cell = &grid[y][x];
            particle_t **particles = cell->particles;
            
            // Iterate over every particle from each cell
            for (long long i = 0; i < cell->n_particles; i++) {

                if (particles[i]->m == 0) continue;

                double px = particles[i]->x;
                double py = particles[i]->y;
                double pm = particles[i]->m;

                double force_x = 0.0;
                double force_y = 0.0;

                // Add force from other particles in the same cell
                for (long long j = 0; j < cell->n_particles; j++) {

                    if (i == j) continue;
                    if (particles[j]->m == 0) continue;

                    double distance_x = cell->particles[j]->x - px;
                    double distance_y = cell->particles[j]->y - py;
                    double distance = sqrt(distance_x * distance_x + distance_y * distance_y) + 1e-10; // not sure if small number is necessary

                    force_x += GRAV_FORCE(pm, cell->particles[j]->m, distance) * (distance_x / distance);
                    force_y += GRAV_FORCE(pm, cell->particles[j]->m, distance) * (distance_y / distance);

                }

                // Add force from neighboring cells
                for (long dx = -1; dx <= 1; dx++) {
                    for (long dy = -1; dy <= 1; dy++) {

                        if (dx == 0 && dy == 0) continue; // skip own cell

                        long nx = (x + dx + ncside) % ncside; // Wrap around edges
                        long ny = (y + dy + ncside) % ncside;

                        cell_t *neighbor_cell = &grid[ny][nx];

                        // Compute force from center of mass of the cell
                        double distance_x = neighbor_cell->x - px;
                        double distance_y = neighbor_cell->y - py;

                        if (x + dx >= ncside) distance_x += side;
                        if (x + dx < 0) distance_x -= side;
                        if (y + dy >= ncside) distance_y += side;
                        if (y + dy < 0) distance_y -= side;

                        double distance = sqrt(distance_x * distance_x + distance_y * distance_y) + 1e-10; // avoid division by zero

                        force_x += GRAV_FORCE(pm, neighbor_cell->m, distance) * (distance_x / distance);
                        force_y += GRAV_FORCE(pm, neighbor_cell->m, distance) * (distance_y / distance);

                        if (i == 0) {
                            //printf("P%lld/C%ld mag: %.6lf fx: %.6lf fy: %.6lf\n", i, cell_index, distance, force_x, force_y);
                        }
                    }
                }

                particles[i]->gravity_x = force_x;
                particles[i]->gravity_y = force_y;
            }

        }

    }

    // Update position and speed of particles
    for (long x = 0; x < ncside; x++) {
        for (long y = id; y < ncside; y += p) {

            cell_t *cell = &grid[y][x];
            particle_t **particles = cell->particles;

            // Iterate over every particle from each cell
            for (long long i = 0; i < cell->n_particles; i++) {

                if (particles[i]->m == 0) continue;

                double acceleration_x = particles[i]->gravity_x / particles[i]->m;
                double acceleration_y = particles[i]->gravity_y / particles[i]->m;
    
                // Update position
                particles[i]->x = POSITION(particles[i]->x, particles[i]->vx, acceleration_x, DELTA);
                particles[i]->y = POSITION(particles[i]->y, particles[i]->vy, acceleration_y, DELTA);
    
                // Update speed
                particles[i]->vx = SPEED(particles[i]->vx, acceleration_x, DELTA);
                particles[i]->vy = SPEED(particles[i]->vy, acceleration_y, DELTA);
    
                // Wrap-around
                particles[i]->x = fmod(particles[i]->x + side, side);
                particles[i]->y = fmod(particles[i]->y + side, side);

                // TODO: if new position is in different row, send it (and receive)
            }  
        }
    }

}


void disable_particle(particle_t *particles, long long *n_part, 
    long long index_to_remove) {

    if (index_to_remove < 0 || index_to_remove >= *n_part) {
        printf("Invalid index for removal %lld\n", index_to_remove);
        return;
    }

    particles[index_to_remove].m = 0;

}


long detect_collisions(particle_t *particles, long long *n_part, long ncside, cell_t grid[][ncside]) {
    long collisions = 0;
    int *to_remove = (int*) calloc(*n_part, sizeof(int));

    for (long i = 0; i < ncside; i++) {
        for (long j = 0; j < ncside; j++) {
            cell_t *cell = &grid[i][j];

            if (cell->n_particles < 2) continue;  // No collisions possible if only 0 or 1 particle

            for (long p1 = 0; p1 < cell->n_particles; p1++) {
                if (cell->particles[p1]->m == 0) continue;  // Skip disabled particles
                for (long p2 = p1 + 1; p2 < cell->n_particles; p2++) {
                    if (cell->particles[p2]->m == 0) continue;  // Skip disabled particles
                    particle_t *particle1 = cell->particles[p1];
                    particle_t *particle2 = cell->particles[p2];
                    double dx = particle1->x - particle2->x;
                    double dy = particle1->y - particle2->y;
                    double distance = sqrt(dx * dx + dy * dy) + 1e-10;

                    if (distance < EPSILON) {  // Collision detected

                        long long id1 = (long long)(particle1 - particles);
                        long long id2 = (long long)(particle2 - particles);
                        //printf("   [Collision] P%lld and P%lld (Distance: %lf)\n", id1, id2, distance);
                        if (to_remove[id1] == 0 && to_remove[id2] == 0) {
                            collisions++;
                        }
                        
                        to_remove[id1] = to_remove[id2] = 1;

                        disable_particle(particles, n_part, id1);
                        disable_particle(particles, n_part, id2);
                    }
                }
            }

        }
    }

    free(to_remove);  // Clean up memory
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
}*/


void calculate_center_of_mass(long ncside, cell_t **grid, int id, int p) {

    long size = ROW_SIZE(id, p, ncside);

    for (long i = 0; i < size; i++) {
        for (long j = 0; j < ncside; j++) {
            cell_t *cell = &grid[i][j];
            particle_t *particles = cell->particles;

            cell->m = 0.0;
            cell->x = 0.0;
            cell->y = 0.0;

            for (long idx = 0; idx < cell->index; idx++) {
                cell->m += particles[idx].m;
                cell->x += particles[idx].x * particles[idx].m;
                cell->y += particles[idx].y * particles[idx].m;
            }
        }
    }

    // Divide by the number of particles in the cell to get the center of mass
    for (long i = 0; i < size; i++) {
        for (long j = 0; j < ncside; j++) {
            if (grid[i][j].m > 0) {
                grid[i][j].x /= grid[i][j].m;
                grid[i][j].y /= grid[i][j].m;
            }
        }
    }
}


void update_particles(long long n_part, long ncside,
                cell_t **grid, double cell_side, double side, int id, int p) {

    MPI_Request requests[4];  // 2 sends + 2 receives
    int request_count = 0;

    int n_rows = ROW_SIZE(id, p, ncside);

    int above = (id - 1 + p) % p;
    int below = (id + 1) % p;

    double *send_above = (double*) malloc(ncside * 3 * sizeof(double));
    double *send_below = (double*) malloc(ncside * 3 * sizeof(double));
    double *recv_above = (double*) malloc(ncside * 3 * sizeof(double));
    double *recv_below = (double*) malloc(ncside * 3 * sizeof(double));

    for (long j = 0; j < ncside; j++) {
        send_above[j * 3] = grid[0][j].m;
        send_above[j * 3 + 1] = grid[0][j].x;
        send_above[j * 3 + 2] = grid[0][j].y;
        
        send_below[j * 3] = grid[n_rows - 1][j].m;
        send_below[j * 3 + 1] = grid[n_rows - 1][j].x;
        send_below[j * 3 + 2] = grid[n_rows - 1][j].y;
    }

    MPI_Irecv(recv_above, ncside * 3, MPI_DOUBLE, above, 0, MPI_COMM_WORLD, &requests[request_count++]);
    MPI_Irecv(recv_below, ncside * 3, MPI_DOUBLE, below, 0, MPI_COMM_WORLD, &requests[request_count++]);

    MPI_Isend(send_above, ncside * 3, MPI_DOUBLE, above, 0, MPI_COMM_WORLD, &requests[request_count++]);
    MPI_Isend(send_below, ncside * 3, MPI_DOUBLE, below, 0, MPI_COMM_WORLD, &requests[request_count++]);

    for (long y = 0; y < n_rows; y++) {
        for (long x = 0; x < ncside; x++) {
            
            cell_t *cell = &grid[y][x];
            int cell_num_particles = cell->index;
            particle_t *particles = cell->particles;

            for (long long i = 0; i < cell_num_particles; i++) {

                double px = particles[i].x;
                double py = particles[i].y;
                double pm = particles[i].m;

                double force_x = 0.0;
                double force_y = 0.0;

                // Add force from other particles in the same cell
                for (long long j = 0; j < cell_num_particles; j++) {

                    if (i == j) continue;

                    double distance_x = particles[j].x - px;
                    double distance_y = particles[j].y - py;
                    double distance = sqrt(distance_x * distance_x + distance_y * distance_y) + 1e-10; // not sure if small number is necessary

                    force_x += GRAV_FORCE(pm, particles[j].m, distance) * (distance_x / distance);
                    force_y += GRAV_FORCE(pm, particles[j].m, distance) * (distance_y / distance);
                }

                // the same as serial
                if (particles[i].id == 661) printf("force before: %.15lf\n", force_x);

                MPI_Waitall(request_count, requests, MPI_STATUSES_IGNORE);

                // Add force from neighboring cells
                for (long dx = -1; dx <= 1; dx++) {
                    for (long dy = -1; dy <= 1; dy++) {

                        if (dx == 0 && dy == 0) continue; // skip own cell

                        long nx = (x + dx + ncside) % ncside; // Wrap around edges

                        double distance_x = 0;
                        double distance_y = 0;
                        double neighbor_cell_m = 0;

                        if (y + dy >= n_rows) {
                            continue;
                            neighbor_cell_m = recv_above[nx*3];
                            distance_x = recv_above[nx*3+1] - px;
                            distance_y = recv_above[nx*3+2] - py;
                        } else if (y + dy < 0) {
                            continue;
                            neighbor_cell_m = recv_below[nx*3];
                            distance_x = recv_below[nx*3+1] - px;
                            distance_y = recv_below[nx*3+2] - py;
                            neighbor_cell_m = 0;
                        } else {
                            long ny = (y + dy + ncside) % ncside;
                            neighbor_cell_m = grid[ny][nx].m;
                            distance_x = grid[ny][nx].x - px;
                            distance_y = grid[ny][nx].y - py;
                        }

                        if (x + dx >= ncside) distance_x += side;
                        if (x + dx < 0) distance_x -= side;
                        if (y + dy >= ncside) distance_y += side;
                        if (y + dy < 0) distance_y -= side;

                        double distance = sqrt(distance_x * distance_x + distance_y * distance_y) + 1e-10; // not sure if small number is necessary;

                        force_x += GRAV_FORCE(pm, neighbor_cell_m, distance) * (distance_x / distance);
                        force_y += GRAV_FORCE(pm, neighbor_cell_m, distance) * (distance_y / distance);
                    }
                }
                // different from serial
                if (particles[i].id == 661) printf("force after: %.15lf\n", force_x);

                particles[i].gravity_x = force_x;
                particles[i].gravity_y = force_y;
            }
        }
    }

    free(send_above);
    free(send_below);
    free(recv_above);
    free(recv_below);

    // Update position and speed of particles
    for (long y = 0; y < n_rows; y++) {
        for (long x = 0; x < ncside; x++) {

            cell_t *cell = &grid[y][x];
            particle_t *particles = cell->particles;

            // Iterate over every particle from each cell
            for (long long i = 0; i < cell->index; i++) {

                if (particles[i].m == 0) continue;

                double acceleration_x = particles[i].gravity_x / particles[i].m;
                double acceleration_y = particles[i].gravity_y / particles[i].m;
    
                // Update position
                particles[i].x = POSITION(particles[i].x, particles[i].vx, acceleration_x, DELTA);
                particles[i].y = POSITION(particles[i].y, particles[i].vy, acceleration_y, DELTA);
    
                // Update speed
                particles[i].vx = SPEED(particles[i].vx, acceleration_x, DELTA);
                particles[i].vy = SPEED(particles[i].vy, acceleration_y, DELTA);
    
                // Wrap-around
                particles[i].x = fmod(particles[i].x + side, side);
                particles[i].y = fmod(particles[i].y + side, side);
            }  
        }
    }

    // TODO: send particles that moved to row belonging to different process to that process
}


/*
---------------- VERSAO QUE USA MPI_cell_t ----------------

void update_particles(long long n_part, long ncside,
                cell_t **grid, double cell_side, double side, int id, int p) {

    MPI_Request requests[4];  // 2 sends + 2 receives
    int request_count = 0;

    int n_rows = ROW_SIZE(id, p, ncside);

    int above = (id - 1 + p) % p;
    int below = (id + 1) % p;

    cell_t *recv_above = (cell_t *)malloc(ncside * sizeof(cell_t));
    cell_t *recv_below = (cell_t *)malloc(ncside * sizeof(cell_t));

    MPI_Irecv(recv_above, ncside, MPI_cell_t, above, 0, MPI_COMM_WORLD, &requests[request_count++]);
    MPI_Irecv(recv_below, ncside, MPI_cell_t, below, 0, MPI_COMM_WORLD, &requests[request_count++]);

    MPI_Isend(grid[0], ncside, MPI_cell_t, above, 0, MPI_COMM_WORLD, &requests[request_count++]);
    MPI_Isend(grid[n_rows - 1], ncside, MPI_cell_t, below, 0, MPI_COMM_WORLD, &requests[request_count++]);

    MPI_Waitall(request_count, requests, MPI_STATUSES_IGNORE);

    for (long y = 0; y < n_rows; y++) {
        for (long x = 0; x < ncside; x++) {
            if (grid[y][x].m <= 0) printf("WHAT "); // not printing anything
        }
    }

    for (long x = 0; x < ncside; x++) {
        if (recv_above[x].m <= 0) printf("WHATabove "); // printing many things
    }

    for (long y = 0; y < n_rows; y++) {
        for (long x = 0; x < ncside; x++) {
            
            cell_t *cell = &grid[y][x];
            int cell_num_particles = cell->index;
            particle_t *particles = cell->particles;

            for (long long i = 0; i < cell_num_particles; i++) {

                double px = particles[i].x;
                double py = particles[i].y;
                double pm = particles[i].m;

                double force_x = 0.0;
                double force_y = 0.0;

                // Add force from other particles in the same cell
                for (long long j = 0; j < cell_num_particles; j++) {

                    if (i == j) continue;

                    double distance_x = particles[j].x - px;
                    double distance_y = particles[j].y - py;
                    double distance = sqrt(distance_x * distance_x + distance_y * distance_y) + 1e-10; // not sure if small number is necessary

                    force_x += GRAV_FORCE(pm, particles[j].m, distance) * (distance_x / distance);
                    force_y += GRAV_FORCE(pm, particles[j].m, distance) * (distance_y / distance);
                }

                // Add force from neighboring cells
                for (long dx = -1; dx <= 1; dx++) {
                    for (long dy = -1; dy <= 1; dy++) {

                        if (dx == 0 && dy == 0) continue; // skip own cell

                        long nx = (x + dx + ncside) % ncside; // Wrap around edges

                        cell_t *neighbor_cell;

                        if (y + dy >= n_rows) {
                            if (particles[i].id == 661) printf("adding from row below\n");
                            neighbor_cell = &recv_below[nx];
                        } else if (y + dy < 0) {
                            if (particles[i].id == 661) printf("adding from row above\n");
                            neighbor_cell = &recv_above[nx];
                        } else {
                            long ny = (y + dy + ncside) % ncside;
                            neighbor_cell = &grid[ny][nx];
                        }

                        if (particles[i].id == 661)printf("neighbor_cell.x = %lf, neighbor_cell.y = %lf, neighbor_cell.m = %lf\n", neighbor_cell->x, neighbor_cell->y, neighbor_cell->m);

                        // Compute force from center of mass of the cell
                        double distance_x = neighbor_cell->x - px;
                        double distance_y = neighbor_cell->y - py;

                        if (x + dx >= ncside) distance_x += side;
                        if (x + dx < 0) distance_x -= side;
                        if (y + dy >= ncside) distance_y += side;
                        if (y + dy < 0) distance_y -= side;

                        double distance = sqrt(distance_x * distance_x + distance_y * distance_y) + 1e-10; // not sure if small number is necessary;

                        force_x += GRAV_FORCE(pm, neighbor_cell->m, distance) * (distance_x / distance);
                        force_y += GRAV_FORCE(pm, neighbor_cell->m, distance) * (distance_y / distance);
                    }
                }

                particles[i].gravity_x = force_x;
                particles[i].gravity_y = force_y;
            }
        }
    }

    free(recv_above);
    free(recv_below);

    // Update position and speed of particles
    for (long y = 0; y < n_rows; y++) {
        for (long x = 0; x < ncside; x++) {

            cell_t *cell = &grid[y][x];
            particle_t *particles = cell->particles;

            // Iterate over every particle from each cell
            for (long long i = 0; i < cell->index; i++) {

                if (particles[i].m == 0) continue;

                double acceleration_x = particles[i].gravity_x / particles[i].m;
                double acceleration_y = particles[i].gravity_y / particles[i].m;
    
                // Update position
                particles[i].x = POSITION(particles[i].x, particles[i].vx, acceleration_x, DELTA);
                particles[i].y = POSITION(particles[i].y, particles[i].vy, acceleration_y, DELTA);
    
                // Update speed
                particles[i].vx = SPEED(particles[i].vx, acceleration_x, DELTA);
                particles[i].vy = SPEED(particles[i].vy, acceleration_y, DELTA);
    
                // Wrap-around
                particles[i].x = fmod(particles[i].x + side, side);
                particles[i].y = fmod(particles[i].y + side, side);
            }  
        }
    }

    // TODO: send particles that moved to row belonging to different process to that process
}
*/


int main(int argc, char **argv)
{
    double exec_time;
    long total_num_collisions = 0;
    long seed;
    double side;
    long ncside;
    long long n_part;
    long long local_n_part;
    long time_steps;
    
    double cell_side;

    MPI_Status status;
    int id, p;

    // Parse arguments from command line
    parse_args(argc, argv, &seed, &side, &ncside, &n_part, &time_steps);

    MPI_Init (&argc, &argv);

    create_mpi_particle();
    create_mpi_cell();

    MPI_Comm_rank (MPI_COMM_WORLD, &id);
    MPI_Comm_size (MPI_COMM_WORLD, &p);
    
    // Calculate the side size of each cell
    cell_side = (double) side / ncside;

    int process_assigned_rows = ROW_SIZE(id, p, ncside);

    cell_t **grid = (cell_t**) malloc(process_assigned_rows * sizeof(cell_t*));
    for (long i = 0; i < process_assigned_rows; i++) {
        grid[i] = (cell_t*) malloc(ncside * sizeof(cell_t));
    }

    // not sure if necessary, probably yes
    init_grid(ncside, grid, n_part / (ncside * ncside), process_assigned_rows);

    exec_time = -omp_get_wtime();

    {
        init_process_particles(seed, side, ncside, n_part, grid, id, p);

        calculate_center_of_mass(ncside, grid, id, p);
            
        for (long i = 0; i < time_steps; i++) {
            //printf("t=%ld\n", i);

            // TODO: update_particles MPI not implemented
            update_particles(n_part, ncside, grid, cell_side, side, id, p);
            if (!id) printf("%.15lf %.15lf %lld\n", grid[0][0].particles[0].x, grid[0][0].particles[0].y, grid[0][0].particles[0].id);
            MPI_Barrier (MPI_COMM_WORLD);
            MPI_Finalize();
            exit(0);
            //calculate_center_of_mass
            // TODO: detect_collisions MPI not implemented
            //total_num_collisions += detect_collisions(particles, &n_part, ncside, grid);
        }
    }

    // TODO: this won't work if particle 0 moves rows and stops being at index 0
    //if (particles[0].id == 0) printf("%.3lf %.3lf\n%ld\n", particles[0].x, particles[0].y, total_num_collisions);

    // this is very convoluted, there's definitely a better way
    for (long i = 0; i < ROW_SIZE(id,p,ncside); i++) {
        for (long j = 0; j < ncside; j++) {
            for (long idx = 0; idx < grid[i][j].index; idx++) {
                if (grid[i][j].particles[idx].id == 0) {
                    printf("%.3lf %.3lf\n%ld\n", grid[i][j].particles[idx].x, grid[i][j].particles[idx].y, total_num_collisions);
                }
            }
        }
    }

    for (long i = 0; i < process_assigned_rows; i++) {
        for (long j = 0; j < ncside; j++) {
            free(grid[i][j].particles);
        }
        free(grid[i]);
    }
    free(grid);

    
    exec_time += omp_get_wtime();
    if (!id) fprintf(stderr, "%.1fs\n", exec_time);
    
    MPI_Barrier (MPI_COMM_WORLD);
    MPI_Finalize();
   
    return 0;
}
