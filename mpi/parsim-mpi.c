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

    displacements[0] = offsetof(center_of_mass, x);
    displacements[1] = offsetof(center_of_mass, y);
    displacements[2] = offsetof(center_of_mass, m);

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


void add_particel_to_buffer(particle_t *particles, int index, int size, particle_t particle) {
    if (index >= size) {
        particles = (particle_t*) realloc(particles, 2 * size * sizeof(particle_t));
        size *= 2;
    }

    particles[index] = particle;
    index++;
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



/*
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
*/


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
    int process_low = ROW_LOW(id, p, ncside);
    int process_high = ROW_HIGH(id, p, ncside);

    int above = (id - 1 + p) % p;
    int below = (id + 1) % p;

    // centers of mass of highest and lowest rows, to send to neighboring processes
    center_of_mass *send_above = (center_of_mass *)malloc(ncside * sizeof(center_of_mass));
    center_of_mass *send_below = (center_of_mass *)malloc(ncside * sizeof(center_of_mass));
    center_of_mass *recv_above = (center_of_mass *)malloc(ncside * sizeof(center_of_mass));
    center_of_mass *recv_below = (center_of_mass *)malloc(ncside * sizeof(center_of_mass));

    for (long x = 0; x < ncside; x++) {
        send_above[x].m = grid[0][x].m;
        send_above[x].x = grid[0][x].x;
        send_above[x].y = grid[0][x].y;
        send_below[x].m = grid[n_rows - 1][x].m;
        send_below[x].x = grid[n_rows - 1][x].x;
        send_below[x].y = grid[n_rows - 1][x].y;
    }

    MPI_Irecv(recv_above, ncside, MPI_cell_t, above, 0, MPI_COMM_WORLD, &requests[request_count++]);
    MPI_Irecv(recv_below, ncside, MPI_cell_t, below, 0, MPI_COMM_WORLD, &requests[request_count++]);

    MPI_Isend(send_above, ncside, MPI_cell_t, above, 0, MPI_COMM_WORLD, &requests[request_count++]);
    MPI_Isend(send_below, ncside, MPI_cell_t, below, 0, MPI_COMM_WORLD, &requests[request_count++]);

    free(send_above);
    free(send_below);

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

                MPI_Waitall(request_count, requests, MPI_STATUSES_IGNORE);

                // Add force from neighboring cells
                for (long dx = -1; dx <= 1; dx++) {
                    for (long dy = -1; dy <= 1; dy++) {

                        if (dx == 0 && dy == 0) continue; // skip own cell

                        long nx = (x + dx + ncside) % ncside; // Wrap around edges

                        double neighbor_cell_x;
                        double neighbor_cell_y;
                        double neighbor_cell_m;

                        if (y + dy >= n_rows) {
                            neighbor_cell_x = recv_below[nx].x;
                            neighbor_cell_y = recv_below[nx].y;
                            neighbor_cell_m = recv_below[nx].m;
                        } else if (y + dy < 0) {
                            neighbor_cell_x = recv_above[nx].x;
                            neighbor_cell_y = recv_above[nx].y;
                            neighbor_cell_m = recv_above[nx].m;
                        } else {
                            long ny = (y + dy + ncside) % ncside;
                            neighbor_cell_x = grid[ny][nx].x;
                            neighbor_cell_y = grid[ny][nx].y;
                            neighbor_cell_m = grid[ny][nx].m;
                        }

                        // Compute force from center of mass of the cell
                        double distance_x = neighbor_cell_x - px;
                        double distance_y = neighbor_cell_y - py;

                        if (x + dx >= ncside) distance_x += side;
                        if (x + dx < 0) distance_x -= side;
                        if (y + dy >= ncside) distance_y += side;
                        if (y + dy < 0) distance_y -= side;

                        double distance = sqrt(distance_x * distance_x + distance_y * distance_y) + 1e-10; // not sure if small number is necessary;

                        force_x += GRAV_FORCE(pm, neighbor_cell_m, distance) * (distance_x / distance);
                        force_y += GRAV_FORCE(pm, neighbor_cell_x, distance) * (distance_y / distance);
                    }
                }

                particles[i].gravity_x = force_x;
                particles[i].gravity_y = force_y;
            }
        }
    }

    free(recv_above);
    free(recv_below);

    int initial_size = 5;
    particle_t *above_particles_to_send = (particle_t*) malloc(initial_size * sizeof(particle_t));
    particle_t *below_particles_to_send = (particle_t*) malloc(initial_size * sizeof(particle_t));
    particle_t *above_particles_to_recv;
    particle_t *below_particles_to_recv;

    int above_particles_count = 0;
    int below_particles_count = 0;
    int recv_above_particles_count;
    int recv_below_particles_count;


    MPI_Request particle_requests[4];
    int particle_request_count = 0;

    // Update position and speed of particles
    for (long y = 0; y < n_rows; y++) {
        for (long x = 0; x < ncside; x++) {

            cell_t *cell = &grid[y][x];
            particle_t *particles = cell->particles;

            // Iterate over every particle from each cell
            for (long long i = 0; i < cell->index; i++) {

                if (particles[i].m == 0) continue;

                particle_t particle = particles[i];

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

                int new_row = particles[i].y / cell_side;
                int new_col = particles[i].x / cell_side;

                if (new_row >= ncside) new_row = ncside - 1;
                if (new_col >= ncside) new_col = ncside - 1;

                if (new_row >= process_low && new_row < process_high) {
                    remove_particle_from_cell(cell, i);
                    add_particle_to_cell(&grid[new_row-process_low][new_col], particles[i]);
                } else if (new_row < process_low) {
                    remove_particle_from_cell(cell, i);
                    add_particel_to_buffer(above_particles_to_send, above_particles_count, initial_size, particles[i]);
                    above_particles_count++;
                } else if (new_row >= process_high) {
                    remove_particle_from_cell(cell, i);
                    add_particel_to_buffer(below_particles_to_send, below_particles_count, initial_size, particles[i]);
                    below_particles_count++;
                }
            }  
        }
    }

    MPI_Sendrecv(&above_particles_count, 1, MPI_INT, above, 0, &recv_above_particles_count, 1, MPI_INT, above, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Sendrecv(&below_particles_count, 1, MPI_INT, below, 0, &recv_below_particles_count, 1, MPI_INT, below, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    above_particles_to_recv = (particle_t*) malloc(recv_above_particles_count * sizeof(particle_t));
    below_particles_to_recv = (particle_t*) malloc(recv_below_particles_count * sizeof(particle_t));

    // Send particles to neighboring processes
    MPI_Isend(above_particles_to_send, above_particles_count, MPI_particle_t, above, 0, MPI_COMM_WORLD, &particle_requests[particle_request_count++]);
    MPI_Isend(below_particles_to_send, below_particles_count, MPI_particle_t, below, 0, MPI_COMM_WORLD, &particle_requests[particle_request_count++]);
    MPI_Irecv(above_particles_to_recv, recv_above_particles_count, MPI_particle_t, above, 0, MPI_COMM_WORLD, &particle_requests[particle_request_count++]);
    MPI_Irecv(below_particles_to_recv, recv_below_particles_count, MPI_particle_t, below, 0, MPI_COMM_WORLD, &particle_requests[particle_request_count++]);

    MPI_Waitall(particle_request_count, particle_requests, MPI_STATUSES_IGNORE);

    /*
    for (long i = 0; i < recv_above_particles_count; i++) {
        double x = above_particles_to_recv[i].x;
        double y = above_particles_to_recv[i].y;

        int row = y / cell_side;
        int col = x / cell_side;

        if (col >= ncside) col = ncside - 1;
        if (row >= ncside) row = ncside - 1;

        add_particle_to_cell(&grid[0][col], above_particles_to_recv[i]);
    }
    */

    /*
    for (long i = 0; i < recv_below_particles_count; i++) {
        double x = below_particles_to_recv[i].x;
        double y = below_particles_to_recv[i].y;

        int row = y / cell_side;
        int col = x / cell_side;

        if (col >= ncside) col = ncside - 1;
        if (row >= ncside) row = ncside - 1;

        add_particle_to_cell(&grid[n_rows-1][col], below_particles_to_recv[i]);
    }
    */



}


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
