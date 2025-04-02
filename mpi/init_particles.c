#define _USE_MATH_DEFINES
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "init_particles.h"
#include "cell.h"
#define G 6.67408e-11
#define EPSILON2 (0.005 * 0.005)
#define DELTAT 0.1

unsigned int seed;
void init_r4uni(int input_seed)
{
    seed = input_seed + 987654321;
}
double rnd_uniform01()
{
    int seed_in = seed;
    seed ^= (seed << 13);
    seed ^= (seed >> 17);
    seed ^= (seed << 5);
    return 0.5 + 0.2328306e-09 * (seed_in + (int)seed);
}
double rnd_normal01()
{
    double u1, u2, z, result;
    do
    {
        u1 = rnd_uniform01();
        u2 = rnd_uniform01();
        z = sqrt(-2 * log(u1)) * cos(2 * M_PI * u2);
        result = 0.5 + 0.15 * z; // Shift mean to 0.5 and scale
    } while (result < 0 || result >= 1);
    return result;
}

void init_particles(long seed, double side, long ncside, long long n_part, particle_t *par)
{
    double (*rnd01)() = rnd_uniform01;
    long long i;

    if (seed < 0)
    {
        rnd01 = rnd_normal01;
        seed = -seed;
    }

    init_r4uni(seed);

    for (i = 0; i < n_part; i++)
    {
        par[i].id = i;
        par[i].x = rnd01() * side;
        par[i].y = rnd01() * side;
        par[i].vx = (rnd01() - 0.5) * side / ncside / 5.0;
        par[i].vy = (rnd01() - 0.5) * side / ncside / 5.0;

        par[i].m = rnd01() * 0.01 * (ncside * ncside) / n_part / G * EPSILON2;
    }
}

void init_process_particles(long seed, double side, long ncside, long long n_part, cell_t **grid, int id, int p, char *main_particle_bool)
{
    double (*rnd01)() = rnd_uniform01;
    long long i;

    double cell_side = (double)side / ncside;

    if (seed < 0)
    {
        rnd01 = rnd_normal01;
        seed = -seed;
    }

    init_r4uni(seed);

    // calculate process assigned lines

    for (i = 0; i < n_part; i++)
    {

        double x = rnd01() * side;
        double y = rnd01() * side;

        int row = y / cell_side;
        int col = x / cell_side;

        if (col >= ncside)
            col = ncside - 1;
        if (row >= ncside)
            row = ncside - 1;

        particle_t particle;

        particle.id = 0;

        particle.x = x;
        particle.y = y;
        particle.vx = (rnd01() - 0.5) * side / ncside / 5.0;
        particle.vy = (rnd01() - 0.5) * side / ncside / 5.0;
        particle.m = rnd01() * 0.01 * (ncside * ncside) / n_part / G * EPSILON2;

        if (row >= ROW_LOW(id, p, ncside) && row < ROW_HIGH(id, p, ncside))
        {
            if (i == 0)
            {
                particle.id = 1;
                *main_particle_bool = 1;
            }

            row -= ROW_LOW(id, p, ncside); // get local index

            add_particle_to_cell(&grid[row][col], particle);
        }
    }
}
