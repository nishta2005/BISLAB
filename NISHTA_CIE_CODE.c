#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>   // For parallel processing

#define POP_ROWS 10
#define POP_COLS 10
#define NUM_FEATURES 50
#define MAX_GEN 100
#define MUTATION_RATE 0.05

typedef struct {
    int features[NUM_FEATURES];  // binary mask of selected features
    double fitness;
} Individual;

// Random number in [0,1)
double rand01() {
    return rand() / (double)RAND_MAX;
}

// Initialize individual randomly
void init_individual(Individual *ind) {
    for (int i = 0; i < NUM_FEATURES; i++)
        ind->features[i] = (rand01() < 0.5) ? 1 : 0;
}

// Dummy fitness function (replace with model evaluation)
double evaluate_fitness(Individual *ind) {
    int count = 0;
    for (int i = 0; i < NUM_FEATURES; i++)
        if (ind->features[i]) count++;
    // Example: reward moderate feature counts
    return 1.0 / (1.0 + abs(count - 10));
}

// Mutation
void mutate(Individual *ind) {
    for (int i = 0; i < NUM_FEATURES; i++)
        if (rand01() < MUTATION_RATE)
            ind->features[i] = 1 - ind->features[i];
}

// Crossover
void crossover(Individual *p1, Individual *p2, Individual *child) {
    int point = rand() % NUM_FEATURES;
    for (int i = 0; i < NUM_FEATURES; i++)
        child->features[i] = (i < point) ? p1->features[i] : p2->features[i];
}

// Select neighbor with highest fitness
int select_best_neighbor(Individual grid[POP_ROWS][POP_COLS], int r, int c) {
    int best_r = r, best_c = c;
    double best_fit = grid[r][c].fitness;
    int dr[4] = {-1, 1, 0, 0};
    int dc[4] = {0, 0, -1, 1};

    for (int i = 0; i < 4; i++) {
        int nr = (r + dr[i] + POP_ROWS) % POP_ROWS;
        int nc = (c + dc[i] + POP_COLS) % POP_COLS;
        if (grid[nr][nc].fitness > best_fit) {
            best_fit = grid[nr][nc].fitness;
            best_r = nr;
            best_c = nc;
        }
    }
    return best_r * POP_COLS + best_c;
}

int main() {
    srand(time(NULL));
    Individual grid[POP_ROWS][POP_COLS], new_grid[POP_ROWS][POP_COLS];

    // Initialization
    for (int r = 0; r < POP_ROWS; r++)
        for (int c = 0; c < POP_COLS; c++) {
            init_individual(&grid[r][c]);
            grid[r][c].fitness = evaluate_fitness(&grid[r][c]);
        }

    // Evolution loop
    for (int gen = 0; gen < MAX_GEN; gen++) {
        #pragma omp parallel for collapse(2)
        for (int r = 0; r < POP_ROWS; r++) {
            for (int c = 0; c < POP_COLS; c++) {
                int best_idx = select_best_neighbor(grid, r, c);
                int br = best_idx / POP_COLS, bc = best_idx % POP_COLS;

                Individual child;
                crossover(&grid[r][c], &grid[br][bc], &child);
                mutate(&child);
                child.fitness = evaluate_fitness(&child);
                new_grid[r][c] = child;
            }
        }

        // Update grid
        for (int r = 0; r < POP_ROWS; r++)
            for (int c = 0; c < POP_COLS; c++)
                grid[r][c] = new_grid[r][c];
    }

    // Find global best
    double best_fitness = 0.0;
    Individual best;
    for (int r = 0; r < POP_ROWS; r++)
        for (int c = 0; c < POP_COLS; c++)
            if (grid[r][c].fitness > best_fitness) {
                best_fitness = grid[r][c].fitness;
                best = grid[r][c];
            }

    printf("Best fitness: %.4f\nSelected features: ", best_fitness);
    for (int i = 0; i < NUM_FEATURES; i++)
        if (best.features[i]) printf("%d ", i);
    printf("\n");

    return 0;
}
