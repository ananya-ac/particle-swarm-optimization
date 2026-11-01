# Particle Swarm Optimization

C++ implementations of Particle Swarm Optimization (PSO): a sequential version, a synchronous MPI version, and two asynchronous MPI versions. Each one can optimize standard benchmark functions or train a small neural network on a dataset.

## Requirements

- A C++20 compiler (e.g. `g++` 10+)
- CMake 3.8+
- An MPI implementation (OpenMPI or MPICH)
- SDL2 development libraries (CMake requires them)

On Ubuntu/Debian:

```bash
sudo apt install build-essential cmake libopenmpi-dev openmpi-bin libsdl2-dev
```

## Building

From the project root:

```bash
mkdir -p build && cd build
cmake ..
make -j
```

The executables end up in `build/bin/`:

| Executable | Description |
|---|---|
| `particle_swarm_optimization` | Sequential PSO |
| `particle_swarm_optimization_mpi` | Synchronous PSO, with the population split across MPI processes |
| `asynch_pso` | Asynchronous (master/worker) MPI PSO |
| `asynch_pso2` | Another asynchronous MPI PSO implementation |

## Running

All executables take the same arguments.

**Benchmark objective functions:**

```
<executable> <population_size> <w> <c1> <c2> <max_epochs> <objective_function> <number_parameters>
```

**Neural network objective function:**

```
<executable> <population_size> <w> <c1> <c2> <max_epochs> neural_network <dataset_filename> <number_layers>
```

| Argument | Meaning |
|---|---|
| `population_size` | Number of particles |
| `w` | Inertia weight |
| `c1` | Cognitive (personal best) coefficient |
| `c2` | Social (global best) coefficient |
| `max_epochs` | Number of iterations. For `asynch_pso` and `asynch_pso2` this is the total number of particle moves |
| `objective_function` | `sphere`, `ackley`, `griewank`, `rastrigin`, `rosenbrock` or `neural_network` |
| `number_parameters` | Number of dimensions of the benchmark function |
| `dataset_filename` | CSV file for the neural network (see [Datasets](#datasets)) |
| `number_layers` | Number of hidden layers in the neural network |

### Examples

Run these from the `build/` directory.

Sequential:

```bash
./bin/particle_swarm_optimization 20 0.7 1.5 1.5 100 sphere 5
./bin/particle_swarm_optimization 20 0.7 1.5 1.5 100 neural_network ../data/iris.csv 1
```

Synchronous MPI (4 processes):

```bash
mpirun -np 4 ./bin/particle_swarm_optimization_mpi 20 0.7 1.5 1.5 100 rastrigin 5
```

Asynchronous MPI (needs at least 2 processes: 1 master and at least 1 worker):

```bash
mpirun -np 4 ./bin/asynch_pso 20 0.7 1.5 1.5 500 ackley 5
mpirun -np 4 ./bin/asynch_pso2 20 0.7 1.5 1.5 500 neural_network ../data/wdbc.csv 1
```

If you have more MPI processes than physical cores, add `--oversubscribe` (OpenMPI).

## Datasets

`data/` contains two classification datasets in CSV format. The first column is the class label and the remaining columns are features:

- `iris.csv`: Iris flower dataset (3 classes, 4 features)
- `wdbc.csv`: Wisconsin Diagnostic Breast Cancer dataset (2 classes, 30 features)

The `.names` files describe each dataset.

## Project structure

```
.
├── CMakeLists.txt
├── data/                  # Datasets for the neural network objective
└── src/
    ├── particle_swarm_optimization.cxx       # Sequential PSO
    ├── particle_swarm_optimization_mpi.cxx   # Synchronous MPI PSO
    ├── asynch_pso.cxx                        # Asynchronous MPI PSO
    ├── asynch_pso2.cxx                       # Asynchronous MPI PSO (alternate)
    ├── dataset.cxx / dataset.hxx             # CSV dataset loader
    └── objectives/                           # Objective functions
        ├── objective_function.*              # Base class
        ├── sphere.*, ackley.*, griewank.*, rastrigin.*, rosenbrock.*
        └── neural_network.*
```
