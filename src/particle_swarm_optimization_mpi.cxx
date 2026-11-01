#include <limits>
using std::numeric_limits;

#include <iomanip>
using std::fixed;
using std::right;
using std::setprecision;
using std::setw;

#include <iostream>
using std::cerr;
using std::cout;
using std::endl;

#include <random>
using std::mt19937;
using std::random_device;
using std::uniform_real_distribution;

#include <string>
using std::stod;
using std::stoi;
using std::string;

#include <vector>
using std::vector;

#include "mpi.h"
#include "objectives/ackley.hxx"
#include "objectives/griewank.hxx"
#include "objectives/neural_network.hxx"
#include "objectives/objective_function.hxx"
#include "objectives/rastrigin.hxx"
#include "objectives/rosenbrock.hxx"
#include "objectives/sphere.hxx"

void print_particle(const vector<double> &particle,
                    const vector<double> &velocity, double fitness) {
  cout << "fitness: " << right << fixed << setw(12) << setprecision(7)
       << fitness << " particle[";
  for (int32_t j = 0; j < particle.size(); j++) {
    cout << right << fixed << setw(12) << setprecision(7) << particle[j];
  }

  if (velocity.size() > 0) {
    cout << " velocity [";
    for (int32_t j = 0; j < velocity.size(); j++) {
      cout << right << fixed << setw(12) << setprecision(7) << velocity[j];
    }
    cout << "]";
  }
  cout << endl;
}

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);

  int32_t number_processes;
  MPI_Comm_size(MPI_COMM_WORLD, &number_processes);

  int32_t process_id;
  MPI_Comm_rank(MPI_COMM_WORLD, &process_id);

  vector<string> arguments = vector<string>(argv, argv + argc);

  // Check to see command line arguments were correctly specified
  if (arguments.size() != 8 && arguments.size() != 9) {

    MPI_Finalize();
    exit(1);
  }

  int32_t population_size = stoi(arguments[1]);
  double w = stod(arguments[2]);
  double c1 = stod(arguments[3]);
  double c2 = stod(arguments[4]);
  int32_t max_epochs = stoi(arguments[5]);
  string objective_function_name(arguments[6]);

  int32_t n_parameters = 0;
  ObjectiveFunction *of = NULL;
  if (process_id == 0)
    cout << "creating objective function: '" << objective_function_name << "'"
         << endl;
  if (objective_function_name == "neural_network") {
    string dataset_filename = arguments[7];

    int32_t n_layers = stoi(arguments[8]);

    of = new NeuralNetwork(-20.0, 20.0, dataset_filename, n_layers);
    n_parameters = of->get_n_parameters();

  } else {
    n_parameters = stoi(arguments[7]);

    if (objective_function_name == "sphere") {
      of = new Sphere(n_parameters, -5.12, 5.12);
    } else if (objective_function_name == "ackley") {
      of = new Ackley(n_parameters, -30, 30);
    } else if (objective_function_name == "griewank") {
      of = new Griewank(n_parameters, -600, 600);
    } else if (objective_function_name == "rastrigin") {
      of = new Rastrigin(n_parameters, -2.048, 2.048);
    } else if (objective_function_name == "rosenbrock") {
      of = new Rosenbrock(n_parameters, -5.12, 5.12);
    } else {
      MPI_Finalize();
      exit(1);
    }
  }
  vector<double> min_bounds = of->get_min_bounds();
  vector<double> max_bounds = of->get_max_bounds();

  if (process_id == 0) {
    for (int32_t i = 0; i < n_parameters; i++) {
      cout << "parameter [" << i << "] min bound: " << min_bounds[i]
           << ", max bound: " << max_bounds[i] << endl;
    }
  }

  // set up the random number generator
  random_device rd;
  mt19937 random_number_generator(rd());
  uniform_real_distribution<double> distribution(0.0, 1.0);

  // TODO:
  // The population of particles should be evenly divided over the number of
  // MPI processes.
  int process_population = population_size / number_processes;

  if (process_id == number_processes - 1) {
    process_population += population_size % number_processes;
  }

  // initialize the particles for particle swarm optimization
  //  vector<vector<double>> particles(population_size,
  //  vector<double>(n_parameters)); vector<vector<double>>
  //  velocities(population_size, vector<double>(n_parameters));
  //  vector<vector<double>> local_bests(population_size,
  //  vector<double>(n_parameters));

  vector<vector<double>> particles(process_population,
                                   vector<double>(n_parameters));
  vector<vector<double>> velocities(process_population,
                                    vector<double>(n_parameters));
  vector<vector<double>> local_bests(process_population,
                                     vector<double>(n_parameters));

  // initialize each fitness to the maximum possible double value
  // so the first time we compute a particle fitness it will become
  // the local best fitness
  //   vector<double> local_best_fitnesses(population_size,
  //                                       numeric_limits<double>::max());

  vector<double> local_best_fitnesses(process_population,
                                      numeric_limits<double>::max());

  vector<double> global_best(n_parameters);
  vector<double> proc_global_best(n_parameters);                   // mpi
  double proc_global_best_fitness = numeric_limits<double>::max(); // mpi
  int32_t global_best_process = -1;                                // mpi
  double global_best_fitness = numeric_limits<double>::max();

  for (int32_t i = 0; i < process_population; i++) {

    // randomly initialize the particle values and local bests
    for (int32_t j = 0; j < n_parameters; j++) {
      double r = distribution(random_number_generator);
      particles[i][j] = ((max_bounds[j] - min_bounds[j]) * r) + min_bounds[j];
      local_bests[i][j] = particles[i][j];

      // give each particle a small random initial velocity
      velocities[i][j] =
          0.05 * (((max_bounds[j] - min_bounds[j]) * r) + min_bounds[j]);
    }
    // print_particle(particles[i], velocities[i], local_best_fitnesses[i]);
  }

  if (process_id == 0) {
    cout << "beginning particle swarm optimization with paramerers:" << endl;
    cout << "\tpopulation_size: " << population_size << endl;
    cout << "\tn_parameters: " << n_parameters << endl;
    cout << "\tw: " << w << endl;
    cout << "\tc1: " << c1 << endl;
    cout << "\tc2: " << c2 << endl;
  }
  for (int32_t epoch = 0; epoch < max_epochs; epoch++) {
    // TODO:
    // modify the following so that each process calculates its own best
    // particle so we can later send these to the other processes to calculate
    // the overall global best

    // compute the fitness for each particle and update the
    // particle local bests and global bests

    vector<double> process_best_fitnesses(number_processes);
    vector<double> process_best_particles(number_processes * n_parameters);

    for (int i = 0; i < process_population; i++) {
      double fitness = of->evaluate(particles[i]);
      if (fitness < local_best_fitnesses[i]) {
        for (int32_t j = 0; j < n_parameters; j++) {
          local_bests[i][j] = particles[i][j];
        }
        local_best_fitnesses[i] = fitness;
        // cout << "new  LOCAL BEST " << setw(3) << i << " ";
        // print_particle(particles[i], velocities[i], local_best_fitnesses[i]);
      }

      if (fitness < proc_global_best_fitness) {
        for (int32_t j = 0; j < n_parameters; j++) {
          proc_global_best[j] = particles[i][j];
        }

        proc_global_best_fitness = fitness;
      }
    }

    // TODO:
    // After each process computes the fitness for its slice of the particles,
    // use MPI_Allgather to send each slice's best fitness values and best
    // particle values to all other processes. Use these combined values to
    // determine the actual global best particle fitness and its parameters to
    // use when generating the next set of particle positions

    MPI_Allgather(&proc_global_best_fitness, 1, MPI_DOUBLE,
                  process_best_fitnesses.data(), 1, MPI_DOUBLE, MPI_COMM_WORLD);

    MPI_Allgather(proc_global_best.data(), n_parameters, MPI_DOUBLE,
                  process_best_particles.data(), n_parameters, MPI_DOUBLE,
                  MPI_COMM_WORLD);

    for (int i = 0; i < number_processes; i++) {
      if (process_best_fitnesses[i] < global_best_fitness) {
        global_best_fitness = process_best_fitnesses[i];

        for (int j = 0; j < n_parameters; j++) {
          global_best[j] = process_best_particles[i * n_parameters + j];
        }
        global_best_process = i;
      }
    }

    // move each particle according to the PSO algorithm
    for (int32_t i = 0; i < process_population; i++) {
      // get the random numbers for moving the particle
      double r1 = distribution(random_number_generator);
      double r2 = distribution(random_number_generator);

      // calculate the particle's new velocity and move it
      for (int32_t j = 0; j < n_parameters; j++) {
        velocities[i][j] = (w * velocities[i][j]) +
                           (r1 * c1 * (local_bests[i][j] - particles[i][j])) +
                           (r2 * c2 * (global_best[j] - particles[i][j]));

        double new_position = particles[i][j] + velocities[i][j];

        // if the particle moves out of bounds, move it back into bounds and
        // adjust the velocity accordingly
        if (new_position < min_bounds[j]) {
          // cout << "particle " << particles[i][j] << " moved to " <<
          // new_position << " out of bounds, velocity was " <<
          // velocities[i][j];
          new_position = min_bounds[j] * 0.9;
          velocities[i][j] = new_position - particles[i][j];
          // velocities[i][j] = new_position - particles[i][j];
          //  cout << " new position " << new_position << " and velocity " <<
          //  velocities[i][j] << endl;
        }

        if (new_position > max_bounds[j]) {
          // cout << "particle " << particles[i][j] << " moved to " <<
          // new_position << " out of bounds, velocity was " <<
          // velocities[i][j];
          new_position = max_bounds[j] * 0.9;
          velocities[i][j] = new_position - particles[i][j];
          // cout << " new position " << new_position << " and velocity " <<
          // velocities[i][j] << endl;
        }

        particles[i][j] = new_position;
      }
      // cout << "particle " << setw(3) << i << " ";
      // print_particle(particles[i], velocities[i], local_best_fitnesses[i]);
    }
    if (process_id == 0) {
      cout << "epoch " << epoch << " global best " << global_best_fitness
           << " at process " << global_best_process << " ";
      print_particle(global_best, vector<double>(), global_best_fitness);
    }
  }
  MPI_Finalize();
  return 0;
}
