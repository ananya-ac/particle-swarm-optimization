#include <limits>
#include <mpi.h>
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

#include "objectives/ackley.hxx"
#include "objectives/griewank.hxx"
#include "objectives/neural_network.hxx"
#include "objectives/objective_function.hxx"
#include "objectives/rastrigin.hxx"
#include "objectives/rosenbrock.hxx"
#include "objectives/sphere.hxx"

const int TAG_PARTICLE = 1;
const int TAG_FITNESS = 2;
const int TAG_TERMINATE = 3;

void print_particle(const vector<double> &particle,
                    const vector<double> &velocity, double fitness) {
  cout << "fitness: " << right << fixed << setw(12) << setprecision(7)
       << fitness << ", particle[";
  for (int32_t j = 0; j < particle.size(); j++) {
    cout << right << fixed << setw(12) << setprecision(7) << particle[j];
  }
  cout << "], ";
  if (velocity.size() > 0) {
    cout << " velocity [";
    for (int32_t j = 0; j < velocity.size(); j++) {
      cout << right << fixed << setw(12) << setprecision(7) << velocity[j];
    }
    cout << "]";
  }
  cout << endl;
}

void masterProcess(double w, double c1, double c2, int population_size,
                   int max_particles, int n_parameters,
                   vector<double> min_bounds, vector<double> max_bounds) {

  // set up the random number generator
  random_device rd;
  mt19937 random_number_generator(rd());
  uniform_real_distribution<double> distribution(0.0, 1.0);

  // initialize the particles for particle swarm optimization
  vector<vector<double>> particles(population_size,
                                   vector<double>(n_parameters));
  vector<vector<double>> velocities(population_size,
                                    vector<double>(n_parameters));
  vector<vector<double>> local_bests(population_size,
                                     vector<double>(n_parameters));

  // initialize each fitness to the maximum possible double value
  // so the first time we compute a particle fitness it will become
  // the local best fitness
  vector<double> local_best_fitnesses(population_size,
                                      numeric_limits<double>::max());

  int32_t global_best_index = -1;
  vector<double> global_best(n_parameters);
  double global_best_fitness = numeric_limits<double>::max();

  for (int32_t i = 0; i < population_size; i++) {

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
  cout << "beginning particle swarm optimization with paramerers:" << endl;
  cout << "\tpopulation_size: " << population_size << endl;
  cout << "\tn_parameters: " << n_parameters << endl;
  cout << "\tw: " << w << endl;
  cout << "\tc1: " << c1 << endl;
  cout << "\tc2: " << c2 << endl;

  int world_size;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  int32_t particles_sent = 0;

  while (!(particles_sent > max_particles)) {
    int val;

    MPI_Request req;
    MPI_Status status_;

    MPI_Irecv(&val, 1, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD,
              &req);
    MPI_Wait(&req, &status_);
    int tag = status_.MPI_TAG;
    int source = status_.MPI_SOURCE;

    if (tag == TAG_PARTICLE) {
      int send_particle_index = particles_sent % population_size;

      int min_send = numeric_limits<int>::max();

      MPI_Request p_req[3];

      MPI_Isend(&send_particle_index, 1, MPI_INT, source, tag, MPI_COMM_WORLD,
                &p_req[0]);
      MPI_Isend(particles[send_particle_index].data(), n_parameters, MPI_DOUBLE,
                source, tag, MPI_COMM_WORLD, &p_req[1]);
      MPI_Isend(velocities[send_particle_index].data(), n_parameters,
                MPI_DOUBLE, source, tag, MPI_COMM_WORLD, &p_req[2]);

      particles_sent++;
    } else if (tag == TAG_FITNESS) {
      int particle_index = val;
      double recv_fitness;

      MPI_Request f_req[3];

      MPI_Irecv(&recv_fitness, 1, MPI_DOUBLE, source, tag, MPI_COMM_WORLD,
                &f_req[0]);
      MPI_Irecv(particles[particle_index].data(), n_parameters, MPI_DOUBLE,
                source, tag, MPI_COMM_WORLD, &f_req[1]);
      MPI_Irecv(velocities[particle_index].data(), n_parameters, MPI_DOUBLE,
                source, tag, MPI_COMM_WORLD, &f_req[2]);
      // cout << "[RANK " << source << "]: particle sent to master proc and
      // received fitness " << recv_fitness << endl;

      MPI_Status f_st[3];
      MPI_Waitall(3, f_req, f_st);

      if (recv_fitness < local_best_fitnesses[particle_index]) {
        for (int32_t i = 0; i < n_parameters; i++) {
          local_bests[particle_index][i] = particles[particle_index][i];
        }
        local_best_fitnesses[particle_index] = recv_fitness;
      }

      if (recv_fitness < global_best_fitness) {
        for (int32_t j = 0; j < n_parameters; j++) {
          global_best[j] = particles[particle_index][j];
        }
        global_best_fitness = recv_fitness;
        global_best_index = particle_index;
        cout << "[RANK 0 disp]: new GLOBAL BEST " << setw(3) << particle_index
             << " ";
        print_particle(particles[particle_index], velocities[particle_index],
                       local_best_fitnesses[particle_index]);
      }

      // get the random numbers for moving the particle
      double r1 = distribution(random_number_generator);
      double r2 = distribution(random_number_generator);

      // calculate the particle's new velocity and move it
      for (int32_t j = 0; j < n_parameters; j++) {
        velocities[particle_index][j] =
            (w * velocities[particle_index][j]) +
            (r1 * c1 *
             (local_bests[particle_index][j] - particles[particle_index][j])) +
            (r2 * c2 * (global_best[j] - particles[particle_index][j]));

        double new_position =
            particles[particle_index][j] + velocities[particle_index][j];

        // if the particle moves out of bounds, move it back into bounds and
        // adjust the velocity accordingly
        if (new_position < min_bounds[j]) {
          // cout << "particle " << particles[i][j] << " moved to " <<
          // new_position << " out of bounds, velocity was " <<
          // velocities[i][j];
          new_position = min_bounds[j] * 0.9;
          velocities[particle_index][j] =
              new_position - particles[particle_index][j];
          // cout << " new position " << new_position << " and velocity " <<
          // velocities[i][j] << endl;
        }

        if (new_position > max_bounds[j]) {
          // cout << "particle " << particles[i][j] << " moved to " <<
          // new_position << " out of bounds, velocity was " <<
          // velocities[i][j];
          new_position = max_bounds[j] * 0.9;
          velocities[particle_index][j] =
              new_position - particles[particle_index][j];
          // cout << " new position " << new_position << " and velocity " <<
          // velocities[i][j] << endl;
        }

        particles[particle_index][j] = new_position;
      }
    }
    if ((particles_sent + 1) % (max_particles / 10) == 0) {
      cout << "[RANK 0 disp]: CURRENT GLOBAL BEST: " << global_best_fitness
           << ", CURRENT GLOBAL BEST INDEX: " << global_best_index << endl;
      print_particle(global_best, velocities[global_best_index],
                     local_best_fitnesses[global_best_index]);
    }
  }

  // Notify workers to TERMINATE
  for (int i = 1; i < world_size; ++i) {
    int dummy = 0;
    MPI_Send(&dummy, 1, MPI_INT, i, TAG_TERMINATE, MPI_COMM_WORLD);
  }

  cout << "[RANK 0 end]: GLOBAL BEST: " << global_best_fitness
       << ", GLOBAL BEST INDEX: " << global_best_index << endl;
  print_particle(global_best, velocities[global_best_index],
                 global_best_fitness);
}

void workerProcess(ObjectiveFunction *of, int n_parameters) {
  bool running = true;
  int my_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
  while (true) {
    int val = 0;

    MPI_Request f_req[2];
    MPI_Status f_status;

    MPI_Isend(&val, 1, MPI_INT, 0, TAG_PARTICLE, MPI_COMM_WORLD, &f_req[0]);

    MPI_Irecv(&val, 1, MPI_INT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &f_req[1]);
    MPI_Wait(&f_req[1], &f_status);

    int tag = f_status.MPI_TAG;
    if (tag == TAG_PARTICLE) {
      int particle_index = val;
      vector<double> particle(n_parameters);
      vector<double> velocity(n_parameters);

      MPI_Request p_sreqs[2];
      MPI_Status p_sstatus[2];

      MPI_Irecv(particle.data(), n_parameters, MPI_DOUBLE, 0, TAG_PARTICLE,
                MPI_COMM_WORLD, &p_sreqs[0]);
      MPI_Irecv(velocity.data(), n_parameters, MPI_DOUBLE, 0, TAG_PARTICLE,
                MPI_COMM_WORLD, &p_sreqs[1]);
      MPI_Waitall(2, p_sreqs, p_sstatus);

      double fitness = of->evaluate(particle);

      MPI_Request p_rreqs[4];

      MPI_Isend(&particle_index, 1, MPI_INT, 0, TAG_FITNESS, MPI_COMM_WORLD,
                &p_rreqs[0]);
      MPI_Isend(&fitness, 1, MPI_DOUBLE, 0, TAG_FITNESS, MPI_COMM_WORLD,
                &p_rreqs[1]);
      MPI_Isend(particle.data(), n_parameters, MPI_DOUBLE, 0, TAG_FITNESS,
                MPI_COMM_WORLD, &p_rreqs[2]);
      MPI_Isend(velocity.data(), n_parameters, MPI_DOUBLE, 0, TAG_FITNESS,
                MPI_COMM_WORLD, &p_rreqs[3]);
    } else if (tag == TAG_TERMINATE) {
      break;
    }
  }
}

int main(int argc, char **argv) {
  int comm_sz; /* number of processes  */
  int my_rank; /* process rank         */

  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

  vector<string> arguments = vector<string>(argv, argv + argc);

  // Check to see command line arguments were correctly specified
  if (arguments.size() != 8 && arguments.size() != 9) {
    if (my_rank == 0) {
      string executable_name = arguments[0];
      cout << "Incorrect arguments." << endl;

      cout << "Usage for benchmark objective functions:" << endl;
      cout << "./" << executable_name
           << " <population size> <w> <c1> <c2> <max_epochs> "
              "<objective_function> <number_parameters"
           << endl;
      cout << "possible objective functions: sphere, ackley, griewank, "
              "rastrigin, rosenbrock, neural_network"
           << endl;

      cout << "Usage for neural network objective functions:" << endl;
      cout << "./" << executable_name
           << " <population size> <w> <c1> <c2> <max_epochs> neural_network "
              "<dataset_filename> <number_layers>"
           << endl;
    }
    MPI_Finalize();
    return -1;
  } else {
    int32_t population_size = stoi(arguments[1]);
    double w = stod(arguments[2]);
    double c1 = stod(arguments[3]);
    double c2 = stod(arguments[4]);
    int32_t max_particles = stoi(arguments[5]);
    string objective_function_name(arguments[6]);

    int32_t n_parameters = 0;
    ObjectiveFunction *of = NULL;
    if (my_rank == 0)
      cout << "creating objective function: '" << objective_function_name << "'"
           << endl;
    if (objective_function_name == "neural_network") {
      string dataset_filename = arguments[7];
      int32_t n_layers = stoi(arguments[8]);

      of = new NeuralNetwork(-20.0, 20.0, dataset_filename, n_layers);
      n_parameters = of->get_n_parameters();
    } else {
      n_parameters = stoi(arguments[7]);

      // Initialize the objective functions with the traditional parameter
      // bounds from
      // https://www.cs.cmu.edu/afs/cs/project/jair/pub/volume24/ortizboyer05a-html/node6.html
      if (objective_function_name == "sphere") {
        of = new Sphere(n_parameters, -5.12, 5.12);

        vector<double> best_parameters(n_parameters, 0);
        double best_fitness = of->evaluate(best_parameters);
        if (my_rank == 0)
          cout << "best fitness for optimal parameters: " << best_fitness
               << endl;
      } else if (objective_function_name == "ackley") {
        of = new Ackley(n_parameters, -30, 30);

        vector<double> best_parameters(n_parameters, 0);
        double best_fitness = of->evaluate(best_parameters);
        if (my_rank == 0)
          cout << "best fitness for optimal parameters: " << best_fitness
               << endl;
      } else if (objective_function_name == "griewank") {
        of = new Griewank(n_parameters, -600, 600);

        vector<double> best_parameters(n_parameters, 0);
        double best_fitness = of->evaluate(best_parameters);
        if (my_rank == 0)
          cout << "best fitness for optimal parameters: " << best_fitness
               << endl;
      } else if (objective_function_name == "rastrigin") {
        of = new Rastrigin(n_parameters, -2.048, 2.048);

        vector<double> best_parameters(n_parameters, 0);
        double best_fitness = of->evaluate(best_parameters);
        if (my_rank == 0)
          cout << "best fitness for optimal parameters: " << best_fitness
               << endl;
      } else if (objective_function_name == "rosenbrock") {
        of = new Rosenbrock(n_parameters, -5.12, 5.12);

        vector<double> best_parameters(n_parameters, 1.0);
        double best_fitness = of->evaluate(best_parameters);
        if (my_rank == 0)
          cout << "best fitness for optimal parameters: " << best_fitness
               << endl;
      } else {
        if (my_rank == 0) {
          cerr << "Unknown objective function '" << objective_function_name
               << "'" << endl;
          cerr << "Options:" << endl;
          cerr << "\tsphere" << endl;
          cerr << "\tackley" << endl;
          cerr << "\tgriewank" << endl;
          cerr << "\trastrigin" << endl;
          cerr << "\tneural_network" << endl;
        }
        exit(1);
      }
    }

    vector<double> min_bounds = of->get_min_bounds();
    vector<double> max_bounds = of->get_max_bounds();

    if (my_rank == 0) {

      for (int32_t i = 0; i < n_parameters; i++) {
        cout << "parameter [" << i << "] min bound: " << min_bounds[i]
             << ", max bound: " << max_bounds[i] << endl;
      }
    }

    if (my_rank == 0) {
      masterProcess(w, c1, c2, population_size, max_particles, n_parameters,
                    min_bounds, max_bounds);
    } else {
      workerProcess(of, n_parameters);
    }
  }

  MPI_Finalize();
  return 0;
}