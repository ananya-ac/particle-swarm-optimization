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
using std::uniform_int_distribution;
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

  int32_t my_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

  vector<string> arguments = vector<string>(argv, argv + argc);

  // Check to see command line arguments were correctly specified
  if ((arguments.size() != 8 &&
       strcmp(arguments[6].c_str(), "neural_network") != 0) ||
      (arguments.size() != 9 &&
       strcmp(arguments[6].c_str(), "neural_network") == 0)) {
    string executable_name = arguments[0];
    cerr << "Incorrect arguments." << endl;

    cout << "Usage for benchmark objective functions:" << endl;
    cerr << "./" << executable_name
         << " <population size> <w> <c1> <c2> <max_epochs> "
            "<objective_function> <number_parameters>"
         << endl;
    cerr << "possible objective functions: sphere, ackley, griewank, "
            "rastrigin, rosenbrock, neural_network"
         << endl;

    cout << "Usage for neural network objective functions:" << endl;
    cerr << "./" << executable_name
         << " <population size> <w> <c1> <c2> <max_particles> neural_network "
            "<dataset_filename> <number_layers>"
         << endl;
    exit(1);
  }

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

    // Initialize the objective functions with the traditional parameter bounds
    // from
    // https://www.cs.cmu.edu/afs/cs/project/jair/pub/volume24/ortizboyer05a-html/node6.html
    if (objective_function_name == "sphere") {
      of = new Sphere(n_parameters, -5.12, 5.12);

      vector<double> best_parameters(n_parameters, 0);
      double best_fitness = of->evaluate(best_parameters);
      cout << "best fitness for optimal parameters: " << best_fitness << endl;

    } else if (objective_function_name == "ackley") {
      of = new Ackley(n_parameters, -30, 30);

      vector<double> best_parameters(n_parameters, 0);
      double best_fitness = of->evaluate(best_parameters);
      cout << "best fitness for optimal parameters: " << best_fitness << endl;

    } else if (objective_function_name == "griewank") {
      of = new Griewank(n_parameters, -600, 600);

      vector<double> best_parameters(n_parameters, 0);
      double best_fitness = of->evaluate(best_parameters);
      cout << "best fitness for optimal parameters: " << best_fitness << endl;

    } else if (objective_function_name == "rastrigin") {
      of = new Rastrigin(n_parameters, -2.048, 2.048);

      vector<double> best_parameters(n_parameters, 0);
      double best_fitness = of->evaluate(best_parameters);
      cout << "best fitness for optimal parameters: " << best_fitness << endl;

    } else if (objective_function_name == "rosenbrock") {
      of = new Rosenbrock(n_parameters, -5.12, 5.12);

      vector<double> best_parameters(n_parameters, 1.0);
      double best_fitness = of->evaluate(best_parameters);
      if (my_rank == 0)
        cout << "best fitness for optimal parameters: " << best_fitness << endl;

    } else {
      cerr << "Unknown object function '" << objective_function_name << "'"
           << endl;
      cerr << "Options:" << endl;
      cerr << "\tsphere" << endl;
      cerr << "\tackley" << endl;
      cerr << "\tgriewank" << endl;
      cerr << "\trastrigin" << endl;
      cerr << "\tneural_network" << endl;
      exit(1);
    }
  }

  random_device rd;
  mt19937 random_number_generator(rd());
  uniform_real_distribution<double> distribution(0.0, 1.0);

  int index = 0;
  bool terminate = false;
  int particles_sent = 0;
  vector<double> global_best(n_parameters);
  double global_best_fitness = numeric_limits<double>::max();

  if (my_rank == 0) {

    vector<double> min_bounds = of->get_min_bounds();
    vector<double> max_bounds = of->get_max_bounds();

    vector<vector<double>> particles(population_size,
                                     vector<double>(n_parameters));
    vector<vector<double>> velocities(population_size,
                                      vector<double>(n_parameters));
    vector<vector<double>> local_bests(population_size,
                                       vector<double>(n_parameters));

    for (int32_t i = 0; i < population_size; i++) {
      for (int32_t j = 0; j < n_parameters; j++) {
        double r = distribution(random_number_generator);
        particles[i][j] = ((max_bounds[j] - min_bounds[j]) * r) + min_bounds[j];
        local_bests[i][j] = particles[i][j];

        velocities[i][j] =
            0.05 * (((max_bounds[j] - min_bounds[j]) * r) + min_bounds[j]);
      }
    }

    for (int32_t i = 0; i < n_parameters; i++) {
      cout << "parameter [" << i << "] min bound: " << min_bounds[i]
           << ", max bound: " << max_bounds[i] << endl;
    }

    while (true) {

      int idle_worker;

      MPI_Status status1;

      MPI_Recv(&idle_worker, 1, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG,
               MPI_COMM_WORLD, &status1);

      int tag = status1.MPI_TAG;

      MPI_Request particle_req;
      MPI_Request work_req;
      MPI_Status particle_status;
      MPI_Status work_status;

      if (tag == 1) {

        int target = status1.MPI_SOURCE;

        if (particles_sent < max_particles && index < population_size) {

          vector<double> send_particle(n_parameters);
          vector<double> send_velocity(n_parameters);

          vector<double> work(n_parameters + 1);
          work[0] = index;
          send_particle = particles[index];
          send_velocity = velocities[index];

          particles_sent++;

          if (particles_sent >= max_particles) {

            for (int i = 0; i < number_processes; i++)
              MPI_Send(send_particle.data(), n_parameters, MPI_DOUBLE, i, 2,
                       MPI_COMM_WORLD);

            break;
          }

          std::copy(send_velocity.begin(), send_velocity.end(),
                    work.begin() + 1);

          MPI_Isend(send_particle.data(), n_parameters, MPI_DOUBLE, target, 1,
                    MPI_COMM_WORLD, &particle_req);

          MPI_Isend(work.data(), n_parameters + 1, MPI_DOUBLE, target, 1,
                    MPI_COMM_WORLD, &work_req);

          index++;

          if (index == population_size - 1)
            index = 0;
        }

        // MPI_Wait(&work_req, &work_status);
        // MPI_Wait(&particle_req, &particle_status);
      }

      if (tag == 2) {

        MPI_Status status2;
        MPI_Request recv_req;

        vector<double> recv_package(n_parameters * 2 + 2);

        MPI_Irecv(recv_package.data(), n_parameters * 2 + 2, MPI_DOUBLE,
                  status1.MPI_SOURCE, 3, MPI_COMM_WORLD, &recv_req);

        MPI_Wait(&recv_req, &status2);

        int recv_index;
        double recv_fitness;

        recv_index = recv_package[0];
        recv_fitness = recv_package[1];

        vector<double> recv_particle(recv_package.begin() + 2,
                                     recv_package.begin() + 2 + n_parameters);
        vector<double> recv_velocity(recv_package.begin() + 2 + n_parameters,
                                     recv_package.begin() + 2 +
                                         2 * n_parameters);

        bool cond1 = global_best_fitness > recv_fitness;
        bool cond2 = of->evaluate(local_bests[recv_index]) > recv_fitness;
        if (cond1) {
          global_best_fitness = recv_fitness;
          velocities[recv_index] = recv_velocity;
          global_best = recv_particle;
        }

        if (cond2) {
          local_bests[recv_index] = recv_particle;
          velocities[recv_index] = recv_velocity;
        }

        double r1 = distribution(random_number_generator);
        double r2 = distribution(random_number_generator);

        for (int32_t j = 0; j < n_parameters; j++) {
          velocities[recv_index][j] =
              (w * velocities[recv_index][j]) +
              (r1 * c1 *
               (local_bests[recv_index][j] - particles[recv_index][j])) +
              (r2 * c2 * (global_best[j] - particles[recv_index][j]));

          double new_position =
              particles[recv_index][j] + velocities[recv_index][j];
          if (new_position < min_bounds[j]) {
            new_position = min_bounds[j] * 0.9;
            velocities[recv_index][j] = new_position - particles[recv_index][j];
          }

          if (new_position > max_bounds[j]) {
            new_position = max_bounds[j] * 0.9;
            velocities[recv_index][j] = new_position - particles[recv_index][j];
          }

          particles[recv_index][j] = new_position;
        }

        cout << "particles moved " << particles_sent << " global best "
             << global_best_fitness << endl;

        print_particle(global_best, vector<double>(), global_best_fitness);
      }
    }

  }

  else {

    while (true) {

      double fitness;
      int index;
      vector<double> particle(n_parameters);
      vector<double> recv_id_vel(n_parameters + 1);
      vector<double> send_package(2 * n_parameters + 2);

      int idle(0);
      MPI_Request send_req;
      MPI_Isend(&idle, 1, MPI_INT, 0, 1, MPI_COMM_WORLD, &send_req);

      MPI_Status idv_status;
      MPI_Request idv_req;
      MPI_Status p;
      MPI_Request p_req;

      MPI_Irecv(particle.data(), n_parameters, MPI_DOUBLE, 0, MPI_ANY_TAG,
                MPI_COMM_WORLD, &p_req);

      MPI_Irecv(recv_id_vel.data(), n_parameters + 1, MPI_DOUBLE, 0,
                MPI_ANY_TAG, MPI_COMM_WORLD, &idv_req);

      MPI_Wait(&p_req, &p);

      int tag = p.MPI_TAG;

      if (tag == 1) {

        fitness = of->evaluate(particle);

        MPI_Wait(&idv_req, &idv_status);
        index = recv_id_vel[0];
        vector<double> velocity(recv_id_vel.begin() + 1,
                                recv_id_vel.begin() + 1 + n_parameters);

        send_package[0] = index;
        send_package[1] = fitness;

        std::copy(particle.begin(), particle.end(), send_package.begin() + 2);
        std::copy(velocity.begin(), velocity.end(),
                  send_package.begin() + 2 + n_parameters);

        MPI_Request idle_req;
        MPI_Isend(&idle, 1, MPI_INT, 0, 2, MPI_COMM_WORLD, &idle_req);

        MPI_Request pcg_req;
        MPI_Isend(send_package.data(), 2 * n_parameters + 2, MPI_DOUBLE, 0, 3,
                  MPI_COMM_WORLD, &pcg_req);
      } else {
        break;
      }
    }
  }
  if (my_rank == 0)
    cout << "particles moved " << particles_sent << " global best "
         << global_best_fitness << endl;

  MPI_Finalize();

  return 0;
}