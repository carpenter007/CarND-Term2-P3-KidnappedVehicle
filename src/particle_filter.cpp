/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>
#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {

  // Set the number of particles
  num_particles = 100;

  // Use the standard c++ random engine
  default_random_engine gen;

  // Create normal distributions for x, y and theta
  normal_distribution<double> dist_x(x, std[0]);
  normal_distribution<double> dist_y(y, std[1]);
  normal_distribution<double> dist_theta(theta, std[2]);

  for (int i = 0; i < num_particles; i++) {

    Particle sampleParticle;
    // Sample  and from these normal distrubtions
    sampleParticle.id = i;
    sampleParticle.weight = 1.0;
    sampleParticle.x = dist_x(gen);
    sampleParticle.y = dist_y(gen);
    sampleParticle.theta = dist_theta(gen);
    particles.push_back(sampleParticle);

  }

  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[],
                                double velocity, double yaw_rate) {

  default_random_engine gen;
  // Add measurements to each particle
  for (int i = 0; i < num_particles; i++) {

    if (fabs(yaw_rate) < 0.00001) {
      particles[i].x += velocity * delta_t * cos(particles[i].theta);
      particles[i].y += velocity * delta_t * sin(particles[i].theta);
    } else {
      particles[i].x += (velocity / yaw_rate)
          * (sin(particles[i].theta + (yaw_rate * delta_t))
              - sin(particles[i].theta));
      particles[i].y += (velocity / yaw_rate)
          * (cos(particles[i].theta)
              - cos(particles[i].theta + (yaw_rate * delta_t)));
      particles[i].theta += yaw_rate * delta_t;
    }

    // Apply normal distributions for x, y and theta
    normal_distribution<double> dist_x(particles[i].x, std_pos[0]);
    normal_distribution<double> dist_y(particles[i].y, std_pos[1]);
    normal_distribution<double> dist_theta(particles[i].theta, std_pos[2]);

    particles[i].x = dist_x(gen);
    particles[i].y = dist_y(gen);
    particles[i].theta = dist_theta(gen);
  }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted,
                                     std::vector<LandmarkObs>& observations) {
  // Find the predicted measurement that is closest to each observed measurement and assign the
  //   observed measurement to this particular landmark.

  for (int i = 0; i < observations.size(); i++) {
    LandmarkObs nearest_landmark;
    double min_distance = 1000;
    vector<double> dists;

    for (LandmarkObs landmark : predicted) {
      double dst = dist(landmark.x, landmark.y, observations[i].x,
                        observations[i].y);
      if (dst < min_distance) {
        min_distance = dst;
        nearest_landmark = landmark;
      }
      observations[i].id = nearest_landmark.id; /* Save landmark id */
    }
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
                                   const std::vector<LandmarkObs> &observations,
                                   const Map &map_landmarks) {
  //   Update the weights of each particle using a mult-variate Gaussian distribution. You can read
  //   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
  // NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
  //   according to the MAP'S coordinate system. You will need to transform between the two systems.
  //   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
  //   The following is a good resource for the theory:
  //   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
  //   and the following is a good resource for the actual equation to implement (look at equation 
  //   3.33
  //   http://planning.cs.uiuc.edu/node99.html

  double gauss_norm = (1 / (2 * M_PI * std_landmark[0] * std_landmark[1]));

  // For each particle object
  for (int i = 0; i < num_particles; i++) {
    // 1. All input observations have to be transformed into map coordinates with respect to the current particle object -> create a observations_transf vector
    std::vector<LandmarkObs> observations_transf;
    for (LandmarkObs obsrv : observations) {
      LandmarkObs obs_t;
      // transformation requires both rotation AND translation from vehicle's to map's coordinate system
      obs_t.x = particles[i].x + (cos(particles[i].theta) * obsrv.x)
          - (sin(particles[i].theta) * obsrv.y);  // x_map= x_part + (np.cos(theta) * x_obs) - (np.sin(theta) * y_obs)
      obs_t.y = particles[i].y + (sin(particles[i].theta) * obsrv.x)
          + (cos(particles[i].theta) * obsrv.y);  // y_map= y_part + (np.sin(theta) * x_obs) + (np.cos(theta) * y_obs)
      //obs_t.id has to be set in dataAssisiation function
      observations_transf.push_back(obs_t);
    }
    // 2. create predicted vector <LandmarkObs>. Predicted landmarks will be only these map_landmarks, which are in the sensor range.
    std::vector<LandmarkObs> predictedLmks;
    for (Map::single_landmark_s singleLmk : map_landmarks.landmark_list) {
      LandmarkObs validLmk;
      if (dist(singleLmk.x_f, singleLmk.y_f, particles[i].x, particles[i].y)
          <= sensor_range) {
        validLmk.x = singleLmk.x_f;
        validLmk.y = singleLmk.y_f;
        validLmk.id = singleLmk.id_i;
        predictedLmks.push_back(validLmk);
      }
    }

    // 3. Assosiate map and observations:
    //    Call of dataAssociation with observation_trans as a reference pointer and the choosen map_landmarks as "predicted" param.
    //    In the observations_transf objects the IDs will be updated by the nearest predicted objects id (map landmark)
    dataAssociation(predictedLmks, observations_transf);

    // 4. Calculate: Distance from the particle position to each landmark on the map, minus the calculated distance to each assosiated landmark.
    // calculate normalization term
    particles[i].weight = 1.0;
    double exponentE = 0.0;
    for (LandmarkObs obsrv : observations_transf) {
      //calculate exponent
      vector<LandmarkObs>::iterator it = find_if(predictedLmks.begin(),
                                                 predictedLmks.end(),
                                                 [&](const LandmarkObs & o) {
                                                   return(o.id == obsrv.id);
                                                 });

      if (it == predictedLmks.end()) {
        // don't use this observation
      } else {
        // calculate weight using normalization terms and exponent
        exponentE = -((pow(it->x - obsrv.x, 2)) / (2 * pow(std_landmark[0], 2))
            + (pow(it->y - obsrv.y, 2)) / (2 * pow(std_landmark[1], 2)));
        // 5. Recalculation of the weight of each particle
        particles[i].weight *= gauss_norm * exp(exponentE);
      }
    }
  }
}

void ParticleFilter::resample() {
  // Resample particles with replacement with probability proportional to their weight.
  // NOTE: You may find std::discrete_distribution helpful here.
  //   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

  /* Pseudo code of random wheel:
   * p_new = []
   * index = int(random.random()*N)
   * beta = 0.0
   * mw = max(w)
   * for i in range(N):
   *  beta += random.random() *  2.0 * mw
   *  while beta > w[index]:
   *    beta -= w[index]
   *    index = (index+1) % N
   *  p_new.append(p[index])
   * p = p_new
   */

  vector<Particle> p_new;

  // Choose random integer for starting particle
  random_device rd;     // only used once to initialise (seed) engine
  mt19937 rng(rd());  // random-number engine used (Mersenne-Twister in this case)
  uniform_int_distribution<int> uni(0, num_particles - 1);  // guaranteed unbiased
  auto index = uni(rng);

  // Make a raw weights vector
  vector<double> w;
  for (int i = 0; i < num_particles; i++) {
    w.push_back(particles[i].weight);
  }

  // Get max weight
  double w_max = *max_element(begin(w), end(w));
  double beta = 0.0;
  uniform_real_distribution<double> distribution(0.0, w_max);

  // Loop num_particles times and pick weighted elements
  for (int i = 0; i < num_particles; i++) {
    beta += distribution(rng) * 2.0;
    while (beta > w[index]) {
      beta -= w[index];
      index = (index + 1) % num_particles;
    }
    p_new.push_back(particles[index]);
  }
  weights = w;
  particles = p_new;
}

Particle ParticleFilter::SetAssociations(Particle& particle,
                                         const std::vector<int>& associations,
                                         const std::vector<double>& sense_x,
                                         const std::vector<double>& sense_y) {
  //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates

  particle.associations = associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  stringstream ss;
  copy(v.begin(), v.end(), ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length() - 1);  // get rid of the trailing space
  return s;
}
string ParticleFilter::getSenseX(Particle best) {
  vector<double> v = best.sense_x;
  stringstream ss;
  copy(v.begin(), v.end(), ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length() - 1);  // get rid of the trailing space
  return s;
}
string ParticleFilter::getSenseY(Particle best) {
  vector<double> v = best.sense_y;
  stringstream ss;
  copy(v.begin(), v.end(), ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length() - 1);  // get rid of the trailing space
  return s;
}
