/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random> // Need this for sampling from distributions
#include <algorithm>
#include <iostream>
#include <numeric>
#include <iterator>

#include "particle_filter.h"

#define NUMBER_OF_PARTICLES 50 // Can be decreased (even 12 particles can pass the test)
#define EPS 0.001  // Just a small number

using namespace std;

// The Particle Filter functions
// Initialize all particles to first position
void ParticleFilter::init(double x, double y, double theta, double std[]) {
	static default_random_engine gen;
    //gen.seed(123);
    num_particles = NUMBER_OF_PARTICLES; // init number of particles to use
	// Create normal distributions for x, y and theta.
	normal_distribution<double> dist_x(x, std[0]);
	normal_distribution<double> dist_y(y, std[1]);
	normal_distribution<double> dist_theta(theta, std[2]);
	particles.resize(num_particles); // Resize the `particles` vector to fit desired number of particles
	weights.resize(num_particles, 1.0);
	double init_weight = 1.0/num_particles;
	for (int i = 0; i < num_particles; i++){
		particles[i].id = i;
		particles[i].x = dist_x(gen);
		particles[i].y = dist_y(gen);
		particles[i].theta = dist_theta(gen);
		particles[i].weight = init_weight; //1.0; //
	}	
	is_initialized = true;
}

// Add measurements to each particle and add random Gaussian noise.
void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// Some constants to save computation power
	const double vel_d_t = velocity * delta_t;
	const double yaw_d_t = yaw_rate * delta_t;
	const double vel_yaw = velocity/yaw_rate;
	static default_random_engine gen;
    //gen.seed(321);
    normal_distribution<double> dist_x(0.0, std_pos[0]);
	normal_distribution<double> dist_y(0.0, std_pos[1]);
	normal_distribution<double> dist_theta(0.0, std_pos[2]);
	for (int i = 0; i < num_particles; i++){
        if (fabs(yaw_rate) < EPS){
            particles[i].x += vel_d_t * cos(particles[i].theta);
            particles[i].y += vel_d_t * sin(particles[i].theta);
            // particles[i].theta unchanged if yaw_rate is too small
        }
        else{
            const double theta_new = particles[i].theta + yaw_d_t;
            particles[i].x += vel_yaw * (sin(theta_new) - sin(particles[i].theta));
            particles[i].y += vel_yaw * (-cos(theta_new) + cos(particles[i].theta));
            particles[i].theta = theta_new;
        }
        // Add random Gaussian noise
        particles[i].x += dist_x(gen);
        particles[i].y += dist_y(gen);
        particles[i].theta += dist_theta(gen);
	}
}


/**
* dataAssociation Finds which observations correspond to which landmarks (likely by using
*   a nearest-neighbors data association).
* @param predicted Vector of predicted landmark observations
* @param observations Vector of landmark observations
*/
void ParticleFilter::dataAssociation(Particle &particle, std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations)
{
	LandmarkObs closest;

	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	/* Loop through each observations */
	for (auto obs : observations) {

		double shortest = 1E10; // some number larger than any possible measurement 

		/* Loop through each predicted landmarks and find the closest landmark match */
		for (auto pred : predicted) {
			double distance = dist(obs.x, obs.y, pred.x, pred.y);
			if (distance < shortest) {
				shortest = distance;
				closest = pred;
			}
		}

		/* Record the closest landmark id for a given particle */
		particle.associations.push_back(closest.id);
		particle.sense_x.push_back(closest.x);
		particle.sense_y.push_back(closest.y);
	}

}

// Update the weights of each particle using a mult-variate Gaussian distribution.
void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		std::vector<LandmarkObs> observations, Map map_landmarks) {

	const double sigma_x = std_landmark[0];
	const double sigma_y = std_landmark[1];
	const double sigma_xx = sigma_x * sigma_x;
	const double sigma_yy = sigma_y * sigma_y;

	/* Loop through each particle */
	for (int i = 0; i < num_particles; i++) 
	{
		// collect all landmarks within sensor range of the current particle in a vector predicted.
		Particle p = particles[i];

		// Transform each observations from the particle coordinate system to the MAP system
		std::vector<LandmarkObs> transformed_observations;
		for (auto observation : observations) {

			LandmarkObs transformed_observation;
			transformed_observation.x = p.x + observation.x * cos(p.theta) - observation.y * sin(p.theta);
			transformed_observation.y = p.y + observation.x * sin(p.theta) + observation.y * cos(p.theta);
			transformed_observation.id = observation.id;

			transformed_observations.push_back(transformed_observation);
		}

		// Fetch all landmarks that are within sight of the particle (i.e.) within sensor_range.
		std::vector<LandmarkObs> predicted_landmarks;
		for (auto landmark : map_landmarks.landmark_list) {

			double distance = dist(p.x, p.y, landmark.x_f, landmark.y_f);
			if (distance < sensor_range) {
				LandmarkObs one_landmark;
				one_landmark.id = landmark.id_i;
				one_landmark.x = landmark.x_f;
				one_landmark.y = landmark.y_f;
				predicted_landmarks.push_back(one_landmark);
			}
		}


		// For every observation of the particle, associate the nearest landmark
		dataAssociation(p, predicted_landmarks, transformed_observations);

		// Calculate the particle's probability using multi-variate Gaussian distribution. 
		double probability = 1;
		for (int j = 0; j < transformed_observations.size(); ++j)
		{
			double dx = transformed_observations.at(j).x - p.sense_x.at(j);
			double dy = transformed_observations.at(j).y - p.sense_y.at(j);
			probability *= 1.0 / (2 * M_PI*sigma_x*sigma_y) * exp(-dx*dx / (2 * sigma_xx))* exp(-dy*dy / (2 * sigma_yy));
		}

		p.weight = probability;
		particles[i] = p;
		weights[i] = probability;
	}
}

// Resampling particles with replacement with probability proportional to their weight. 
void ParticleFilter::resample() {

	try
	{
		static default_random_engine gen;
		//gen.seed(123);

		discrete_distribution<> dist_particles(weights.begin(), weights.end());
		vector<Particle> new_particles(num_particles);

		for (int i = 0; i < num_particles; i++) {
			new_particles[i] = particles[dist_particles(gen)];
		}
		particles = new_particles;
	}
	catch (int ex)
	{
		cout << "resampling exception: " << ex << endl;
	}
}


Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations= associations;
 	particle.sense_x = sense_x;
 	particle.sense_y = sense_y;

 	return particle;
}


string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
