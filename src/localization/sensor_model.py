import numpy as np
import math
from scan_simulator_2d import PyScanSimulator2D

import rospy
import tf
from nav_msgs.msg import OccupancyGrid
from tf.transformations import quaternion_from_euler

class SensorModel:


    def __init__(self):
        # Fetch parameters
        
        self.map_topic = rospy.get_param("~map_topic")
        self.num_beams_per_particle = rospy.get_param("~num_beams_per_particle")
        self.scan_theta_discretization = rospy.get_param("~scan_theta_discretization")
        self.scan_field_of_view = rospy.get_param("~scan_field_of_view")
        self.lidar_scale_to_map_scale = rospy.get_param("~lidar_scale_to_map_scale")
        

        ####################################
        # TODO: Tune these parameters
        self.alpha_hit = 0.74
        self.alpha_short = 0.07
        self.alpha_max = 0.07
        self.alpha_rand = 0.12
        self.alphas = [self.alpha_hit, self.alpha_short, self.alpha_max, self.alpha_rand]
        self.sigma_hit = 8.0
        self.eps = 1

        # Your sensor table will be a `table_width` x `table_width` np array:
        self.table_width = 201
        ####################################

        # Precompute the sensor model table
        self.sensor_model_table = np.zeros((self.table_width, self.table_width))
        self.precompute_sensor_model()
        # Create a simulated laser scan
        self.scan_sim = PyScanSimulator2D(
                self.num_beams_per_particle,
                self.scan_field_of_view,
                0, # This is not the simulator, don't add noise
                0.01, # This is used as an epsilon
                self.scan_theta_discretization) 

        # Subscribe to the map
        self.map = None
        self.map_set = False
        rospy.Subscriber(
                self.map_topic,
                OccupancyGrid,
                self.map_callback,
                queue_size=1)
        
        
    def calc_probability(self, z_k, d, z_max):
        p_hit = 0.0
        p_short = 0.0
        p_max = 0.0
        p_rand = 0.0
        if 0 <= z_k <= z_max:
            term_1 = (1/(2*np.pi*self.sigma_hit**2)**0.5)
            exp_num = -1*(z_k - d)**2
            exp_denom = 2 * self.sigma_hit**2
            p_hit = term_1 * math.exp(exp_num/exp_denom)
        if 0 <= z_k <= d and d != 0:
            p_short = 2/d * (1 - z_k/d)
        if z_max-self.eps <= z_k <= z_max and z_k >= z_max - self.eps:
            p_max = 1/self.eps
        if 0 <= z_k <= z_max:
            p_rand = 1/z_max
        return [self.alpha_hit * p_hit, self.alpha_short * p_short, self.alpha_max * p_max, self.alpha_rand * p_rand]
    
    def precompute_sensor_model(self):
        """
        Generate and store a table which represents the sensor model.
        
        For each discrete computed range value, this provides the probability of 
        measuring any (discrete) range. This table is indexed by the sensor model
        at runtime by discretizing the measurements and computed ranges from
        RangeLibc.
        This table must be implemented as a numpy 2D array.

        Compute the table based on class parameters alpha_hit, alpha_short,
        alpha_max, alpha_rand, sigma_hit, and table_width.

        args:
            N/A
        
        returns:
            No return type. Directly modify `self.sensor_model_table`.
        """
        z_max = self.table_width - 1
        for d in range(self.table_width):
            hit_array = np.zeros(self.table_width)
            total_hit = 0.0
            for z_k in range(self.table_width):
                p_short = 0
                p_max = 0
                p_rand = 0
                p_hit = ((2*np.pi*self.sigma_hit**2)**(-0.5))*np.e**(-0.5*((z_k-d)**2)/self.sigma_hit**2)
                if z_k <= d and d != 0:
                    p_short = 2.0/float(d) * (1 - z_k/float(d))
                if z_k == self.table_width - 1:
                    p_max = 1.0
                if z_k <= z_max:
                    p_rand = 1.0/float(z_max)
                hit_array[z_k] = p_hit
                total_hit += p_hit
                total = (self.alpha_short * p_short) + (self.alpha_max * p_max) + (self.alpha_rand * p_rand)
                self.sensor_model_table[z_k][d] = total
            self.sensor_model_table[:, d] += self.alpha_hit * hit_array/total_hit

        self.sensor_model_table = self.sensor_model_table/self.sensor_model_table.sum(axis=0,keepdims=1)
        self.map_set = True


    def evaluate(self, particles, observation):
        """
        Evaluate how likely each particle is given
        the observed scan.

        args:
            particles: An Nx3 matrix of the form:
            
                [x0 y0 theta0]
                [x1 y0 theta1]
                [    ...     ]

            observation: A vector of lidar data measured
                from the actual lidar.

        returns:
           probabilities: A vector of length N representing
               the probability of each particle existing
               given the observation and the map.
        """

        if not self.map_set:
            return

        ####################################
        # TODO
        # Evaluate the sensor model here!
        #
        # You will probably want to use this function
        # to perform ray tracing from all the particles.
        # This produces a matrix of size N x num_beams_per_particle 
        N = len(particles)
        probabilities = np.zeros(N)
        to_px = 1.0/(self.map_resolution*self.lidar_scale_to_map_scale)
       
        # measured distance
        scaled_observations = observation * to_px

        # ray tracing: true distance
        scaled_scans = self.scan_sim.scan(particles) * to_px
        
        scaled_observations[scaled_observations > 200] = 200.0
        scaled_observations[scaled_observations < 0] = 0.0
        scaled_scans[scaled_scans < 0] = 0.0
        scaled_scans[scaled_scans > 200] = 200.0
        for p in range(N): 
            current_prob = 1.0
            for n in range(self.num_beams_per_particle):
                d = np.rint(scaled_scans[p][n]).astype(int)
                z = np.rint(scaled_observations[n]).astype(int)
                current_prob *= self.sensor_model_table[z][d]
            probabilities[p] = current_prob

        #Descrease peaks in distribution 2.175
        return np.power(probabilities,1/2.2)

        

        ####################################

    def map_callback(self, map_msg):
        # Convert the map to a numpy array
        self.map = np.array(map_msg.data, np.double)/100.
        self.map = np.clip(self.map, 0, 1)

        # Convert the origin to a tuple
        origin_p = map_msg.info.origin.position
        origin_o = map_msg.info.origin.orientation
        origin_o = tf.transformations.euler_from_quaternion((
                origin_o.x,
                origin_o.y,
                origin_o.z,
                origin_o.w))
        origin = (origin_p.x, origin_p.y, origin_o[2])

        # Initialize a map with the laser scan
        self.scan_sim.set_map(
                self.map,
                map_msg.info.height,
                map_msg.info.width,
                map_msg.info.resolution,
                origin,
                0.5) # Consider anything < 0.5 to be free

        # Make the map set
        self.map_set = True
        self.map_resolution = map_msg.info.resolution
        print("Map initialized")


