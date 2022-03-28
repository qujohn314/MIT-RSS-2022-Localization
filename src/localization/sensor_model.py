import numpy as np
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
        self.eps = 0.1

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
        
        # self.map_resolution = 
        
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
        print("Computing Sensor Model Table")
        z_max = self.table_width - 1
        vals = np.zeros(self.table_width,self.table_width)
        for z in range(self.table_width):
            p = np.zeros(4)
            total_hit = 0
            for d in range(self.table_width):
                # p[0] = 1.0/(np.sqrt(2.0*np.pi*self.sigma_hit**2))*np.exp(-((z - d)**2)/(2.0*self.sigma_hit**2))
                # p[1] = 2.0/d*(1 - z/float(d)) if (0 <= z <= d and d != 0) else 0
                # p[2] = 1.0/self.eps if (z_max - self.eps) <= z <= z_max else 0
                # p[3] = 1.0/float(z_max) if 0 <= z <= z_max else 0                
                # self.sensor_model_table[z][d] = np.dot(p, self.alphas)
                vals[z][d] = self.calc_probability(z, d, z_max)
                self.sensor_model_table[z][d] = vals[z][d][0]
                total_hit += vals[z][d][0]

            #normalization
            for d in range(self.table_width):
                self.sensor_model_table[z][d] = vals[z][d][0]/total_hit

        print(self.sensor_model_table.sum(axis=1,keepdims=1))
        #self.sensor_model_table = self.sensor_model_table / np.linalg.norm(self.sensor_model_table, axis=1) # normalize

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

            observation: A vector of lidar data meas.ured
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
        scaled_observations = observation * to_px

        scans = self.scan_sim.scan(particles)
        print(scans)
        for p in range(N): 
            current_prob = 1.0
            for n in range(self.num_beams_per_particle):
                d = int(scans[p, n]) 
                z = int(scaled_observations[n])
                current_prob *= self.sensor_model_table[z,d]
                probabilities[p] += current_prob
            probabilities[p] = probabilities[p]/self.num_beams_per_particle # average of probs across beams
            probabilities[p] = current_prob
        return probabilities

        

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


if __name__ == "__main__":
    SensorModel()
