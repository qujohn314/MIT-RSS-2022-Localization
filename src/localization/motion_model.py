import rospy
import numpy as np


class MotionModel:

    def __init__(self):

        ####################################
        # Do any precomputation for the motion
        # model here.

        self.num_particles = int(rospy.get_param("~num_particles"))
        self.deltas = np.zeros((self.num_particles, 3))
        self.deterministic = bool(rospy.get_param("~deterministic"))

        ####################################

    def evaluate(self, particles, odometry):
        """
        Update the particles to reflect probable
        future states given the odometry data.

        args:
            particles: An Nx3 matrix of the form:
            
                [x0 y0 theta0]
                [x1 y0 theta1]
                [    ...     ]

            odometry: A 3-vector [dx dy dtheta]

        returns:
            particles: An updated matrix of the
                same size
        """
        
        ####################################
        # convert 
        
        new_odometry_x = odometry[0] + np.random.normal(loc=0.0, scale=0.05, size=particles.shape[0])
        new_odometry_y = odometry[1] + np.random.normal(loc=0.0, scale=0.05, size=particles.shape[0])
        new_odometry_theta = odometry[2] + np.random.normal(loc=0.0, scale=0.05, size=particles.shape[0])

        if self.deterministic:
            new_odometry_x = odometry[0]
            new_odometry_y = odometry[1]
            new_odometry_theta = odometry[2]
            
        cosines = np.cos(particles[:, 2])
        sines = np.sin(particles[:, 2])

        rotation_map = [[cosines, sines, 0], [-sines, cosines, 0], [0, 0, 1]]
        self.deltas[:, 0] = cosines*new_odometry_x - sines*new_odometry_y
        self.deltas[:,1] = sines*new_odometry_x + cosines*new_odometry_y
        self.deltas[:,2] = new_odometry_theta

        particles[:, :] += self.deltas
        # if not self.deterministic:
        #     particles[:, 0] += np.random.normal(loc=0.0, scale=0.05, size=particles.shape[0])
        #     particles[:, 1] += np.random.normal(loc=0.0, scale=0.05, size=particles.shape[0])
        #     particles[:, 2] += np.random.normal(loc=0.0, scale=0.05, size=particles.shape[0])


        return particles        

        ####################################
