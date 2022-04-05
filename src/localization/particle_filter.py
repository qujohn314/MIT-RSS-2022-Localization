#!/usr/bin/env python2

import rospy
from sensor_model import SensorModel
from motion_model import MotionModel

from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry, OccupancyGrid
from geometry_msgs.msg import PoseWithCovariance, PoseWithCovarianceStamped, Transform, TransformStamped, PoseArray, Pose, Quaternion

import numpy as np
import math
import threading
import tf2_ros
import tf 
import tf.transformations

from scipy.spatial.transform import Rotation as R

lock = threading.Lock()

class ParticleFilter:

    def __init__(self):
        ###
        # IMPORTANT: Below are parameters to tweak or to change between simulation/real robot
        self.particles_initialized = False
        self.map_acquired = False

        # Starting value. Can raise this later
        self.MAX_PARTICLES = int(rospy.get_param("~num_particles"))
        self.transform_topic = "/base_link" # for sim
        # self.transform_topic = "/base_link" # for actual car

        ###
        self.pub_tf = tf.TransformBroadcaster()


        # Get parameters
        self.particle_filter_frame = \
                rospy.get_param("~particle_filter_frame")

        # Initialize publishers/subscribers
        #
        #  *Important Note #1:* It is critical for your particle
        #     filter to obtain the following topic names from the
        #     parameters for the autograder to work correctly. Note
        #     that while the Odometry message contains both a pose and
        #     a twist component, you will only be provided with the
        #     twist component, so you should rely only on that
        #     information, and *not* use the pose component.
        scan_topic = rospy.get_param("~scan_topic", "/scan")
        odom_topic = rospy.get_param("~odom_topic", "/odom")
        self.laser_sub = rospy.Subscriber(scan_topic, LaserScan,
                                          self.lidar_callback,
                                          queue_size=1)
        self.odom_sub = rospy.Subscriber(odom_topic, Odometry,
                                          self.odom_callback,
                                          queue_size=1)

        #  *Important Note #2:* You must respond to pose
        #     initialization requests sent to the /initialpose
        #     topic. You can test that this works properly using the
        #     "Pose Estimate" feature in RViz, which publishes to
        #     /initialpose.
        self.pose_sub = rospy.Subscriber("/initialpose", PoseWithCovarianceStamped,
                                          self.initialize_particles, queue_size=1)

        #  *Important Note #3:* You must publish your pose estimate to
        #     the following topic. In particular, you must use the
        #     pose field of the Odometry message. You do not need to
        #     provide the twist part of the Odometry message. The
        #     odometry you publish here should be with respect to the
        #     "/map" frame.
        self.odom_pub = rospy.Publisher("/pf/pose/odom", Odometry, queue_size=1)
        self.particle_pub = rospy.Publisher("/pf/pose/particles", PoseArray, queue_size=1)


        #self.transform_pub = rospy.Publisher(self.transform_topic, PoseWithCovarianceStamped, queue_size=1)

        # Initialize the models
        self.motion_model = MotionModel()
        self.sensor_model = SensorModel()
        self.particles = np.zeros((self.MAX_PARTICLES, 3))
        self.proposed_particles = np.zeros((self.MAX_PARTICLES, 3))

        self.weights = np.ones(self.MAX_PARTICLES) / float(self.MAX_PARTICLES)

        self.map_sub = rospy.Subscriber("/map", OccupancyGrid, self.map_initializer_callback, queue_size=1)

        #threading.Thread(target=thread_function, args=(index,))


        # Implement the MCL algorithm
        # using the sensor model and the motion model

        # Make sure you include some way to initialize
        # your particles, ideally with some sort
        # of interactive interface in rviz
        #
        # Publish a transformation frame between the map
        # and the particle_filter_frame.

    def map_initializer_callback(self, msg):
        self.map_acquired = True

    def initialize_particles(self, msg):
        # get clicked point from rostopic /initialpose
        # generate spread of particles around clicked points
        print("Initialized particle!")
        # print(msg.header.frame_id)
        self.particles[:, 0] = msg.pose.pose.position.x + np.random.normal(loc=0.0, scale=0.5, size=self.MAX_PARTICLES)
        self.particles[:, 1] = msg.pose.pose.position.y + np.random.normal(loc=0.0, scale=0.5, size=self.MAX_PARTICLES)
        self.particles[:, 2] = self.quat_to_euler(msg.pose.pose.orientation)[-1] + np.random.normal(loc=0.0, scale=0.4,
                                                                                      size=self.MAX_PARTICLES)
        self.publish_particles()

    def euler_to_quat(self, euler):
        r = R.from_euler('xyz', euler)
        return r.as_quat()

    def quat_to_euler(self, orientation):
        quat = np.array([orientation.x, orientation.y, orientation.z, orientation.w])
        r = R.from_quat(quat)
        return r.as_euler('xyz')

    def particle_to_pose(self, particle):
        pose = Pose()
        pose.position.x = particle[0]
        pose.position.y = particle[1]
        quat_array = self.euler_to_quat([0, 0, particle[2]])
        pose.orientation = Quaternion(quat_array[0], quat_array[1], quat_array[2], quat_array[3])

        return pose

    def publish_particles(self):
        p = PoseArray()
        p.header.frame_id = "/map"
        # p.poses = np.apply_along_axis(self.particle_to_pose, 1, self.proposed_particles)
        p.poses = np.apply_along_axis(self.particle_to_pose, 1, self.particles)
        #p.poses = np.vectorize(self.particle_to_pose)(self.particles)
        #p.poses = map(self.particles, self.particle_to_pose)

        self.particle_pub.publish(p)
        self.particles_initialized = True


    def lidar_callback(self, msg):
        print("LIDAR CALLBACK --------------------------")
        # get the laser scan data and then feed the data into the sensor model evaluate function
        if hasattr(self, 'map_acquired') and not self.map_acquired or not self.particles_initialized:
            return
        observation = np.array(msg.ranges[::5])
        # rospy.loginfo(np.size(self.particles))
        # rospy.loginfo("particles")
        # rospy.loginfo(self.particles.shape)
        # rospy.loginfo("observation")
        # rospy.loginfo(observation.shape)
        self.weights = self.sensor_model.evaluate(self.particles, observation)
        self.MCL()
        self.publish_transform(msg)
        self.publish_particles()
        self.weights = np.ones(self.MAX_PARTICLES) / float(self.MAX_PARTICLES)


    def odom_callback(self, msg):
        
        print("odom callback")
        if hasattr(self, 'map_acquired') and not self.map_acquired or not self.particles_initialized:
            return

        x = msg.twist.twist.linear.x
        y = msg.twist.twist.linear.y
        theta = msg.twist.twist.angular.z
        odometry = [x, y, theta]
        
        if hasattr(self, 'prev_odom_msg'):
            self.prev_odom_arr = [x, y]
            last_time = self.prev_odom_msg.header.stamp.secs
            this_time = msg.header.stamp.secs
            dt = this_time - last_time
            if dt == 0:
                return

            odometry = odometry * dt # velocity * change in dt 
            #print(odometry)
        
        self.prev_odom_msg = msg
        self.prev_odom_arr = odometry
        
        # self.proposed_particles = self.motion_model.evaluate(self.particles, odometry)
        lock.acquire()
        self.particles = self.motion_model.evaluate(self.particles, odometry)
        self.publish_particles()
        self.publish_transform(msg)
        lock.release()

    def MCL(self):
        # using weights and proposed particles, update particles
        sample_idx = np.random.choice(range(self.MAX_PARTICLES), size=self.MAX_PARTICLES, p=self.weights/np.sum(self.weights))
        # rospy.loginfo(self.weights/np.sum(self.weights))
        # rospy.loginfo(sample_idx)
        # self.particles = self.proposed_particles[sample_idx]

        lock.acquire()
        self.particles = self.particles[sample_idx]
        lock.release()

        # self.publish_transform(msg)
        #self.publish_particles()
        

    def publish_transform(self, msg):
        # This is the previous code for transform
        '''
        transform = Transform()
        x_mean = np.mean(self.particles[:,0])
        y_mean = np.mean(self.particles[:,1])

        angular_mean = np.arctan2(np.sum(np.sin(self.particles[:,2])), np.sum(np.cos(self.particles[:,2])))
        transform.translation = [x_mean, y_mean, 0]

        quat_array = self.euler_to_quat([0, 0, angular_mean])
        transform.rotation = Quaternion(quat_array[0], quat_array[1], quat_array[2], quat_array[3])
        print(transform)
        self.br.sendTransform(transform)
        '''
        
        time = msg.header.stamp
        return_odom = Odometry()
        transform = PoseWithCovariance()

        x_mean = np.mean(self.particles[:,0])
        y_mean = np.mean(self.particles[:,1])

        angular_mean = np.arctan2(np.sum(np.sin(self.particles[:,2])), np.sum(np.cos(self.particles[:,2])))

        return_odom.header.frame_id = "/map"
        return_odom.header.stamp = time

        transform.pose.position.x = x_mean
        transform.pose.position.y = y_mean

        quat_array = self.euler_to_quat([0, 0, angular_mean])
        transform.pose.orientation = Quaternion(quat_array[0], quat_array[1], quat_array[2], quat_array[3])
        return_odom.pose = transform

        # self.transform_pub.publish(transform)
        self.odom_pub.publish(return_odom)

        self.pub_tf.sendTransform((x_mean,y_mean,0),tf.transformations.quaternion_from_euler(0, 0, -angular_mean), 
            time , "/laser", "/map")

        # map_laser_pos = np.array( (x_mean, y_mean, 0))
        # map_laser_rotation = np.array(tf.transformations.quaternion_from_euler(0, 0, angular_mean))
        # laser_base_link_offset = (0.265, 0, 0)
        # map_laser_pos -= np.dot(tf.transformations.quaternion_matrix(tf.transformations.unit_vector(map_laser_rotation))[:3, :3], laser_base_link_offset).T    

        # self.pub_tf.sendTransform(map_laser_pos, map_laser_rotation, time, self.particle_filter_frame, "/map")

if __name__ == "__main__":
    rospy.init_node("particle_filter")
    rate = rospy.Rate(50)
    pf = ParticleFilter()
    while not rospy.is_shutdown():
        pf.publish_particles()
        rate.sleep()
    
