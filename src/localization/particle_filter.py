#!/usr/bin/env python2

import rospy
from sensor_model import SensorModel
from motion_model import MotionModel

from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseWithCovarianceStamped

import numpy as np
import quaternion


class ParticleFilter:

    def __init__(self):
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
                                          lidar_callback,
                                          queue_size=1)
        self.odom_sub = rospy.Subscriber(odom_topic, Odometry,
                                          odom_callback,
                                          queue_size=1)

        #  *Important Note #2:* You must respond to pose
        #     initialization requests sent to the /initialpose
        #     topic. You can test that this works properly using the
        #     "Pose Estimate" feature in RViz, which publishes to
        #     /initialpose.
        self.pose_sub = rospy.Subscriber("/initialpose", PoseWithCovarianceStamped,
                                          initialize_particles, # TODO: Fill this in
                                          queue_size=1)

        #  *Important Note #3:* You must publish your pose estimate to
        #     the following topic. In particular, you must use the
        #     pose field of the Odometry message. You do not need to
        #     provide the twist part of the Odometry message. The
        #     odometry you publish here should be with respect to the
        #     "/map" frame.
        self.odom_pub = rospy.Publisher("/pf/pose/odom", Odometry, queue_size = 1)

        self.particle_pub = rospy.Publisher("/pf/pose/particles", PoseArray, queue_size = 1)
        
        # Initialize the models
        self.motion_model = MotionModel()
        self.sensor_model = SensorModel()

        # Implement the MCL algorithm
        # using the sensor model and the motion model
        
        # Make sure you include some way to initialize
        # your particles, ideally with some sort
        # of interactive interface in rviz
        #
        # Publish a transformation frame between the map
        # and the particle_filter_frame.
    
    def initialize_particles(self, pose):
        #TODO: Does this work?
        
        # get clicked point from rostopic /initialpose
        # generate spread of particles around clicked point

        self.weights = np.ones(self.MAX_PARTICLES) / float(self.MAX_PARTICLES)
        self.particles[:,0] = pose.position.x + np.random.normal(loc=0.0,scale=0.5,size=self.MAX_PARTICLES)
        self.particles[:,1] = pose.position.y + np.random.normal(loc=0.0,scale=0.5,size=self.MAX_PARTICLES)
        self.particles[:,2] = np.as_euler_angles(pose.orientation) + np.random.normal(loc=0.0,scale=0.4,size=self.MAX_PARTICLES)

    def particle_to_pose(particle):
        #TODO: Does this work?
        pose = Pose()
        pose.position.x = particle[0]
        pose.position.y = particle[1]
        pose.orientation = np.from_euler_angles(particle[2])
        return pose

    def publish_particles(self):
        #TODO: Does this work?
        p = PoseArray()
        p.poses = map(self.particles, particle_to_pose)
        self.particle_pub.publish(p)

    def lidar_callback(msg):
        #TODO: Fill this out
        pass

    def odom_callback(msg):
        #TODO: Fill this out
        pass 

    def MCL(self):
        #TODO: Fill this out
        raise NotImplementedError
    
    def publish_transform(self):
        #TODO: Fill this out
        raise NotImplementedError


if __name__ == "__main__":
    rospy.init_node("particle_filter")
    pf = ParticleFilter()
    rospy.spin()
