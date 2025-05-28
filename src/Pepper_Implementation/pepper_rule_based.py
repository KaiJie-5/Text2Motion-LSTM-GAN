#! /usr/bin/env python
# -*- encoding: UTF-8 -*-

"""Example: Use say Method"""

import qi
import argparse
import sys
import time
import math
from naoqi import ALProxy

#A man is dancing in the ballet.
#An old man is singing on stage

def home(robot_ip, robot_port, fractionMaxSpeed_home=0.2, time_home=0.3):
    motionProxy = ALProxy("ALMotion", robot_ip, robot_port)
    time.sleep(time_home)
    degree = math.pi/180
    HeadYaw=0*degree
    HeadPitch=0*degree
    HipRoll=-0.1*degree
    HipPitch=-2.1*degree
    LShoulderPitch=89.6*degree
    LShoulderRoll=8.1*degree
    LElbowYaw=-70.9*degree
    LElbowRoll=-6.3*degree
    LWristYaw=0*degree
    LHand = 0.25
    RShoulderPitch=100.1*degree
    RShoulderRoll=-5.8*degree
    RElbowYaw=96.5*degree
    RElbowRoll=5.4*degree
    RWristYaw=0*degree
    RHand = 0.25
    KneePitch = 0*degree

    joints = ['HeadYaw', 'HeadPitch', 'HipRoll', 'HipPitch', 'KneePitch', 
        'LShoulderPitch', 'LShoulderRoll', 'LElbowYaw', 'LElbowRoll', 'LWristYaw', 'LHand',
        'RShoulderPitch','RShoulderRoll','RElbowYaw','RElbowRoll','RWristYaw','RHand']
    sent_data =  (HeadYaw, HeadPitch, HipRoll, HipPitch, KneePitch,
                LShoulderPitch, LShoulderRoll, LElbowYaw, LElbowRoll, LWristYaw, LHand,
                RShoulderPitch, RShoulderRoll, RElbowYaw, RElbowRoll, RWristYaw, RHand)
    
    # Fixed motionProxy reference
    motionProxy.setAngles(joints, sent_data, fractionMaxSpeed_home)
    return 1


def main(session):
    """
    Say a text with a local configuration.
    """
    # Get the service ALAnimatedSpeech.

    asr_service = session.service("ALAnimatedSpeech")

    # set the local configuration
    configuration = {"bodyLanguageMode":"contextual"}

    # say the text with the local configuration
    asr_service.say("A man is dancing in the ballet.", configuration)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", type=str, default="10.22.9.13",
                        help="Robot IP address. On robot or Local Naoqi: use '127.0.0.1'.")
    parser.add_argument("--port", type=int, default=9559,
                        help="Naoqi port number")

    args = parser.parse_args()
    session = qi.Session()
    try:
        session.connect("tcp://" + args.ip + ":" + str(args.port))
    except RuntimeError:
        print ("Can't connect to Naoqi at ip \"" + args.ip + "\" on port " + str(args.port) +".\n"
               "Please check your script arguments. Run with -h option for help.")
        sys.exit(1)
    main(session)
    home(args.ip, args.port)