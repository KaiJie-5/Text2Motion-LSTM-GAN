# subscriber_py2.py (Python 2)
import socket
import json
import math
import time
import numpy as np
import threading
from scipy.signal import lfilter
from naoqi import ALProxy
# 10.22.9.13
# 127.0.0.1
class Subscriber:
    def __init__(self, host='localhost', port=5000, robot_ip="10.22.9.13", robot_port=9559):
        self.host = host
        self.port = port
        self.robot_ip = robot_ip
        self.robot_port = robot_port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        
        # Initialize NAOqi proxies
        try:
            self.motionProxy = ALProxy("ALMotion", self.robot_ip, self.robot_port)
            self.Posture = ALProxy("ALRobotPosture", self.robot_ip, self.robot_port)
            self.tts = ALProxy("ALTextToSpeech", self.robot_ip, self.robot_port)
            self.tts.setParameter('speed' , 80)
            print("[PY2] Connected to Pepper")
        except Exception as e:
            print("[PY2] Failed to connect to Pepper: %s" % str(e))
            raise

    def connect(self):
        try:
            self.sock.settimeout(1.0)
            self.sock.connect((self.host, self.port))
            print("[PY2] Connected to publisher")
            return True
        except socket.error as e:
            print("[PY2] Connection failed: %s" % str(e))
            return False

    def receive_loop(self):
        buffer = ""
        try:
            while True:
                try:
                    data = self.sock.recv(1024)
                    if not data:
                        break  # Connection closed
                    buffer += data
                    
                    # Process complete JSON messages
                    while True:
                        try:
                            msg, idx = json.JSONDecoder().raw_decode(buffer)
                            buffer = buffer[idx:].lstrip()
                            self._handle_message(msg)
                            # time.sleep(0.5)

                            # print("home")
                        except ValueError:
                            break  # Incomplete data
                except socket.timeout:
                    continue
                except socket.error:
                    break
        except KeyboardInterrupt:
            print("\n[PY2] Closing connections...")
            self.sock.close()
            self.motionProxy.exit()  # Cleanup NAOqi proxies
            self.Posture.exit()
        self.sock.close()

    def _handle_message(self, msg):
        print("[PY2] Received data for processing")
        if "data" in msg:
            # Validate data shape
            if not isinstance(msg["data"], (list, np.ndarray)) or len(np.shape(msg["data"])) != 3:
                print("[PY2] Invalid data format. Expected 3D array")
                return
            pepper_data = self.Processing_data(msg["data"])

            motion_thread = threading.Thread(target=self.sendToPepper, args=(pepper_data,))
            motion_thread.start()
            print("[PY2] Motion started in thread")

        if "text" in msg:
            t = threading.Thread(target=self.say_text, args= (str(msg["text"]),))
            t.start()
        else:
            print("[PY2] Message missing 'data' field")

    def say_text(self, text):
        try:
            self.tts.say(text)
        except Exception as e:
            print("[PY2] Speech failed: %s" % str(e))

    def Processing_data(self, data, scale_f=1.0):
        data = np.array(data).astype("float")
        data = data*scale_f
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
        
        for rows in range(np.shape(data)[2]):
            # Fixed method calls with self.
            left_hand = data[7, 0:3, rows]
            left_hand = self.RotationZ(math.pi)*left_hand.reshape((-1,1))
            left_hand = np.array(left_hand).flatten()
            
            right_hand = data[4,0:3, rows]
            right_hand = self.RotationZ(math.pi)*right_hand.reshape((-1,1))
            right_hand = np.array(right_hand).flatten()
            
            # ... (similar fixes for all RotationZ and NormalizeVector calls below)
            left_shoulder = data[5, 0:3, rows]
            left_shoulder = self.RotationZ(math.pi)*left_shoulder.reshape((-1,1))
            left_shoulder = np.array(left_shoulder).flatten()
            
            right_shoulder = data[2, 0:3, rows]
            right_shoulder = self.RotationZ(math.pi)*right_shoulder.reshape((-1,1))
            right_shoulder = np.array(right_shoulder).flatten()
            
            left_elbow = data[6, 0:3, rows]
            left_elbow = self.RotationZ(math.pi)*left_elbow.reshape((-1,1))
            left_elbow = np.array(left_elbow).flatten()
            
            right_elbow = data[3, 0:3, rows]
            right_elbow = self.RotationZ(math.pi)*right_elbow.reshape((-1,1))
            right_elbow = np.array(right_elbow).flatten()
            
            neck = data[1, 0:3, rows]
            neck = self.RotationZ(math.pi)*neck.reshape((-1,1))
            neck = np.array(neck).flatten()

            head = data[0, 0:3, rows]
            head = self.RotationZ(math.pi)*head.reshape((-1,1))
            head = np.array(head).flatten()

            # Ref axes calculation
            left_right_hip = np.subtract(right_shoulder,left_shoulder)
            torsor_right_hip = np.subtract(right_shoulder,neck)
            
            z_ref = np.cross(left_right_hip,torsor_right_hip)
            z_ref = z_ref/np.linalg.norm(z_ref)
            
            x_ref = left_right_hip/np.linalg.norm(left_right_hip)
            
            y_ref = np.cross(z_ref,x_ref)

            x_ref = self.NormalizeVector(x_ref)
            y_ref = self.NormalizeVector(y_ref)
            z_ref = self.NormalizeVector(z_ref)
            
            left_shoulder_elbow = np.subtract(left_elbow,left_shoulder)
            left_shoulder_neck = np.subtract(right_shoulder,left_shoulder)
            left_shoulder_elbow = self.NormalizeVector(left_shoulder_elbow)
            left_shoulder_neck = self.NormalizeVector(left_shoulder_neck)
            left_shoulder_angle_roll = 0
            left_shoulder_angle_roll = math.acos(np.clip(np.dot(left_shoulder_elbow,left_shoulder_neck), -1, 1))
            LShoulderRoll = left_shoulder_angle_roll - math.pi/2

            left_elbow_hand = np.subtract(left_hand,left_elbow)
            left_elbow_shoulder = np.subtract(left_shoulder,left_elbow)
            left_elbow_hand = self.NormalizeVector(left_elbow_hand)
            left_elbow_shoulder = self.NormalizeVector(left_elbow_shoulder)
            left_shoulder_angle_pitch = 0
            
            left_shoulder_angle_pitch = math.acos(np.clip(np.dot(y_ref,left_elbow_shoulder), -1, 1))
            LShoulderPitch = left_shoulder_angle_pitch - math.pi/2 
            
            left_elbow_angle_roll = 0
            left_elbow_angle_roll = math.acos(np.clip(np.dot(left_elbow_shoulder,left_elbow_hand), -1, 1))
            left_elbow_angle_roll = left_elbow_angle_roll - math.pi
            LElbowRoll = left_elbow_angle_roll + 0.35 #######################
            
            left_elbow_angle_yaw = 0
            left_elbow_angle_yaw = (left_elbow_hand[2]/ math.sin(left_elbow_angle_roll))*math.pi/2 #+ math.pi/18
            if left_elbow_angle_yaw > 1.0 :
                left_elbow_angle_yaw = -1.5
            if left_elbow_angle_yaw < -1.5 :
                left_elbow_angle_yaw = -1.5
            
            #if (left_hand[1] > left_elbow[1]):
            #    left_elbow_angle_yaw = 0
            LElbowYaw = left_elbow_angle_yaw 

            right_shoulder_elbow = np.subtract(right_elbow,right_shoulder)
            right_shoulder_neck = np.subtract(left_shoulder,right_shoulder)
            right_shoulder_elbow = self.NormalizeVector(right_shoulder_elbow)
            right_shoulder_neck = self.NormalizeVector(right_shoulder_neck)
            right_shoulder_angle_roll =0
            right_shoulder_angle_roll = math.acos(np.clip(np.dot(right_shoulder_elbow,right_shoulder_neck), -1, 1))
            right_shoulder_angle_roll = -(right_shoulder_angle_roll - math.pi/2)
            RShoulderRoll = right_shoulder_angle_roll

            right_elbow_hand = np.subtract(right_hand,right_elbow)
            right_elbow_shoulder = np.subtract(right_shoulder,right_elbow)
            right_elbow_hand = self.NormalizeVector(right_elbow_hand)
            right_elbow_shoulder = self.NormalizeVector(right_elbow_shoulder)
            right_shoulder_angle_pitch = 0
            right_shoulder_angle_pitch = math.acos(np.clip(np.dot(y_ref,right_elbow_shoulder), -1, 1))
            RShoulderPitch = right_shoulder_angle_pitch - math.pi/2 
            
            right_elbow_angle_roll = 0
            right_elbow_angle_roll = math.acos(np.clip(np.dot(right_elbow_hand,right_elbow_shoulder), -1, 1))
            right_elbow_angle_roll = -(right_elbow_angle_roll - math.pi)
            RElbowRoll = right_elbow_angle_roll - 0.35 #######################

            right_elbow_angle_yaw = 0
            right_elbow_angle_yaw = (right_elbow_hand[2] / math.sin(right_elbow_angle_roll))*math.pi/2 #- math.pi/18
            if (right_elbow_angle_yaw > 1.5):
                right_elbow_angle_yaw = 1.5
            if (right_elbow_angle_yaw < -1.5):
                right_elbow_angle_yaw = 1.5
            
            #if (right_hand[1] > right_elbow[1]):
            #    right_elbow_angle_yaw = 0
            RElbowYaw = right_elbow_angle_yaw 

            Hip_Roll = math.acos(np.clip(np.dot(right_shoulder_neck,[0,0,0.6]), -1, 1)) 
            HipRoll = (Hip_Roll - math.pi/2)
            if (HipRoll > 5*math.pi/36):
                HipRoll = 5*math.pi/36
            if (HipRoll < -5*math.pi/36):
                HipRoll = -5*math.pi/36

            if rows==0:
                Pepper_data = np.array([HeadYaw, HeadPitch, HipRoll, HipPitch, 0,
                            LShoulderPitch, LShoulderRoll, LElbowYaw, LElbowRoll, LWristYaw, LHand,
                            RShoulderPitch, RShoulderRoll, RElbowYaw, RElbowRoll, RWristYaw, RHand])
            else:
                Pepper_row_data = np.hstack((HeadYaw, HeadPitch, HipRoll, HipPitch, 0,
                            LShoulderPitch, LShoulderRoll, LElbowYaw, LElbowRoll, LWristYaw, LHand,
                            RShoulderPitch, RShoulderRoll, RElbowYaw, RElbowRoll, RWristYaw, RHand))
                Pepper_data = np.vstack((Pepper_data,Pepper_row_data))
            
        return Pepper_data
        
    def RotationZ(self, theta):
        R_z = np.array([[math.cos(theta), -math.sin(theta), 0],
                       [math.sin(theta), math.cos(theta), 0],
                       [0, 0, 1]])
        return np.mat(R_z)

    def NormalizeVector(self, vector):
        norm = np.linalg.norm(vector)
        return vector / norm if norm != 0 else vector

    def sendToPepper(self, data, filter_after=4, fractionMaxSpeed=0.2, time_delay=0.1):
        joints = ['HeadYaw', 'HeadPitch', 'HipRoll', 'HipPitch', 'KneePitch',
                  'LShoulderPitch', 'LShoulderRoll', 'LElbowYaw', 'LElbowRoll', 'LWristYaw', 'LHand',
                  'RShoulderPitch', 'RShoulderRoll', 'RElbowYaw', 'RElbowRoll', 'RWristYaw', 'RHand']
        
        # Handle single frame data
        if len(data.shape) == 1:
            data = data.reshape(1, -1)

        if filter_after:
            n = filter_after
            b = [1.0 / n] * n
            a = 1
            for i in range(data.shape[1]-1):
                data[:, i] = lfilter(b, a, data[:, i])

        for row in data:
            try:
                self.motionProxy.setAngles(joints, row.tolist(), fractionMaxSpeed)
                time.sleep(time_delay)
            except Exception as e:
                print("[PY2] Failed to send command: %s" % str(e))

        try:
            self.home()
            print("[PY2] Returned to home position")
        except Exception as e:
            print("[PY2] Failed to return home: %s" %str(e))

    def home(self, fractionMaxSpeed_home=0.2, time_home=0.3):
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
        self.motionProxy.setAngles(joints, sent_data, fractionMaxSpeed_home)
        return 1

if __name__ == "__main__":
    subscriber = Subscriber()
    try:
        if subscriber.connect():
            subscriber.receive_loop()
    except KeyboardInterrupt:
        print("\n[PY2] Terminated by user.")
    finally:
        subscriber.sock.close()