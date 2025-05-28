from os import listdir, remove
from os.path import isfile, join
import gzip
import gensim
import numpy as np
from util.one_euro_filter import OneEuroFilter
from util.geo import *
import math
import re

def normalize_text(text):
    # remove special characters\whitespaces
    text = [re.sub(r"[^a-zA-Z0-9]+", ' ', k) for k in text.split("\n")]

    return text

def load_w2v(w2v_path):
    w2v_model = gensim.models.KeyedVectors.load_word2vec_format(w2v_path, binary=True)
    return w2v_model

def load_metadata(metadata_path):
    npzfile = np.load(metadata_path)

    train_action = npzfile['arr_0']
    train_script = npzfile['arr_1']
    train_length = npzfile['arr_2']
    sentence_steps = int(npzfile['arr_3'])    
    
    return train_action, train_script, train_length, sentence_steps  

# Input: (24, time_seq)
# Output: (9, 3, time_seq)
def construct_joint3D(plot_vec):
    plot_vec = np.reshape(plot_vec, [8, 3, -1])

    mean_len = [0.6, 0.7, 0.9, 0.9, 0.7, 0.9, 0.9]

    plot_pose = np.zeros(plot_vec.shape)
    plot_pose[1, :, :] = plot_vec[0, :, :];
    plot_pose[0, :, :] = plot_pose[1, :, :]+\
                         mean_len[0] * np.divide(plot_vec[1, :, :], 
                                                 np.tile(np.linalg.norm(plot_vec[1, :, :], axis=0), (3, 1)))
    plot_pose[2, :, :] = plot_pose[1, :, :]+\
                         mean_len[1] * np.divide(plot_vec[2, :, :], 
                                                 np.tile(np.linalg.norm(plot_vec[2, :, :], axis=0), (3, 1)))
    plot_pose[3, :, :] = plot_pose[2, :, :]+\
                         mean_len[2] * np.divide(plot_vec[3, :, :], 
                                                 np.tile(np.linalg.norm(plot_vec[3, :, :], axis=0), (3, 1)))
    plot_pose[4, :, :] = plot_pose[3, :, :]+\
                         mean_len[3] * np.divide(plot_vec[4, :, :], 
                                                 np.tile(np.linalg.norm(plot_vec[4, :, :], axis=0), (3, 1)))
    plot_pose[5, :, :] = plot_pose[1, :, :]+\
                         mean_len[4] * np.divide(plot_vec[5, :, :], 
                                                 np.tile(np.linalg.norm(plot_vec[5, :, :], axis=0), (3, 1)))
    plot_pose[6, :, :] = plot_pose[5, :, :]+\
                         mean_len[5] * np.divide(plot_vec[6, :, :], 
                                                 np.tile(np.linalg.norm(plot_vec[6, :, :], axis=0), (3, 1)))
    plot_pose[7, :, :] = plot_pose[6, :, :]+\
                         mean_len[6] * np.divide(plot_vec[7, :, :], 
                                                 np.tile(np.linalg.norm(plot_vec[7, :, :], axis=0), (3, 1)))
    
    # plot the virtal central_hip, left hip, right hip
    v2h = np.array([plot_pose[2, 0, :],plot_pose[2, 1, :], plot_pose[2, 2, :]-1.5])
    v5h = np.array([plot_pose[5, 0, :],plot_pose[5, 1, :], plot_pose[5, 2, :]-1.5])
    
    pelvis = (v2h+v5h)/2
    pelvis = np.expand_dims(pelvis, axis=0)
    plot_pose = np.concatenate([plot_pose, pelvis])
    return plot_pose

# Input: [dims, time_seq]
# Output: [dims, time_seq]
def oneEFilter(motion_data, beta = 0.04, min_cutoff = 0.5):
    t = np.arange(1,np.shape(motion_data)[1]+1)
    x_hat = np.zeros_like(motion_data)
    for joint in range(np.shape(motion_data)[0]):
        x_hat[joint, 0] = motion_data[joint ,0]
        one_euro_filter = OneEuroFilter(
            t[0],  motion_data[joint, 0],
            min_cutoff=min_cutoff,
            beta=beta
        )
        for i in range(1, len(t)):
            x_hat[joint, i] = one_euro_filter(t[i], motion_data[joint, i])
            
    return x_hat

# Input (9, 3, time_seq)
# Output: Robot joint angles
def Processing_data(data, scale_f=1.0):
    data = np.array(data).astype("float")
    data = data*scale_f
    for rows in range(np.shape(data)[2]):
        HT = data[0, 0:3, rows] # 0
        NO = None
        NE = data[1, 0:3, rows] # 1
        PE = data[8, 0:3, rows] # 8
        LS = data[5, 0:3, rows] # 5
        LE = data[6, 0:3, rows] # 6
        LW = data[7, 0:3, rows] # 7
        RS = data[2, 0:3, rows] # 2
        RE = data[3, 0:3, rows] # 3
        RW = data[4,0:3, rows] # 4
        jas = xyz_to_angles(Point(HT), None, Point(NE), Point(PE), 
                            Point(LS), Point(LE), Point(LW), 
                            Point(RS), Point(RE), Point(RW))
        if rows==0:
            Pepper_data = jas
        else:
            Pepper_row_data = jas
            Pepper_data = np.vstack((Pepper_data,Pepper_row_data))

    return Pepper_data

def xyz_to_angles(HT, NO, NE, PE, LS, LE, LW, RS, RE, RW):
    '''
    Joint positions (x,y,z) as Point objects in the ordering:
        HeadTop, Nose, Neck, Pelvis, 
        LShoulder, LElbow, LWrist, 
        RShoulder, RElbow, RWrist
        
    Auxiliary geometric entities --------------------- can complete the list below
        p_... = plane
        l_... = line
        P_... = point
        
    Points:
        P_PHT
        P_PNO
        
    Lines:
        l_NE_PE
        l_NE_HT
        l_NE_ChestNormal
        l_NE_PHT
        l_iHead
    
    Planes:
        p_Chest
        p_NeckH
        p_NeckF
        p_NeckS
        p_Head
        
    '''
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
    ####################################################################
    # HeadPitch (up/down); positive = down
    
    p_Chest = Plane(PE, LS, RS) # facing plane of chest, normal forward (from person to camera)
    
    l_NE_PE = Line(NE, PE)
    l_NE_HT = Line(NE, HT)
    p_NeckH = Plane(NE, l_NE_PE) # horizontal plane passing thru NE, normal downwards to PE
    
    l_NE_ChestNormal = p_Chest.normal().projected_on(p_NeckH) # chest normal projected on plane p_NeckH
    p_NeckF = Plane(NE, l_NE_ChestNormal) # facing plane passing thru neck, perpendicular to p_NeckH, normal forward, not collinear with chestNormal
    
    HeadPitch = l_NE_PE.angle_to(l_NE_HT) * p_NeckF.orientation(HT) # positive if bow forward
    
    ####################################################################
    # HeadYaw (left/right); positive = left
    
    p_NeckS = Plane(NE, Point(p_NeckH.normal().r2), Point(p_NeckF.normal().r2)) # side plane, perpendicular to p_NeckH and p_NeckF, normal to right shoulder
    
    if NO == None: # Nose joint is not available, neglect roll rotation of head
#         P_PHT = HT.projected_on(p_NeckH) # project onto horizontal neck plane
#         l_NE_PHT = Line(NE, P_PHT)
        
#         if p_NeckF.orientation(P_PHT) > 0: # is in front
#             HeadYaw = - l_NE_PHT.angle_to(p_NeckS) * p_NeckS.orientation(P_PHT) # positive towards left shoulder
#         else:
#             HeadYaw = - (np.pi - l_NE_PHT.angle_to(p_NeckS)) * p_NeckS.orientation(P_PHT)
        HeadYaw=0*degree

    else: # Nose is available, can calculate more accurately, w/o neglecting roll rotation of head
        p_Head = Plane(NE, NO, HT)
        l_iHead = p_Head.intersection(p_NeckH) # line from intersection of head plane and neck horizontal plane
        P_PNO = NO.projected_on(l_iHead) # nose projected on this line
        
        if p_NeckF.orientation(P_PNO) > 0: # is in front
            HeadYaw = - l_iHead.angle_to(p_NeckS) * p_NeckS.orientation(P_PNO) # positive towards left shoulder
        else:
            HeadYaw = - (np.pi - l_iHead.angle_to(p_NeckS)) * p_NeckS.orientation(P_PNO)
    ####################################################################
    # LShoulderRoll (left/right); positive = left, away from body

    l_LS_RS = Line(LS, RS) # connects shoulders
    P_PPE = PE.projected_on(l_LS_RS) # projection of Pelvis onto shoulders connector
    l_PPE_PE = Line(P_PPE, PE)
    p_ShouldersH = Plane(P_PPE, l_PPE_PE) # horizontal plane passign thru shoulders, perpendicular to p_Chest, normal downwards
    
    p_LShoulder = Plane(LS, l_LS_RS) # vertical side plane, thru LS, perpendicular to p_ShouldersH, normal to RS
    l_LS_LE = Line(LS, LE)
    
    if p_ShouldersH.orientation(LE) > 0: # elbow below shoulder
        LShoulderRoll = - p_LShoulder.angle_to(l_LS_LE) * p_LShoulder.orientation(LE)
    else:
        LShoulderRoll = np.pi + p_LShoulder.angle_to(l_LS_LE) * p_LShoulder.orientation(LE)
        
    ####################################################################
    # LShoulderPitch (up/down); positive = down, neutral position is hands forward
    
    if p_Chest.orientation(LE) > 0: # elbow in front of the chest plane
        LShoulderPitch = p_ShouldersH.angle_to(l_LS_LE) * p_ShouldersH.orientation(LE)
    else:
        LShoulderPitch = (np.pi - p_ShouldersH.angle_to(l_LS_LE)) * p_ShouldersH.orientation(LE)
    
    ####################################################################
    # LElbowRoll negative only
    
    l_LE_LS = Line(LE, LS)
    l_LE_LW = Line(LE, LW)
    p_LElbowP = Plane(LE, l_LE_LS) # plane perpendicular to limb elbow-shoulder, passing thru LE, normal to LS
    
    if p_LElbowP.orientation(LW) > 0: # acute angle LW-LE-LS
        LElbowRoll = - p_LElbowP.angle_to(l_LE_LW) - (np.pi/2)
    else:
        LElbowRoll = p_LElbowP.angle_to(l_LE_LW) - (np.pi/2)

    ####################################################################
    # LElbowYaw positive = rotate to body center
    
    P_LECHN = Point(LE.r + p_Chest.n) # endpoint of chest normal moved to LE joint
    p_LElbowN = Plane(LE, LS, P_LECHN) # plane perpendicular to p_Chest, passign thru LS, LE, normal away from body center

    P_LEENN = Point(LE.r + p_LElbowN.n) # endpoint of p_LElbowN's normal, equivalent to Point(p_LElbowN.normal().r2)
    p_LElbow = Plane(LE, LS, P_LEENN) # plane perpendicular to p_LElbowN, containing limb LE-LS, normal backwards - away from camera

    p_LArm = Plane(LE, LW, LS) # plane of all 3 left-hand joints, normal towards RS if elbow bent and LElbowYaw=-pi/2
    
    if p_LElbowN.orientation(LW) > 0: # is on the further side - away from body center
        LElbowYaw = (np.pi - p_LElbow.angle_to(p_LArm)) * p_LElbow.orientation(LW)
    else:
        LElbowYaw = p_LElbow.angle_to(p_LArm) * p_LElbow.orientation(LW)
            
    
    ####################################################################
    # RShoulderRoll (left/right); negative = right, away from body
    
    l_RS_LS = Line(RS, LS)
    p_RShoulder = Plane(RS, l_RS_LS) # vertical side plane, thru RS, perpendicular to p_ShouldersH, normal to LS
    l_RS_RE = Line(RS, RE)
    
    if p_ShouldersH.orientation(RE) > 0: # elbow below shoulder
        RShoulderRoll = p_RShoulder.angle_to(l_RS_RE) * p_RShoulder.orientation(RE)
    else:
        RShoulderRoll = - np.pi - p_RShoulder.angle_to(l_RS_RE) * p_RShoulder.orientation(RE)
        
    ####################################################################
    # RShoulderPitch (up/down); positive = down
    
    if p_Chest.orientation(RE) > 0: # elbow in front of the chest plane
        RShoulderPitch = p_ShouldersH.angle_to(l_RS_RE) * p_ShouldersH.orientation(RE)
    else:
        RShoulderPitch = (np.pi - p_ShouldersH.angle_to(l_RS_RE)) * p_ShouldersH.orientation(RE)
    
    ####################################################################
    # RElbowRoll positive only
    
    l_RE_RS = Line(RE, RS)
    l_RE_RW = Line(RE, RW)
    p_RElbowP = Plane(RE, l_RE_RS) # plane perpendicular to limb elbow-shoulder, passing thru RE, normal to RS
    
    if p_RElbowP.orientation(RW) > 0: # 
        RElbowRoll = (np.pi/2) + p_RElbowP.angle_to(l_RE_RW) 
    else:
        RElbowRoll = (np.pi/2) - p_RElbowP.angle_to(l_RE_RW)
    
    ####################################################################
    # RElbowYaw positive = rotate away from body center
    
    P_RECHN = Point(RE.r + p_Chest.n) # endpoint of chest normal moved to RE joint
    p_RElbowN = Plane(RE, RS, P_RECHN) # plane perpendicular to p_Chest, passign thru RS, RE, normal towards body center
    
    P_REENN = Point(RE.r + p_RElbowN.n) # endpoint of p_RElbowN's normal, equivalent to Point(p_RElbowN.normal().r2)
    p_RElbow = Plane(RE, RS, P_REENN) # plane perpendicular to p_RElbowN, containing limb RE-RS, normal backwards - away from camera
    
    p_RArm = Plane(RE, RS, RW) # plane of all 3 left-hand joints, normal towards LS if elbow bent and RElbowYaw=pi/2
    
    if p_RElbowN.orientation(RW) > 0: # is on the closer side - closer to body center
        RElbowYaw = - p_RElbow.angle_to(p_RArm) * p_RElbow.orientation(RW)
    else:
        RElbowYaw = - (np.pi - p_RElbow.angle_to(p_RArm)) * p_RElbow.orientation(RW)
        
    ####################################################################
    # HipRoll positive = rotate to left
   
    # Alternative
    P_SM = LS.midpoint_to(RS) # midpoint between shoulders
    l_SM_PE = Line(P_SM, PE)
    p_ShouldersS = Plane(P_SM, l_LS_RS) # side plane perpendicular to shoulders connector, normal towards RS
    
    HipRoll = p_ShouldersS.angle_to(l_SM_PE) * p_ShouldersS.orientation(PE)
    
    ####################################################################
    ### Avoiding abnormal hand gesture ###
    HipPitch=-2.1*degree
    HeadPitch = HeadPitch - math.pi/18

    if (RElbowYaw < -0.25):
        RElbowYaw = -0.205
    if (LElbowYaw > 0.25):
        LElbowYaw = 0.25
    if (RShoulderPitch>100*degree):
        RShoulderPitch=100*degree
    elif (RShoulderPitch<-80*degree):
        RShoulderPitch=-80*degree
        
    if (LShoulderPitch>100*degree):
        LShoulderPitch=100*degree
    elif LShoulderPitch<-80*degree:
        LShoulderPitch=-80*degree
    
    ### Avoiding abnormal hand gesture ###
    ### Safety hip movements ###
    if HipRoll > math.pi/18:
        HipRoll = math.pi/18
    elif HipRoll < -math.pi/18:
        HipRoll = -math.pi/18
    ### Safety hip movements ###
    return np.array([
        HeadYaw, HeadPitch, HipRoll, HipPitch, 0,
        LShoulderPitch, LShoulderRoll, LElbowYaw, LElbowRoll, LWristYaw, LHand,
        RShoulderPitch, RShoulderRoll, RElbowYaw, RElbowRoll, RWristYaw, RHand
    ])

