class DummyRobot():
    
    def __init__(self):
        self.joints = { # standard position
            'l_shoulder_z':0.0,
            'l_shoulder_y':1.0,
            'l_arm_x':0.0,
            'l_elbow_y':91.0,
            'l_wrist_z':-4.0,
            'l_wrist_x':-56.0,
            'l_thumb_z':-57.0,
            'l_thumb_x':-180.0,
            'l_indexfinger_x':-180.0,
            'l_middlefingers_x':-180.0,
            'r_shoulder_z':0.0,
            'r_shoulder_y':1.0,
            'r_arm_x':0.0,
            'r_elbow_y':91.0,
            'r_wrist_z':-5.0,
            'r_wrist_x':-56.0,
            'r_thumb_z':-57.0,
            'r_thumb_x':-180.0,
            'r_indexfinger_x':-180.0,
            'r_middlefingers_x':-180.0,
            'head_z':1.0,
            'head_y':1.0
        }
        self.ranges = {
            'head_z':(-90.0,90.0),
            'head_y':(-40.0,30.0),
            'l_shoulder_z':(-30.0,80.0),
            'l_shoulder_y':(-30.0,180.0),
            'l_arm_x':(-0.0,70.0),
            'l_elbow_y':(50.0,180.0),
            'r_shoulder_z':(-30.0,80.0),
            'r_shoulder_y':(-30.0,180.0),
            'r_arm_x':(-0.0,70.0),
            'r_elbow_y':(50.0,180.0),
            'r_wrist_z':(-180.0,180.0),
            'r_wrist_x':(-180.0,180.0),
            'r_thumb_z':(-180.0,180.0),
            'r_thumb_x':(-180.0,180.0),
            'r_indexfinger_x':(-180.0,180.0),
            'r_middlefingers_x':(-180.0,180.0),
            'l_wrist_z':(-180.0,180.0),
            'l_wrist_x':(-180.0,180.0),
            'l_thumb_z':(-180.0,180.0),
            'l_thumb_x':(-180.0,180.0),
            'l_indexfinger_x':(-180.0,180.0),
            'l_middlefingers_x':(-180.0,180.0)
        }
    
    def getJointNames(self):
        return ['l_shoulder_z', 'l_shoulder_y', 'l_arm_x', 'l_elbow_y', 'l_wrist_z', 'l_wrist_x', 'l_thumb_z', 'l_thumb_x', 'l_indexfinger_x', 'l_middlefingers_x', 'r_shoulder_z', 'r_shoulder_y', 'r_arm_x', 'r_elbow_y', 'r_wrist_z', 'r_wrist_x', 'r_thumb_z', 'r_thumb_x', 'r_indexfinger_x', 'r_middlefingers_x', 'head_z', 'head_y']
        
    def getAngleLowerLimit(self,k):
        return self.ranges[k][0]
    
    def getAngleUpperLimit(self,k):
        return self.ranges[k][1]
    
    def getAngle(self,k):
        return self.joints[k]

    def setAngle(self,k,position,speed):
        self.joints[k] = position
        
    def enableTorque(self,k):
        pass
    
    def disableTorque(self,k):
        pass
    
    def getPalmSensorReading(self,k):
        return 10.0
