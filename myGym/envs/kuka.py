import os, inspect
import pkg_resources
currentdir = pkg_resources.resource_filename("myGym", "envs")
repodir = pkg_resources.resource_filename("myGym", "")

import pybullet as p
import numpy as np
import copy
import math
import pybullet_data


class Kuka:

  def __init__(self, urdfRootPath=pybullet_data.getDataPath(), timeStep=0.01,
               position = [-0.100000, 0.000000, 0.070000],
               orientation = [0.000000, 0.000000, 0.000000, 1.000000]):
    self.urdfRootPath = urdfRootPath
    self.timeStep = timeStep
    self.maxVelocity = .35
    self.maxForce = 200.
    self.useInverseKinematics = 1
    self.useSimulation = 1
    self.useNullSpace = 1
    self.useOrientation = 1

    self.position = position
    self.orientation = orientation

    self.kukaEndEffectorIndex = 6
    self.kukaGripperIndex = 7
    self.motorNames = []
    self.motorIndices = []
    #lower limits for null space
    self.ll = [-.967, -2, -2.96, 0.19, -2.96, -2.09, -3.05]
    #upper limits for null space
    self.ul = [.967, 2, 2.96, 2.29, 2.96, 2.09, 3.05]
    #joint ranges for null space
    self.jr = [5.8, 4, 5.8, 4, 5.8, 4, 6]
    #restposes for null space
    self.rp = [0, 0, 0, 0.5 * math.pi, 0, -math.pi * 0.5 * 0.66, 0]
    #joint damping coefficents
    #kuka with magnetic gripper has only 11 joints
    self.jd = [
        0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001,
        0.00001, 0.00001, 0.00001, 0.00001
    ]

    self.end_effector_limits_x = [0, 0.5]
    self.end_effector_limits_y = [0.4, 1]
    self.end_effector_limits_z = [1.1, 1.2]

    objects = p.loadSDF(pkg_resources.resource_filename("myGym", '/envs/robots/kuka_magnetic_gripper_sdf/kuka_magnetic_gripper.sdf'))
    self.kukaUid = objects[0]
    p.resetBasePositionAndOrientation(self.kukaUid, self.position,
                                      self.orientation)
    self.reset()
    self.set_motors()

  def reset(self):
    #kuka with magnetic gripper has only 11 joints
    #start position of joints
    self.jointPositions = [
        0.006418, 0.413184, -0.011401, -1.589317, 0.005379, 1.137684, -0.006539, 0.000048,
        -0.299912, 0.000000, -0.000043, 0.299960, 0.000000, -0.000200
    ]
    self.numJoints = p.getNumJoints(self.kukaUid)
    for jointIndex in range(self.numJoints):
      p.resetJointState(self.kukaUid, jointIndex, self.jointPositions[jointIndex])
      #return joints to initial state if there is no external force
      p.setJointMotorControl2(self.kukaUid,
                              jointIndex,
                              p.POSITION_CONTROL,
                              targetPosition=self.jointPositions[jointIndex],
                              force=self.maxForce)
    self.endEffectorPos = np.add(list(p.getLinkState(self.kukaUid, 8)[0]), self.position) #[0.537, 0.0, 0.5]

    self.endEffectorAngle = 0

    # self.motorNames = []
    # self.motorIndices = []

  def resetup(self):
        #kuka with magnetic gripper has only 11 joints
    #start position of joints
    self.jointPositions = [
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    ]
    self.numJoints = p.getNumJoints(self.kukaUid)
    for jointIndex in range(self.numJoints):
      p.resetJointState(self.kukaUid, jointIndex, self.jointPositions[jointIndex])
      #return joints to initial state if there is no external force
      p.setJointMotorControl2(self.kukaUid,
                              jointIndex,
                              p.POSITION_CONTROL,
                              targetPosition=self.jointPositions[jointIndex],
                              force=self.maxForce)
    self.endEffectorPos = np.add(list(p.getLinkState(self.kukaUid, 8)[0]), self.position) #[0.537, 0.0, 0.5]

    self.endEffectorAngle = 0

  def resetrandom(self, positions=[]):
        #kuka with magnetic gripper has only 11 joints
    #start position of joints
    if len(positions) > 0:
      assert(len(positions) == 3),"provide [x,y,z] position or empty []"
      joints = p.calculateInverseKinematics(self.kukaUid,
                                              self.kukaEndEffectorIndex,
                                              positions,
      )
      self.jointPositions = joints
    else:
      self.jointPositions = list(np.random.uniform(self.ll, self.ul))*2

    self.numJoints = p.getNumJoints(self.kukaUid)
    for jointIndex,jointVal in enumerate(self.jointPositions):
      p.resetJointState(self.kukaUid, jointIndex, jointVal)
      #return joints to initial state if there is no external force
      p.setJointMotorControl2(self.kukaUid,
                              jointIndex,
                              p.POSITION_CONTROL,
                              targetPosition=jointVal,
                              force=self.maxForce)
    self.endEffectorPos = np.add(list(p.getLinkState(self.kukaUid, 8)[0]), self.position) #[0.537, 0.0, 0.5]

    self.endEffectorAngle = 0

  def set_motors(self):
    #find motors among all joints (fixed joints aren't motors)
    for i in range(self.numJoints):
      jointInfo = p.getJointInfo(self.kukaUid, i)
      qIndex = jointInfo[3]
      #fixed joints have qIndex -1
      if qIndex > -1:
        self.motorNames.append(str(jointInfo[1]))
        self.motorIndices.append(i)

  def getActionDimension(self):
    if (self.useInverseKinematics):
      return len(self.motorIndices) #amount of motors
    return 6  #position x,y,z and roll/pitch/yaw euler angles of end effector

  def getObservationDimension(self):
    return len(self.getObservation())

  def getObservation(self):
    #returns orientation and position of gripper (center of mass)
    observation = []
    state = p.getLinkState(self.kukaUid, self.kukaGripperIndex)
    pos = state[0]
    orn = state[1]
    euler = p.getEulerFromQuaternion(orn)

    observation.extend(list(pos))
    observation.extend(list(euler))

    return observation

  def applyAction(self, motorCommands, direct_joint_control = False):

    if (self.useInverseKinematics and not direct_joint_control):

      dx = motorCommands[0]
      dy = motorCommands[1]
      dz = motorCommands[2]

      state = p.getLinkState(self.kukaUid, self.kukaEndEffectorIndex)
      actualEndEffectorPos = state[0]

      #check position limitations of end effector
      self.endEffectorPos[0] = self.endEffectorPos[0] + dx
      if (self.endEffectorPos[0] < self.end_effector_limits_x[0]):
        self.endEffectorPos[0] = self.end_effector_limits_x[0]
      if (self.endEffectorPos[0] > self.end_effector_limits_x[1]):
        self.endEffectorPos[0] = self.end_effector_limits_x[1]

      self.endEffectorPos[1] = self.endEffectorPos[1] + dy
      if (self.endEffectorPos[1] < self.end_effector_limits_y[0]):
        self.endEffectorPos[1] = self.end_effector_limits_y[0]
      if (self.endEffectorPos[1] > self.end_effector_limits_y[1]):
        self.endEffectorPos[1] = self.end_effector_limits_y[1]

      self.endEffectorPos[2] = self.endEffectorPos[2] + dz
      if (self.endEffectorPos[2] < self.end_effector_limits_z[0]):
        self.endEffectorPos[2] = self.end_effector_limits_z[0]
      if (self.endEffectorPos[2] > self.end_effector_limits_z[1]):
        self.endEffectorPos[2] = self.end_effector_limits_z[1]

      pos = self.endEffectorPos
      orn = p.getQuaternionFromEuler([0, -math.pi, 0])  # -math.pi,yaw])
      if (self.useNullSpace):
        if (self.useOrientation):
          jointPoses = p.calculateInverseKinematics(self.kukaUid, self.kukaEndEffectorIndex, pos,
                                                    orn, self.ll, self.ul, self.jr, self.rp)
        else:
          jointPoses = p.calculateInverseKinematics(self.kukaUid,
                                                    self.kukaEndEffectorIndex,
                                                    pos,
                                                    lowerLimits=self.ll,
                                                    upperLimits=self.ul,
                                                    jointRanges=self.jr,
                                                    restPoses=self.rp)
      else:
        if (self.useOrientation):
          jointPoses = p.calculateInverseKinematics(self.kukaUid,
                                                    self.kukaEndEffectorIndex,
                                                    pos,
                                                    orn,
                                                    jointDamping=self.jd)
        else:
          jointPoses = p.calculateInverseKinematics(self.kukaUid, self.kukaEndEffectorIndex, pos)


      if (self.useSimulation):
        for i in range(self.kukaEndEffectorIndex + 1):
          p.setJointMotorControl2(bodyUniqueId=self.kukaUid,
                                  jointIndex=i,
                                  controlMode=p.POSITION_CONTROL,
                                  targetPosition=jointPoses[i],
                                  targetVelocity=0,
                                  force=self.maxForce,
                                  maxVelocity=self.maxVelocity,
                                  positionGain=0.3,
                                  velocityGain=1)
      else:
        #reset the joint state (ignoring all dynamics, not recommended to use during simulation)
        for i in range(self.numJoints):
          p.resetJointState(self.kukaUid, i, jointPoses[i])
      #gripper control
      p.setJointMotorControl2(self.kukaUid,
                              7,
                              p.POSITION_CONTROL,
                              targetPosition=self.endEffectorAngle,
                              force=self.maxForce)

    else: #forward kinematics
      for action in range(len(motorCommands)):
        motor = self.motorIndices[action]
        p.setJointMotorControl2(self.kukaUid,
                                motor,
                                p.POSITION_CONTROL,
                                targetPosition=motorCommands[action],
                                force=self.maxForce)
