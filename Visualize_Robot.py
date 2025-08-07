import pybullet as p
import pybullet_data
import time

URDF_PATH = "simple_4dof.urdf"   # Path to your URDF

# Start GUI
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())

robot = p.loadURDF(URDF_PATH, useFixedBase=True)

# Get revolute joints
joint_ids = [i for i in range(p.getNumJoints(robot))
             if p.getJointInfo(robot, i)[2] == p.JOINT_REVOLUTE]

# Add sliders for each joint
sliders = []
for jid in joint_ids:
    info = p.getJointInfo(robot, jid)
    lo, hi = info[8], info[9]
    sliders.append(
        p.addUserDebugParameter(f"Joint {jid+1}", lo, hi, 0.0)
    )

while True:
    # Read slider values and set joints
    for jid, sid in zip(joint_ids, sliders):
        val = p.readUserDebugParameter(sid)
        p.resetJointState(robot, jid, val)
    p.stepSimulation()
    time.sleep(0.01)