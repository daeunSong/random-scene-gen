from hpp.gepetto import Viewer
from hpp.corbaserver import Client
from hpp.corbaserver.rbprm.talos_abstract import Robot
import pickle
import sys
Robot.urdfName += "_large"

packageName = 'hpp_environments'
meshPackageName = 'hpp_environments'
pbName = sys.argv[1]

### read problem file to set start/goal postiion, and the joint bounds
f = open('../pb/'+pbName,'r')
line = f.readline().split()
p_start = [float(v) for v in line]; p_start[2] += 0.98
line = f.readline().split()
p_goal = [float(v) for v in line]; p_goal[2] += 0.98
line = f.readline().split()
x_min = float(line[0]); x_max = float(line[1])
line = f.readline().split()
y_min = float(line[0]); y_max = float(line[1])
line = f.readline().split()
z_min = float(line[0]); z_max = float(line[1])

rbprmBuilder = Robot ()
# rbprmBuilder.setJointBounds ("root_joint", [-3.2,1.8,0.19,0.21, 0.95,1.7])
rbprmBuilder.setJointBounds ("root_joint", [x_min, x_max, y_min, y_max, z_min + 0.95, z_max + 1.0])
# As this scenario only consider walking, we fix the DOF of the torso :
rbprmBuilder.setJointBounds ('torso_1_joint', [0,0])
rbprmBuilder.setJointBounds ('torso_2_joint', [0.,0.])
vMax = 1.# linear velocity bound for the root
aMax = 2. # linear acceleration bound for the root
extraDof = 6
mu=0.5# coefficient of friction
rbprmBuilder.setFilter([Robot.rLegId,Robot.lLegId])

rbprmBuilder.setAffordanceFilter(Robot.rLegId, ['Support',])
rbprmBuilder.setAffordanceFilter(Robot.lLegId, ['Support'])
rbprmBuilder.boundSO3([-4.,4.,-0.1,0.1,-0.1,0.1])
# Add 6 extraDOF to the problem, used to store the linear velocity and acceleration of the root
rbprmBuilder.client.robot.setDimensionExtraConfigSpace(extraDof)
# We set the bounds of this extraDof with velocity and acceleration bounds (expect on z axis)
extraDofBounds = [-vMax,vMax,-vMax,vMax,-10.,10.,-aMax,aMax,-aMax,aMax,-10.,10.]
rbprmBuilder.client.robot.setExtraConfigSpaceBounds(extraDofBounds)
indexECS = rbprmBuilder.getConfigSize() - rbprmBuilder.client.robot.getDimensionExtraConfigSpace()

from hpp.corbaserver.problem_solver import ProblemSolver
ps = ProblemSolver( rbprmBuilder )
from hpp.gepetto import ViewerFactory
vf = ViewerFactory (ps)

from hpp.corbaserver.affordance.affordance import AffordanceTool
afftool = AffordanceTool ()
afftool.setAffordanceConfig('Support', [0.5, 0.03, 0.00005])
afftool.loadObstacleModel (packageName, "multicontact/daeun/"+pbName, "planning", vf)#,reduceSizes=[0.07,0.,0.])
v = vf.createViewer(displayArrows = True)
afftool.visualiseAffordances('Support', v, [0.25, 0.5, 0.5])
v.addLandmark(v.sceneName,1)

ps.setParameter("Kinodynamic/velocityBound",vMax)
ps.setParameter("Kinodynamic/accelerationBound",aMax)
# force the orientation of the trunk to match the direction of the motion
ps.setParameter("Kinodynamic/forceYawOrientation",True)
ps.setParameter("Kinodynamic/synchronizeVerticalAxis",True)
ps.setParameter("Kinodynamic/verticalAccelerationBound",10.)
ps.setParameter("DynamicPlanner/sizeFootX",0.2)
ps.setParameter("DynamicPlanner/sizeFootY",0.12)
ps.setParameter("DynamicPlanner/friction",mu)
# sample only configuration with null velocity and acceleration :
ps.setParameter("ConfigurationShooter/sampleExtraDOF",False)
ps.setParameter("PathOptimization/RandomShortcut/NumberOfLoops",100)
# Choosing RBPRM shooter and path validation methods.
ps.selectConfigurationShooter("RbprmShooter")
ps.addPathOptimizer ("RandomShortcutDynamic")
ps.selectPathValidation("RbprmPathValidation",0.05)
# Choosing kinodynamic methods :
ps.selectSteeringMethod("RBPRMKinodynamic")
ps.selectDistance("Kinodynamic")
ps.selectPathPlanner("DynamicPlanner")

### BEGIN climb the stairs #####
ps.setParameter("Kinodynamic/velocityBound",0.3)
ps.setParameter("Kinodynamic/accelerationBound",0.1)
q_init = rbprmBuilder.getCurrentConfig ();
q_init [0:3] = p_start 
# q_init [3:7] = [ 0., 0., 1., 0.]; v (q_init)
q_goal = q_init [::]
q_goal [0:3] = p_goal #top of stairs
q_goal[-6:-3] = [0,0,0]; v (q_goal)
# q_goal [3:7] = [ 0., 0., 1., 0.]; v (q_goal)
ps.setInitialConfig (q_init)
q_init_0 = q_init[::]
ps.addGoalConfig (q_goal)
v(q_goal)

t = ps.solve ()
print "done planning, optimize path ..."
#v.solveAndDisplay('rm',2,0.005)

for i in range(5):
  ps.optimizePath(ps.numberPaths() -1)

pathId = ps.numberPaths() -1


print "done optimizing."
from hpp.gepetto import PathPlayer
pp = PathPlayer (v)
pp.dt=0.1
pp.displayVelocityPath(pathId)
v.client.gui.setVisibility("path_"+str(pathId)+"_root","ALWAYS_ON_TOP")
pp.dt = 0.01


q_far = q_goal[::]
q_far[2] = -5
v(q_far)
q_init = q_init_0[::]

