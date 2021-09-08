#%%
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import solve_ivp

# In this code we use the gradient descent and Newton algorithm to solve numerically the problem
# ys = T(xs) knowing ys, for T describing the kinematics of a three-link robotic manipulator

#%%

L1 = 2
L2 = 3
L3 = 1

def T_function(x):
    "diffeomorphism definition"
    T0 = L1*np.sin(x[0]) + L2*np.sin(x[0]+x[1]) + L3*np.sin(x[0]+x[1]+ x[2])
    T1 = - L1*np.cos(x[0])  - L2*np.cos(x[0]+x[1]) - L3*np.cos(x[0]+x[1]+ x[2])
    T2 = x[0]+x[1]+ x[2]
    T = np.array([T0, T1, T2])
    return T


def Grad_function(x):
    "gradient definition"
    J00 = L1*np.cos(x[0]) + L2*np.cos(x[0]+x[1]) + L3*np.cos(x[0]+x[1]+ x[2])
    J01 = L2*np.cos(x[0]+x[1]) + L3*np.cos(x[0]+x[1]+ x[2])
    J02 = L3*np.cos(x[0]+x[1]+ x[2])
    J10 = L1*np.sin(x[0])  + L2*np.sin(x[0]+x[1]) + L3*np.sin(x[0]+x[1]+ x[2])
    J11 = L2*np.sin(x[0]+x[1]) + L3*np.sin(x[0]+x[1]+ x[2])
    J12 = L3*np.sin(x[0]+x[1]+ x[2])
    J20 = 1
    J21 = 1
    J22 = 1
    J = np.array([[J00, J01, J02],[J10, J11, J12],[J20, J21, J22]])
    return J


def gradient_solver(t,x):
    return -np.transpose(Grad_function(x))@(T_function(x) - ys)

def newton_solver(t,x):
    return -np.linalg.solve(Grad_function(x),T_function(x) - ys)

T_Simu = 10  # simulation horizon
dt = 0.01 
t_eval = np.arange(0, T_Simu, dt) # times at which we want the solution for plotting
xs = np.array([np.pi,np.pi/4,4]) 
x0 = np.array([np.pi*1.8,np.pi/12,-5]) # initialization
ys = T_function(xs)
y0 = T_function(x0)

#%%
##############################
## Gradient algorithm
##################################

solG = solve_ivp(gradient_solver, [0, T_Simu], x0, t_eval=t_eval)
vect_pointG = np.transpose(solG.y)

size = vect_pointG.shape
vect_TpointG = np.zeros(size)
for j in range(size[0]):
    vect_TpointG[j,:] = T_function(vect_pointG[j,:])

fig = plt.figure()
plt.plot(vect_pointG[:, 0]-xs[0], label=r'$\hat{x}_1-x_{s,1}$')
plt.plot(vect_pointG[:, 1]-xs[1], label=r'$\hat{x}_2-x_{s,2}$')
plt.plot(vect_pointG[:, 2]-xs[2], label=r'$\hat{x}_3-x_{s,3}$')
plt.legend()
plt.grid()
plt.title('Gradient descent')

fig = plt.figure()
plt.plot(np.sqrt((vect_TpointG[:, 0]-ys[0])**2 + (vect_TpointG[:, 1]-ys[1])**2 + (vect_TpointG[:, 2]-ys[2])**2 ))
plt.grid()
plt.title(r'Distance $|T(\hat{x})-y_s|$ with gradient descent')


fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot(vect_TpointG[:, 0], vect_TpointG[:, 1], vect_TpointG[:, 2], 'b')
ax.plot([y0[0]],[y0[1]],[y0[2]],'go')
ax.plot([ys[0]],[ys[1]],[ys[2]],'mo')
plt.title('Gradient trajectory in image set')


#%%
##############################
## Newton algorithm
##################################

T_Simu = 0.665
dt = 0.01
t_eval = np.arange(0, T_Simu, dt)

solN = solve_ivp(newton_solver, [0, T_Simu], x0, t_eval=t_eval)
vect_pointN = np.transpose(solN.y)

size = vect_pointN.shape
vect_TpointN = np.zeros(size)
for j in range(size[0]):
    vect_TpointN[j,:] = T_function(vect_pointN[j,:])

fig = plt.figure()
plt.plot(vect_pointN[:, 0]-xs[0], label=r'$\hat{x}_1-x_{s,1}$')
plt.plot(vect_pointN[:, 1]-xs[1], label=r'$\hat{x}_2-x_{s,2}$')
plt.plot(vect_pointN[:, 2]-xs[2], label=r'$\hat{x}_3-x_{s,3}$')
plt.grid()
plt.legend()
plt.title('Newton method')

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot(vect_TpointN[:, 0], vect_TpointN[:, 1], vect_TpointN[:, 2], 'b')
ax.plot([y0[0]],[y0[1]],[y0[2]],'go')
ax.plot([ys[0]],[ys[1]],[ys[2]],'mo')
plt.title('Newton trajectory in image set')


# %%

fig = plt.figure()
plt.plot(np.sqrt((vect_pointG[:, 0]-xs[0])**2 + (vect_pointG[:, 1]-xs[1])**2 + (vect_pointG[:, 2]-xs[2])**2 ),label="Gradient")
plt.plot(np.sqrt((vect_pointN[:, 0]-xs[0])**2 + (vect_pointN[:, 1]-xs[1])**2 + (vect_pointN[:, 2]-xs[2])**2 ),label="Newton")
plt.grid()
plt.legend()
plt.title(r'Estimation error')


# Drawing section of T(S)
# 2 circles are drawn, one of radius |L1-L2| and the other (L1+L2)
Nb_point = 100
V1 = np.zeros((Nb_point+1, 2))
V2 = np.zeros((Nb_point+1, 2))
V3 = np.zeros((Nb_point+1, 2))
V4 = np.zeros((Nb_point+1, 2))
for i in range(0, Nb_point+1):
    V1[i, :] = np.array([np.abs(L1-L2)*np.cos(i/Nb_point*2*np.pi),np.abs(L1-L2)*np.sin(i/Nb_point*2*np.pi)])
    V2[i, :] = np.array([(L1+L2)*np.cos(i/Nb_point*2*np.pi),(L1+L2)*np.sin(i/Nb_point*2*np.pi)])
    V3[i, :] = np.array([np.abs(L1-L2)*(1-i/Nb_point)+(L1+L2)*i/Nb_point,0])
    V4[i, :] = np.array([np.abs(L1-L2)*(1-i/Nb_point)+(L1+L2)*i/Nb_point,-0.000001])

fig = plt.figure()
plt.plot(V1[:, 0], V1[:, 1], 'y',label='Section of image set')
plt.plot(V2[:, 0], V2[:, 1], 'y')
plt.plot(V3[:, 0], V3[:, 1], 'y')
plt.plot(V4[:, 0], V4[:, 1], 'y')
plt.plot(vect_TpointG[:, 0]-L3*np.sin(vect_TpointG[:, 2]), vect_TpointG[:, 1]+L3*np.cos(vect_TpointG[:, 2]),  label='Gradient Standard')
plt.plot(vect_TpointN[:, 0]-L3*np.sin(vect_TpointN[:, 2]), vect_TpointN[:, 1]+L3*np.cos(vect_TpointN[:, 2]),  label='Newton Standard')
plt.plot(y0[0]-L3*np.sin(y0[2]),y0[1]+L3*np.cos(y0[2]),'go', label='Init')
plt.plot(ys[0]-L3*np.sin(ys[2]),ys[1]+L3*np.cos(ys[2]),'mo', label='Goal')
plt.legend()
plt.grid()
plt.axis('scaled')
plt.title('Image section')

plt.show()