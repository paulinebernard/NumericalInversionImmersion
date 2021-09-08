#%%
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
from scipy.integrate import solve_ivp

# In this code we use the gradient descent algorithm to solve numerically the problem
# ys = T(xs) knowing ys
# where T is a non convex mapping, diffeomorphism on R^2, not surjective

k=10 # k is a parameter on the diffeo. large k, implies strong "non" convexity
def T_function(x):
    "diffeo definition"
    T0 = 2*np.arctan(x[0])
    T1 = 2*np.arctan(x[1]) + k*np.sin(2*np.arctan(x[0]))
    T = np.array([T0, T1])
    return T

def Grad_function(x):
    "Gradient definition"
    J00 = 2/(1+x[0]**2)
    J01 = 0
    J10 = 2*k*np.cos(2*np.arctan(x[0]))/(1+x[0]**2)
    J11 = 2/(1+x[1]**2)
    J = np.array([[J00, J01],[J10,J11]])
    return J


#%%
################################ 
# Diffeomorphism image extension
################################
# T(R^2) = E1 \cap E2 where
# E1={ y=(y1,y2), |y1| < pi } and E2={y=(y1,y2), |y2-ksin(y1)| < pi }
# So we build Phi2 : E2 -> E1 and Phi1 : E1 -> R^2   knowing that Phi2 acts only
# on y2 and Phi1 only on y1. 
# And we consider Phi = Phi1 o Phi2 : E1 \cap E2 -> R^2
# Phi is a diffeomorphism from E1 \cap E2 to R^2

epsilon = 0.2
def Phi(y):
    "Diffeomorphism Phi : y in E_1 cap E_2"
    y1 = y[0]
    y2 = y[1]
    lim2 = y2-k*np.sin(y1)
    # Phi2
    if np.abs(lim2)<= np.pi*np.exp(-epsilon):
        y_Phi2 = y
    else:
        t2 = np.log(np.abs(lim2)/np.pi)   # between -epsilon and 0
        nu2 = (epsilon+t2)**2/t2  # between -\infty and 0
        y_Phi2 = np.array([y1,np.exp(-nu2)*lim2+k*np.sin(y1)])
    # Phi1
    if np.abs(y1)<= np.pi*np.exp(-epsilon):
        y_Phi21 = y_Phi2
    else:
        t1 = np.log(np.abs(y1)/np.pi)  # between -epsilon and 0
        nu1 = (epsilon+t1)**2/t1  # between -\infty and 0
        y_Phi21 = np.array([y1*np.exp(-nu1),y_Phi2[1]])
    return y_Phi21

def grad_Phi(y):
    "Jacobian of Phi : y in E_1 inter E_2"
    y1 = y[0]
    y2 = y[1]
    lim2 = y2-k*np.sin(y1)
    dlim2 = [-k*np.cos(y1),1]
    # Phi2
    if np.abs(lim2)<= np.pi*np.exp(-epsilon):
        #y_Phi2 = y
        jac2 = np.eye(2)
    else:
        t2 = np.log(np.abs(lim2)/np.pi)   # between -epsilon and 0
        dt2 = dlim2/lim2
        nu2 = (epsilon+t2)**2/t2  # between -\infty and 0
        dnu2 = 1-epsilon**2/t2**2 
        #y_Phi2 = np.array([y1,np.exp(-nu2)*lim2+k*np.sin(y1)])
        jac2 = np.array([[1,0],[np.exp(-nu2)*(-dnu2*dt2[0]*lim2+dlim2[0])+k*np.cos(y1),np.exp(-nu2)*(-dnu2*dt2[1]*lim2+dlim2[1])]])
    # Phi1
    if np.abs(y1)<= np.pi*np.exp(-epsilon):
        #y_Phi21 = y1
        jac1 = np.eye(2)
    else:
        t1 = np.log(np.abs(y1)/np.pi)  # between -epsilon and 0
        dt1 = 1/y1
        nu1 = (epsilon+t1)**2/t1  # between -\infty and 0
        dnu1 = 1-epsilon**2/t1**2 
        #y_Phi21 = np.array([y1*np.exp(-nu1),y_Phi2[1]])
        jac1 = np.array([[(1-dnu1*dt1*y1)*np.exp(-nu1),0],[0,1]])
    return jac1@jac2


def phi(ybar):
    "Inverse of Phi : ybar in R^2"
    if np.abs(ybar[0])<= np.pi*np.exp(-epsilon):
        ybar1 = ybar
    else:
        t1 = np.log(np.abs(ybar[0])/np.pi)  # between -epsilon and +\infty
        tau1 = epsilon**2/(2*epsilon+t1)  # between 0 and epsilon
        ybar1 = np.array([ybar[0]*np.exp(-(t1+tau1)),ybar[1]])
    lim2 = ybar1[1]-k*np.sin(ybar1[0])
    if np.abs(lim2)<= np.pi*np.exp(-epsilon):
        ybar2 = ybar1
    else:
        t2 = np.log(np.abs(lim2)/np.pi)  # between -epsilon and +\infty
        tau2 = epsilon**2/(2*epsilon+t2)  # between 0 and epsilon
        ybar2 = np.array([ybar1[0],lim2*np.exp(-(t2+tau2))+k*np.sin(ybar1[0])])
    return ybar2

def grad_phi(ybar):
    "Jacobian of phi, inverse of Phi : ybar in R^2"
    if np.abs(ybar[0])<= np.pi*np.exp(-epsilon):
        ybar1 = ybar
        jac1 = np.eye(2)
    else:
        t1 = np.log(np.abs(ybar[0])/np.pi)  # between -epsilon and +\infty
        dt1 = 1/ybar[0]
        tau1 = epsilon**2/(2*epsilon+t1)  # between 0 and epsilon
        dtau1 = -epsilon**2/(2*epsilon+t1)**2
        ybar1 = np.array([ybar[0]*np.exp(-(t1+tau1)),ybar[1]])
        jac1 = np.array([[np.exp(-(t1+tau1))*(1-(1+dtau1)*ybar[0]*dt1),0],[0,1]])
    lim2 = ybar1[1]-k*np.sin(ybar1[0])
    dlim2 = [-k*np.cos(ybar1[0]),1]
    if np.abs(lim2)<= np.pi*np.exp(-epsilon):
        #ybar2 = ybar1
        jac2 = np.eye(2)
    else:
        t2 = np.log(np.abs(lim2)/np.pi)  # between -epsilon and +\infty
        dt2 = dlim2/lim2
        tau2 = epsilon**2/(2*epsilon+t2)  # between 0 and epsilon
        dtau2 = -epsilon**2/(2*epsilon+t2)**2
        #ybar2 = np.array([ybar1[0],lim2*np.exp(-(t2+tau2))+k*np.sin(ybar1[0])])
        jac2 = np.array([[1,0],[(dlim2[0]-lim2*(1+dtau2)*dt2[0])*np.exp(-(t2+tau2))+k*np.cos(ybar1[0]),(dlim2[1]-lim2*(1+dtau2)*dt2[1])*np.exp(-(t2+tau2))]])
    return jac2@jac1




#%%
###########################
# Inversion algorithms
##########################

P = np.array([[1,0],[0,1]]) # to modify metric for gradient descent
def gradient_solver(t,x):
    return -np.transpose(Grad_function(x)).dot(P).dot(T_function(x) - ys)

def gradient_solver_ext(t,x):
    yhat = T_function(x)
    return -np.transpose(grad_Phi(yhat)@Grad_function(x))@(Phi(yhat) - ys)

def newton_solver(t,x):
    return -np.linalg.solve(Grad_function(x),T_function(x) - ys)

def newton_solver_ext(t,x):
    yhatbar = Phi(T_function(x))
    return -np.linalg.solve(Grad_function(x),grad_phi(yhatbar)@(yhatbar - ys))

#%%

T_Simu = 100   # increasing the time horizon enables to achieve convergence but longer simulation
dt = 0.01
t_eval = np.arange(0, T_Simu, dt)
x0 = np.array([2, 1])
#xs = np.array([1, 0])  # target which works with a standard gradient descent/newton
xs = np.array([-1, 1]) # target which requires diffeomorphism extension
ys = T_function(xs)
y0 = T_function(x0)

#%%
##############################
## Gradient algorithm
##################################

solG = solve_ivp(gradient_solver, [0, T_Simu], x0, t_eval=t_eval)
vect_pointG = np.transpose(solG.y)

solG_ext = solve_ivp(gradient_solver_ext, [0, T_Simu], x0, t_eval=t_eval)
vect_pointG_ext = np.transpose(solG_ext.y)

size = vect_pointG.shape
vect_TpointG = np.zeros(size)
for j in range(size[0]):
    vect_TpointG[j,:] = T_function(vect_pointG[j,:])

size_ext = vect_pointG_ext.shape
vect_TpointG_ext = np.zeros(size_ext)
vect_TextpointG_ext = np.zeros(size_ext)
for j in range(size_ext[0]):
    vect_TpointG_ext[j,:] = T_function(vect_pointG_ext[j,:])
    vect_TextpointG_ext[j,:] = Phi(T_function(vect_pointG_ext[j,:]))

fig = plt.figure()
plt.plot(vect_pointG_ext[:, 0]-xs[0],'b', label=r'Extended')
plt.plot(vect_pointG_ext[:, 1]-xs[1],'b')
plt.plot(vect_pointG[:, 0]-xs[0], 'r', label=r'Standard')
plt.plot(vect_pointG[:, 1]-xs[1], 'r')
plt.legend()
plt.grid()
plt.title('Estimation error gradient descent')

fig = plt.figure()
plt.plot(np.sqrt((vect_TextpointG_ext[:, 0]-ys[0])**2 + (vect_TextpointG_ext[:, 1]-ys[1])**2 ),'b',label='Ext')
plt.plot(np.sqrt((vect_TpointG[:, 0]-ys[0])**2 + (vect_TpointG[:, 1]-ys[1])**2 ),'r',label='Normal')
plt.title(r'Distance $|T(\hat{x})-y_s|$ with gradient descent')
plt.legend()
plt.grid()

# Drawing T(S)
# 4 curves are drawn. Each one corresponds to the image of an edge of the square (of size pi) by the diffeo :
# (x1,x2)  -> (x1,x2+k sin (x1))
Nb_point = 100
V1 = np.zeros((Nb_point+1, 2))
V2 = np.zeros((Nb_point+1, 2))
V3 = np.zeros((Nb_point+1, 2))
V4 = np.zeros((Nb_point+1, 2))
for i in range(0, Nb_point+1):
    V1[i, :] = np.array([np.pi,-np.pi + 2 * i / Nb_point * np.pi])
    V2[i, :] = np.array([-np.pi + 2 * i / Nb_point * np.pi,np.pi + k* np.sin(-np.pi + 2 * i / Nb_point * np.pi)])
    V3[i, :] = np.array([-np.pi,-np.pi + 2 * i / Nb_point * np.pi])
    V4[i, :] = np.array([-np.pi + 2 * i / Nb_point * np.pi,-np.pi + k*np.sin(-np.pi + 2 * i / Nb_point * np.pi)])


fig = plt.figure()
plt.plot(V1[:, 0], V1[:, 1], 'y',label='Image set')
plt.plot(V2[:, 0], V2[:, 1], 'y')
plt.plot(V3[:, 0], V3[:, 1], 'y')
plt.plot(V4[:, 0], V4[:, 1], 'y')
plt.plot(vect_TextpointG_ext[:, 0], vect_TextpointG_ext[:, 1],'b', label=r'$T_e(\hat{x})$ Ext')
plt.plot(vect_TpointG_ext[:, 0], vect_TpointG_ext[:, 1], 'b--', label=r'$T(\hat{x})$ Ext')
plt.plot(vect_TpointG[:, 0], vect_TpointG[:, 1], 'r', label=r'$T(\hat{x})$ Standard')
plt.plot(y0[0],y0[1],'go', label='Init')
plt.plot(ys[0],ys[1],'mo', label='Goal')
plt.legend()
plt.grid()
plt.title('Trajectory with gradient method in image set')


#%%
##############################
## Newton algorithm
##################################

T_Simu = 800
dt = 0.01
t_eval = np.arange(0, T_Simu, dt)

solN = solve_ivp(newton_solver, [0, T_Simu], x0, t_eval=t_eval)
vect_pointN = np.transpose(solN.y)

solN_ext = solve_ivp(newton_solver_ext, [0, T_Simu], x0, t_eval=t_eval)
vect_pointN_ext = np.transpose(solN_ext.y)

size = vect_pointN.shape
vect_TpointN = np.zeros(size)
for j in range(size[0]):
    vect_TpointN[j,:] = T_function(vect_pointN[j,:])

size_ext = vect_pointN_ext.shape
vect_TpointN_ext = np.zeros(size_ext)
vect_TextpointN_ext = np.zeros(size_ext)
for j in range(size_ext[0]):
    vect_TpointN_ext[j,:] = T_function(vect_pointN_ext[j,:])
    vect_TextpointN_ext[j,:] = Phi(T_function(vect_pointN_ext[j,:]))


fig = plt.figure()
plt.plot(vect_pointN_ext[:, 0]-xs[0],'b', label='Extended')
plt.plot(vect_pointN_ext[:, 1]-xs[1],'b')
plt.plot(vect_pointN[:, 0]-xs[0], 'r', label='Standard')
plt.plot(vect_pointN[:, 1]-xs[1],'r')
plt.grid()
plt.legend()
plt.title('Estimation error with Newton method')

fig = plt.figure()
plt.plot(V1[:, 0], V1[:, 1], 'y',label='Image set')
plt.plot(V2[:, 0], V2[:, 1], 'y')
plt.plot(V3[:, 0], V3[:, 1], 'y')
plt.plot(V4[:, 0], V4[:, 1], 'y')
plt.plot(vect_TextpointN_ext[:, 0], vect_TextpointN_ext[:, 1], 'b', label=r'$T_e(\hat{x})$ Ext')
plt.plot(vect_TpointN_ext[:, 0], vect_TpointN_ext[:, 1],'b--', label=r'$T(\hat{x})$ Ext')
plt.plot(vect_TpointN[:, 0], vect_TpointN[:, 1], 'r', label=r'$T(\hat{x})$ Standard')
plt.plot(y0[0],y0[1],'go', label='Init')
plt.plot(ys[0],ys[1],'mo', label='Goal')
plt.grid()
plt.legend()
plt.title('Trajectory with Newton method in image set')

#%%
# Global plots

fig = plt.figure()
plt.plot(np.sqrt((vect_TextpointG_ext[:, 0]-ys[0])**2 + (vect_TextpointG_ext[:, 1]-ys[1])**2 ),'b',label='Gradient Ext')
plt.plot(np.sqrt((vect_TpointG[:, 0]-ys[0])**2 + (vect_TpointG[:, 1]-ys[1])**2 ),'m',label='Gradient Standard')
plt.plot(np.sqrt((vect_TextpointN_ext[:, 0]-ys[0])**2 + (vect_TextpointN_ext[:, 1]-ys[1])**2 ),'k',label='Newton Ext')
plt.plot(np.sqrt((vect_TpointN[:, 0]-ys[0])**2 + (vect_TpointN[:, 1]-ys[1])**2 ),'r',label='Newton Standard')
plt.title(r'Estimation error $|\hat{x}-x_s|$')
plt.legend()
plt.grid()

plt.show()

# %%
