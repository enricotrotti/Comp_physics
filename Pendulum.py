import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

L1, L2 = 1.0, 1.0     # lengths of the pendulums (m)
m1, m2 = 1.0, 1.0     # masses of the pendulums (kg)
g = 9.81              

def equations(y, t, L1, L2, m1, m2, g): # equations of motion
    theta1, z1, theta2, z2 = y
    cos_diff = np.cos(theta1 - theta2)
    sin_diff = np.sin(theta1 - theta2)
    
    denominator1 = (m1 + m2) * L1 - m2 * L1 * cos_diff * cos_diff #from the Lagrangian
    denominator2 = (L2 / L1) * denominator1

    theta1_dot = z1
    theta2_dot = z2
    
    z1_dot = ((m2 * g * np.sin(theta2) * cos_diff - m2 * sin_diff * (L1 * z1**2 * cos_diff + L2 * z2**2) 
               - (m1 + m2) * g * np.sin(theta1)) / denominator1)
    z2_dot = (((m1 + m2) * (L1 * z1**2 * sin_diff - g * np.sin(theta2) + g * np.sin(theta1) * cos_diff) 
               + m2 * L2 * z2**2 * sin_diff * cos_diff) / denominator2)
    
    return [theta1_dot, z1_dot, theta2_dot, z2_dot]

y0 = [np.pi / 2, 0, np.pi / 2, 0]  # initial conditions [theta1, theta1_dot, theta2, theta2_dot]

t = np.linspace(0, 20, 2000) # time array for the solution

solution = odeint(equations, y0, t, args=(L1, L2, m1, m2, g))
theta1, theta2 = solution[:, 0], solution[:, 2]

x1 = L1 * np.sin(theta1) # to carthesian coordinates
y1 = -L1 * np.cos(theta1)
x2 = x1 + L2 * np.sin(theta2)
y2 = y1 - L2 * np.cos(theta2)

fig, ax = plt.subplots(figsize=(5, 5))
ax.set(xlim=(-2, 2), ylim=(-2, 2))
line, = ax.plot([], [], 'o-', lw=2)
trail, = ax.plot([], [], 'r-', alpha=0.4, lw=1)  

def animate(i):
    line.set_data([0, x1[i], x2[i]], [0, y1[i], y2[i]])
    trail.set_data(x2[:i], y2[:i]) 
    return line, trail

ani = FuncAnimation(fig, animate, frames=len(t), interval=30, blit=True)
plt.show()
