import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Diffusion coefficients 
D_square = 0.2          
D_ellipse = 0.15           
D_left = 0.05        
D_right = 0.1          
Lx, Ly = 1.0, 1.0      # domain size
Nx, Ny = 100, 100      # number of points 
Nt = 500               # Number of time steps
dx, dy = Lx / (Nx - 1), Ly / (Ny - 1)  # step size
dt = 0.0001            

T_0_square = 70.0       
T_0_ellipse = 120.0       
T_surroundings = 300.0 


T = np.ones((Nx, Ny)) * T_surroundings # Initialization of the temperature array


a, b = 0.2, 0.1       
ellipse_center_x, ellipse_center_y = Lx / 2, Ly / 2

# temperature in elliptical region
for i in range(Nx):
    for j in range(Ny):
        x = i * dx
        y = j * dy
        if ((x - ellipse_center_x)**2 / a**2 + (y - ellipse_center_y)**2 / b**2) <= 1:
            T[i, j] = T_0_ellipse

square_side = np.sqrt(0.2 * 0.1 * np.pi) # same area as ellipse
square_center_x, square_center_y = 0.75, Ly / 2

D_grid = np.ones((Nx, Ny)) 
for j in range(Ny):
    x = j * dx
    if x < Lx / 2:
        D_grid[:, j] = D_left  # [:,j] = all rows for the columns j
    else:
        D_grid[:, j] = D_right  


for i in range(Nx):
    for j in range(Ny):
        x = i * dx
        y = j * dy
        # square 
        if (np.abs(x - square_center_x) <= square_side / 2) and (np.abs(y - square_center_y) <= square_side / 2):
            D_grid[i, j] = D_square  
            T[i, j] = T_0_square
        # elliptical 
        elif ((x - ellipse_center_x)**2 / a**2 + (y - ellipse_center_y)**2 / b**2) <= 1:
            D_grid[i, j] = D_ellipse  


data = [T.copy()] # store data

for _ in range(Nt):
    T_new = T.copy()
    for i in range(1, Nx - 1): # update of T with finite difference
        for j in range(1, Ny - 1): # Local diffusion coefficient based on material
            alpha_x = D_grid[i, j] * dt / dx**2
            alpha_y = D_grid[i, j] * dt / dy**2
            T_new[i, j] = T[i, j] + alpha_x * (T[i + 1, j] - 2 * T[i, j] + T[i - 1, j]) \
                                      + alpha_y * (T[i, j + 1] - 2 * T[i, j] + T[i, j - 1])
    T = T_new.copy()
    data.append(T.copy())

fig, ax = plt.subplots()
cax = ax.imshow(data[0], extent=[0, Lx, 0, Ly], origin='lower', cmap='hot', vmin=min(T_0_square,T_0_ellipse), vmax=T_surroundings)
fig.colorbar(cax, label="Temperature (K)")

def update_2d(frame):
    cax.set_array(data[frame])
    return cax,

ani = FuncAnimation(fig, update_2d, frames=len(data), interval=50)
plt.xlabel('x')
plt.ylabel('y')
plt.title('2D Heat Diffusion with Multiple Regions')
plt.show()
