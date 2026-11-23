import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.signal import find_peaks

g = 9.81 # Gravitational Acceleration 
m = 28.0 # Pendulum Mass (kg)
L = 67.0 # Pendulum Length (m)
Omega = 2 * np.pi /86400 # Earth's Angular Frequency (s)
latitude = 48 # Pendulum Latitidue
sigma = np.radians(90-latitude) # Pendulum Co-latitude
RE = 6.371e6 # Earth's Radius (m)
gamma = 0.0 # Drag Coefficient

# Returns a system of first order ODEs for Sovler
def foucault_ODE_system(t, u):

    # Intial Conditions
    x = u[0]
    dxdt = u[1]
    y = u[2]
    dydt = u[3]

    # Velocity Differential Equations
    dx_dt = dxdt
    dy_dt = dydt

    # z Terms
    z = - np.sqrt(L**2 - x**2 - y**2)
    dzdt = - (x * dxdt + y * dydt) / z

    # Equation Terms
    Eq_x = 2 * Omega * np.cos(sigma) * dy_dt + Omega**2 * RE * np.cos(sigma) * np.sin(sigma) - gamma / m * abs(dx_dt) * dx_dt
    Eq_y = -2 * Omega * np.sin(sigma) * dzdt - 2 * Omega * np.cos(sigma) * dx_dt - gamma / m * abs(dy_dt) * dy_dt
    Eq_z = 2 * Omega * np.sin(sigma) * dy_dt - g - gamma / m * abs(dzdt) * dzdt

    # Accerleration Differential Equations
    dxx_dtt = 1 / (L**2 + x**2 + y**2) * (Eq_x * (L**2 + y**2 - x**2) - 2 * Eq_y * x * y - Eq_z * x * z + x / z**2 * ((dx_dt * y - dy_dt * x)**2 - L**2 * (dx_dt**2 + dy_dt**2)))
    dyy_dtt = 1 / (L**2 + x**2 + y**2) * (Eq_y * (L**2 + x**2 - y**2) - 2 * Eq_x * x * y - Eq_z * y * z + y / z**2 * ((dx_dt * y - dy_dt * x)**2 - L**2 * (dx_dt**2 + dy_dt**2)))
    
    return [dx_dt, dxx_dtt, dy_dt, dyy_dtt]

# Initial Conditions - Small Occilations (3m)
u0 = [3.0, 0, 0, 0]

# Time 
t_max = 6 * 3600 # 6 hours
t_eval = np.linspace(0, t_max, 200000) 

# Solution
solution = solve_ivp(foucault_ODE_system, [0, t_max], u0, method="RK45", t_eval=t_eval, rtol=1e-9, atol=1e-11)

t = solution.t
x = solution.y[0]
y = solution.y[2]

# Direction of occisaltion at each truning point
r = np.sqrt(x**2 + y**2)
peaks, _ = find_peaks(r, distance=30)
angles = np.arctan2(y[peaks], x[peaks])
angles = np.unwrap(2 * angles) / 2
precession = angles - angles[0]
t_peaks = t[peaks]

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(x[::20], y[::20], c=t[::20]/3600, s=0.5)
plt.colorbar(label="Time (hours)")
plt.xlabel("x (m)")
plt.ylabel("y (m)")
plt.title("Bob trajectory")
plt.axis("equal")

plt.subplot(1, 2, 2)
plt.plot(t_peaks/3600, np.degrees(precession))
plt.xlabel("Time (hours)")
plt.ylabel("Precession angle (degrees)")
plt.title("Foucault Pendulum Precession")
plt.grid()

plt.tight_layout()
plt.show()


