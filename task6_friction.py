"""Task 6: how long can the Pantheon pendulum swing with air friction?

Linear (Stokes) drag:   x'' = -(g/L) x - (gamma/m) x'   with gamma = 6*pi*mu*R (> 0)
Quadratic drag check:   x'' = -(g/L) x - k |x'| x'      with k = rho*Cd*A/(2m)
Note: restoring term written as -g/L x (the brief's +g/L x looks like a typo).
"""
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

g, L, m, R, mu = 9.81, 67.0, 28.0, 0.085, 1.81e-5
rho, Cd = 1.2, 0.47                     # air density, drag coeff of a sphere
A0 = 3.0                                # initial amplitude [m] (as in task 5)
omega = np.sqrt(g / L)

gamma = 6 * np.pi * mu * R              # Stokes, kg/s (magnitude; brief's minus sign is the direction of the force)
tau = 2 * m / gamma                     # amplitude e-folding time, A(t) = A0 exp(-t/tau)
k = rho * Cd * np.pi * R**2 / (2 * m)

print(f"omega = {omega:.4f} rad/s, period = {2*np.pi/omega:.2f} s")
print(f"gamma = {gamma:.3e} kg/s, gamma/m = {gamma/m:.3e} 1/s")
print(f"Stokes: tau = 2m/gamma = {tau:.3e} s = {tau/86400:.1f} days, half-life = {tau*np.log(2)/86400:.1f} days")
print(f"If eq (2) is taken literally (gamma not divided by m): tau = 2/gamma = {2/gamma/3600:.1f} h")
v = omega * A0
print(f"max speed = {v:.2f} m/s, Reynolds number = {rho*v*2*R/mu:.0f}  (Stokes needs Re << 1)")
t_half_q = 3 * np.pi / (4 * k * omega * A0)
print(f"Quadratic drag: amplitude halves after {t_half_q/3600:.1f} h")

def stokes(t, s): x, v = s; return [v, -(g/L)*x - (gamma/m)*v]
def quad(t, s):   x, v = s; return [v, -(g/L)*x - k*abs(v)*v]

def run(f, T, n=20000):
    t = np.linspace(0, T, n)
    sol = solve_ivp(f, (0, T), [A0, 0], t_eval=t, rtol=1e-10, atol=1e-10, method="DOP853")
    return sol.t, sol.y[0]

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
T1 = 2 * 86400
t, x = run(stokes, T1, 200000)
# numerical decay rate from peaks
pk = (x[1:-1] > x[:-2]) & (x[1:-1] > x[2:])
tp, xp = t[1:-1][pk], x[1:-1][pk]
fit = np.polyfit(tp, np.log(xp), 1)
print(f"Numerical (2 days, Stokes): fitted tau = {-1/fit[0]/86400:.1f} days")
td = np.linspace(0, 40*86400, 400)
ax[0].plot(td/86400, A0*np.exp(-td/tau), label="Stokes: $A_0e^{-\\gamma t/2m}$")
ax[0].plot(tp/86400, xp, ".", ms=2, label="RK (DOP853) peaks, 2 days")
ax[0].axhline(A0/2, ls=":", c="gray"); ax[0].set_xlabel("time [days]"); ax[0].set_ylabel("amplitude [m]")
ax[0].set_title("Stokes drag (as in the brief)"); ax[0].legend()

T2 = 12 * 3600
t, x = run(quad, T2, 200000)
ax[1].plot(t/3600, x, lw=0.3)
th = np.linspace(0, T2, 300)
ax[1].plot(th/3600, A0/(1 + 4*k*omega*A0*th/(3*np.pi)), "r", label="envelope (averaged)")
ax[1].plot(th/3600, A0*np.exp(-th/tau), "g--", label="Stokes, for comparison")
ax[1].set_xlabel("time [hours]"); ax[1].set_ylabel("x [m]")
ax[1].set_title(f"Quadratic drag (Re ~ {rho*v*2*R/mu:.0f})"); ax[1].legend()
plt.tight_layout(); import os; plt.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), "task6_friction.png"), dpi=150)
