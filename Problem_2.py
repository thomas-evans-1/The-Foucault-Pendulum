import numpy as np
import matplotlib.pyplot as plt
import ODE_RK4
from pathlib import Path

# Directory for generated figures
FIG_DIR = Path(__file__).resolve().parent / "figures"
FIG_DIR.mkdir(exist_ok=True)


# ==================================================
# 1. Physical parameters
# ==================================================

g = 9.81
L = 67
m = 28
gamma = 0

goL = g / L
omega0 = np.sqrt(goL)

T_analytical = 2 * np.pi / omega0


# ==================================================
# 2. Numerical parameters
# ==================================================

dt = 0.001
fig_dt = 0.01
t0 = 0
t_end = 100


# ==================================================
# 3. Equations of motion
# ==================================================

def F(t, V):

    x, vx, y, vy = V

    return np.array([
        vx,
        -goL * x - (gamma / m) * vx**2,
        vy,
        -goL * y - (gamma / m) * vy**2
    ])


# ==================================================
# 4. Analytical solution (gamma = 0)
# ==================================================

def analytical_solution(t, V0):

    x0, vx0, y0, vy0 = V0

    x = (
        x0 * np.cos(omega0 * t)
        + (vx0 / omega0) * np.sin(omega0 * t)
    )

    y = (
        y0 * np.cos(omega0 * t)
        + (vy0 / omega0) * np.sin(omega0 * t)
    )

    return x, y


# ==================================================
# 5. Run an individual experiment
# ==================================================

def run_experiment(name, V0):

    # Solve the system numerically
    t_list, V_list = ODE_RK4.runge_kutta_2nd_order_system(
        F, V0, t0, t_end, dt, fig_dt
    )

    # Convert results into NumPy arrays
    t = np.array(t_list)
    V = np.array(V_list)

    # Extract positions and velocities
    x = V[:, 0]
    vx = V[:, 1]
    y = V[:, 2]
    vy = V[:, 3]

    # Calculate analytical solution
    x_exact, y_exact = analytical_solution(t, V0)

    # Calculate numerical errors
    x_error = np.abs(x - x_exact)
    y_error = np.abs(y - y_exact)

    position_error = np.sqrt(
        (x - x_exact)**2 + (y - y_exact)**2
    )

    # Calculate numerical period
    min_list, max_list = ODE_RK4.get_min_max(t, x)

    t_list_min = [val[0] for val in min_list]

    T_numerical = ODE_RK4.period(t_list_min)

    # Calculate period errors
    T_error = abs(T_numerical - T_analytical)

    T_relative_error = (
        T_error / T_analytical * 100
    )

    # Store results
    return {
        "name": name,
        "V0": V0,
        "t": t,
        "x": x,
        "y": y,
        "vx": vx,
        "vy": vy,
        "x_exact": x_exact,
        "y_exact": y_exact,
        "x_error": x_error,
        "y_error": y_error,
        "position_error": position_error,
        "T_numerical": T_numerical,
        "T_error": T_error,
        "T_relative_error": T_relative_error
    }


# ==================================================
# 6. Define and run experiments
# ==================================================

if __name__ == "__main__":

    experiments = [
        ("Linear", [1, 0, 0, 0]),
        ("Diagonal", [1, 0, 1, 0]),
        ("Elliptical", [1, 0, 0, 0.1]),
        ("Circular", [1, 0, 0, omega0])
    ]

    results = []

    for name, V0 in experiments:

        result = run_experiment(name, V0)

        results.append(result)


    # ==============================================
    # 7. Print numerical results
    # ==============================================

    print(f"Analytical period: {T_analytical:.8f} s")

    for result in results:

        print("\n" + "=" * 50)
        print(result["name"])
        print("=" * 50)

        print(f"Initial conditions: {result['V0']}")

        print(
            f"Numerical period: "
            f"{result['T_numerical']:.8f} s"
        )

        print(
            f"Absolute period error: "
            f"{result['T_error']:.8e} s"
        )

        print(
            f"Percentage period error: "
            f"{result['T_relative_error']:.8f}%"
        )

        print(
            f"Maximum x error: "
            f"{np.max(result['x_error']):.8e} m"
        )

        print(
            f"Maximum y error: "
            f"{np.max(result['y_error']):.8e} m"
        )

        print(
            f"Maximum position error: "
            f"{np.max(result['position_error']):.8e} m"
        )

        print(
            f"Mean position error: "
            f"{np.mean(result['position_error']):.8e} m"
        )


    # ==============================================
    # 8. Plot horizontal trajectories
    # ==============================================

    fig, axes = plt.subplots(
        2, 2, figsize=(8, 8),
        sharex=True, sharey=True
    )

    for i, (ax, result) in enumerate(zip(axes.flat, results)):
        ax.plot(
            result["x"],
            result["y"],
            linewidth=2,
            label="Numerical"
        )

        t_plot = np.linspace(0, T_analytical, 500)

        x_plot, y_plot = analytical_solution(
            t_plot, result["V0"]
        )

        ax.plot(
            x_plot,
            y_plot,
            linestyle="--",
            linewidth=1.5,
            label="Analytical"
        )

        ax.scatter(
            result["V0"][0],
            result["V0"][2],
            color="black",
            s=20,
            label="Initial position",
            zorder=3
        )

        ax.set_title(f"({chr(97 + i)}) {result['name']}")
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")

        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)
        ax.set_aspect("equal")

        ax.grid(alpha=0.3)

    handles, labels = axes[0, 0].get_legend_handles_labels()

    fig.legend(
        handles, labels,
        loc="lower center",
        ncol=3
    )

    fig.tight_layout(rect=[0, 0.05, 1, 1])

    fig.savefig(
        FIG_DIR / "problem_2_trajectories.pdf",
        bbox_inches="tight"
    )

    plt.show()


    # ==============================================
    # 9. Plot displacement against time
    # ==============================================

    fig, axes = plt.subplots(
        2, 2, figsize=(10, 6),
        sharex=True, sharey=True
    )

    for ax, result in zip(axes.flat, results):

        ax.plot(
            result["t"],
            result["x"],
            label="x(t)"
        )

        ax.plot(
            result["t"],
            result["y"],
            linestyle="--",
            label="y(t)"
        )

        ax.set_title(result["name"])
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Displacement (m)")
        ax.grid(alpha=0.3)
        ax.legend()

    fig.tight_layout()

    fig.savefig(
        FIG_DIR / "problem_2_displacements.pdf",
        bbox_inches="tight"
    )

    plt.show()


    # ==============================================
    # 10. Plot numerical position errors
    # ==============================================

    fig, ax = plt.subplots(figsize=(8, 4))

    for result in results:

        ax.plot(
            result["t"],
            result["position_error"],
            label=result["name"]
        )

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Absolute position error (m)")
    ax.grid(alpha=0.3)
    ax.legend()

    fig.tight_layout()

    fig.savefig(
        FIG_DIR / "problem_2_errors.pdf",
        bbox_inches="tight"
    )

    plt.show()