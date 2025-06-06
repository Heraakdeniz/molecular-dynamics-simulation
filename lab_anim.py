import library
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML

# Physical and simulation constants
sigma = 1.0
epsilon = 1.0
mAr = 1.0
dt = 0.01
L = 10.0
kB = 1.0

library.sigma = sigma
library.epsilon = epsilon
library.mAr = mAr
library.dt = dt
library.L = L
library.kB = kB

N = 3        # Particles per side (total: N*N)
steps = 200  # Number of simulation steps

# Initialize grid positions and small random velocities
positions = np.array(library.get_initial_positions(L, N))
velocities = np.random.randn(N*N, 2) * 0.01
positions_prev = positions - velocities * dt

all_positions = []

for step in range(steps):
    forces = library.get_lj_forces(positions)
    positions_next = library.update_Verlet(positions, positions_prev, forces)
    v_now = library.get_velocities(positions, positions_prev)
    positions_next, v_now = library.apply_BC(positions_next, v_now, L)
    all_positions.append(positions_next.copy())
    positions_prev = positions
    positions = positions_next

all_positions = np.array(all_positions)

fig, ax = plt.subplots(figsize=(5, 5))
scat = ax.scatter([], [], s=100, color="blue")
ax.set_xlim(0, L)
ax.set_ylim(0, L)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("Particle Simulation")
ax.grid(True)
ax.set_aspect('equal', adjustable='box')

def animate(frame):
    data = all_positions[frame]
    scat.set_offsets(data)
    ax.set_title(f"Step {frame+1} / {steps}")
    return scat,

ani = FuncAnimation(fig, animate, frames=steps, interval=50, blit=True, repeat=False)

# Save as GIF (no ffmpeg needed)
ani.save("particle_animation.gif", writer="pillow")

# If you want to display in a Jupyter Notebook cell:
# HTML(ani.to_jshtml())

# If ffmpeg is available and you want an mp4:
# ani.save("particle_animation.mp4", writer="ffmpeg")
