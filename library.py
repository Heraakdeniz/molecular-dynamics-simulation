import numpy as np
import math

def get_initial_positions(L, Npnts):
    d = L / (Npnts + 1)
    positions = []
    for i in range(1, Npnts + 1):
        for j in range(1, Npnts + 1):
            x = i * d
            y = j * d
            positions.append([x, y])
    return positions

def get_ULJ(li_pos, sigma=1.0, epsilon=1.0):
    if len(li_pos) < 2:
        return 0.0
    U_LJ = 0.0
    N = len(li_pos)
    for i in range(N):
        for j in range(i + 1, N):
            dx = li_pos[i][0] - li_pos[j][0]
            dy = li_pos[i][1] - li_pos[j][1]
            r = math.sqrt(dx**2 + dy**2)
            if r != 0:
                sr6 = (sigma / r)**6
                sr12 = sr6**2
                U_LJ += 4 * epsilon * (sr12 - sr6)
    return round(U_LJ, 4)

def update_Verlet(pos_now, pos_prev, f_now, mAr=1.0, dt=0.01):
    pos_now = np.array(pos_now)
    pos_prev = np.array(pos_prev)
    f_now = np.array(f_now)
    pos_next = 2*pos_now - pos_prev + (f_now/mAr)*dt**2
    return pos_next

def calculate_temp(v_now, kB=1.0):
    N = len(v_now)
    m_kg = 1
    v_m_s = np.array(v_now) * 1e2
    KE_total_j = 0.5 * m_kg * np.sum(v_m_s**2)
    T = KE_total_j / (N * kB)
    return float(T)

def get_velocities(pos_now, pos_prev, dt=0.01):
    return (np.array(pos_now) - np.array(pos_prev)) / dt

def rescale_vel(v_now, T_now, T_target):
    alpha = np.sqrt(T_target / T_now)
    return v_now * alpha

def apply_BC(positions, velocities, L):
    updated_positions = []
    updated_velocities = []
    for i in range(len(positions)):
        x, y = positions[i]
        vx, vy = velocities[i]
        if x < 0:
            x = -x
            vx = -vx
        elif x > L:
            x = 2*L - x
            vx = -vx
        if y < 0:
            y = -y
            vy = -vy
        elif y > L:
            y = 2*L - y
            vy = -vy
        updated_positions.append([x, y])
        updated_velocities.append([vx, vy])
    return np.array(updated_positions), np.array(updated_velocities)

def get_lj_forces(positions, sigma=1.0, epsilon=1.0):
    positions = np.array(positions)
    N = len(positions)
    forces = np.zeros_like(positions)
    for i in range(N):
        for j in range(i + 1, N):
            rij = positions[i] - positions[j]
            r = np.linalg.norm(rij)
            if r != 0:
                sr6 = (sigma / r) ** 6
                sr12 = sr6 ** 2
                force_mag = 24 * epsilon * (2 * sr12 - sr6) / r**2
                force_vec = force_mag * rij
                forces[i] += force_vec
                forces[j] -= force_vec
    return forces
