import numpy as np
from numba import jit

#@jit(nopython=False)
def advection_diffusion(dt, tra, tra_bot, Grid, Advec, Diff):
    """
    Compute tracer advection and diffusion using an explicit solver.

    Parameters:
        dt (float): Time step (seconds).
        tra (np.ndarray): Tracer concentration at two time steps [current, previous].
        tra_top (float): Top boundary condition for tracer.
        tra_bot (float): Bottom boundary condition for tracer.
        dz (float): Grid spacing (meters).
        w_up (np.ndarray): Vertical velocity (m/s).
        Kv (np.ndarray): Vertical diffusivity (m²/s).

    Returns:
        np.ndarray: Updated tracer concentrations after advection and diffusion.
    """
    
    # Compute numerical coefficients
    alpha = Advec.w[:-1] * dt / (2 * Grid.dz)
    beta = -dt / (2 * Grid.dz) * (Advec.w[:-1] - Advec.w[1:])
    gamma = Diff.Kv[:-1] * dt / (Grid.dz**2)
    delta = dt / (4 * Grid.dz) * (Diff.Kv[:-1] - Diff.Kv[1:])

    # Integration coefficients
    coeff1 = 1 + beta - 2 * gamma
    coeff2 = alpha + gamma - delta
    coeff3 = -alpha + gamma + delta

    # Apply boundary conditions
    tra[-1, 1] = tra_bot  # Bottom boundary

    # Update tracer concentrations
    tra[1:-1, 1] = (
        tra[1:-1, 1] * coeff1[:-1] +
        tra[2:,   1] * coeff2[:-1] +
        tra[:-2,  1] * coeff3[:-1]
    )

    return tra

#@jit(nopython=False)
def mix_mld(dt, tra, mld, tmld, Grid):
    """
    Perform mixing within the mixed layer.

    Parameters:
        dt (float): Time step (seconds).
        tra (np.ndarray): Tracer concentration at two time steps [current, previous].
        mld (float): Mixed layer depth (meters).
        tmld (float): Timescale for mixing in the mixed layer (1/s).

    Returns:
        np.ndarray: Updated tracer concentrations after mixed layer mixing.
    """
    # Identify mixed layer indices
    mld_bool = -Grid.zgrid < mld

    # Compute average tracer concentration in the mixed layer
    tra_mld_avg = np.mean(tra[mld_bool, 1])

    # Mix tracer in the mixed layer
    tra[mld_bool,1] = tra[mld_bool,1] + (tra_mld_avg - tra[mld_bool,1]) * tmld * dt

    return tra


#@jit(nopython=False)
def restore(dt, tra, itra, trest, Grid):
    """
    Perform restoring in deeper layers.

    Parameters:
        dt (float): Time step (seconds).
        tra (np.ndarray): Tracer concentration at two time steps [current, previous].
        tra (np.ndarray): Initial tracer concentrations to restore towards.
        trest (float): Timescale for restoring (1/s).

    Returns:
        np.ndarray: Updated tracer concentrations after mixed layer mixing.
    """
    # Identify mixed layer indices
    deep_bool = -Grid.zgrid > 300.0

    # Restore tracer values
    tra[deep_bool,1] = tra[deep_bool,1] + (itra[deep_bool] - tra[deep_bool,1]) * trest * dt

    return tra