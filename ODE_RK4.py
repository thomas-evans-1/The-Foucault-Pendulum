# -*- coding: utf-8 -*-
"""
Created on Wed Aug  6 07:00:19 2025

@author: Bernard Piette
"""
import numpy as np
from math import sqrt

def runge_kutta_2nd_order_system(f, y0, t0, t_end, dt, fig_dt):
    """
    Solves a system of N second-order ODEs using 4th-order Runge-Kutta method.

    Parameters:
    - f: function f(t, y, v) → returns array of accelerations (d²y/dt²)
    - y0: initial positions (NumPy array of shape N)
    - t0: initial time
    - t_end: final time
    - dt: integration time step
    - fig_dt: number of steps between stored results

    Returns:
    - times: array of stored time points
    - y_list: array of stored position vectors
    """
    y = np.array(y0, dtype=float)
    t = t0

    times = []
    y_list = []

    next_fig_t = 1e-20;

    step = 0
    while t <= t_end:
        if t >= next_fig_t:
            next_fig_t += fig_dt
            times.append(t)
            y_list.append(y.copy())

        # RK4 coefficients
        k1 = f(t, y)
        k2 = f(t + 0.5 * dt, y + 0.5 * dt * k1)
        k3 = f(t + 0.5 * dt, y + 0.5 * dt * k2)
        k4 = f(t + dt, y + dt * k3)
        y += (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
     
        t += dt
        step += 1

    return times, y_list

def get_min_max(t_list, v_list):
    """
    Find the local minima and local maxima of lst and return them as 2 lists
    of list [t,val], one list for the minima and one for the maxima
    t_list : list of t values
    v_list : list of values (same size as t)
    Return min_list, max_list
    """
    min_list = []
    max_list = []
    but_last_v = v_list[0]
    last_v = v_list[1]

    for i in range(2,len(v_list)):
        v = v_list[i]
        if but_last_v < last_v and last_v > v:
            max_list.append([t_list[i-1], last_v])
        if but_last_v > last_v and last_v < v:
            min_list.append([t_list[i-1], last_v])
        but_last_v = last_v
        last_v = v
    return(min_list, max_list)        
 
def get_Radial_min_max(t_list, x_list, y_list):
    """
    Find the local minima and local maxima of R from x,y values in x_list
    and y_list

    t_list : list of t values
    x_list : list of x values (same size as t)
    y_list : list of x values (same size as t)
    Return list of [t,x,y,R]
    """
    min_list = []
    max_list = []
    but_last_Rs = x_list[0]**2+y_list[0]**2
    last_Rs     = x_list[1]**2+y_list[1]**2

    for i in range(2,len(t_list)):
        Rs = x_list[i]**2+y_list[i]**2
        if but_last_Rs < last_Rs and last_Rs > Rs:
            max_list.append([t_list[i-1], x_list[i-1], y_list[i-1], sqrt(last_Rs)])
        if but_last_Rs > last_Rs and last_Rs < Rs:
            min_list.append([t_list[i-1], x_list[i-1], y_list[i-1], sqrt(last_Rs)])
        but_last_Rs = last_Rs
        last_Rs = Rs
    return(min_list, max_list) 
       
def period(t_list):
    """
    Parameters
    ----------
    t_list : list of t values obtained for extrema
    Compute the avearage period of oscillation based on the 
    interval between the second and last value provided
    Returns
    -------
    Average Period of oscilations

    """
    t1 = t_list[1]
    t2 = t_list[-1]
    return (t2-t1)/(len(t_list)-2)
