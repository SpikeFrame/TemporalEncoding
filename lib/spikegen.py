#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  1 17:10:48 2016

@author: Brian

Routines for generating spike trains
"""

from __future__ import division
import numpy as np

from pyNN.parameters import Sequence
from pyNN.random import NumpyRNG


def unif(n_spikes, T, rng=NumpyRNG(), rounding=False, t_min=1.0, min_isi=10.0):
    """
    Generate uniformally distributed spikes between [t_min, T),
    with minimum inter-spike separation and optional rounding

    pyNN.nest is generally unstable with rounding for input spikes
    pyNN.nest errors if lowest spike value is exactly equal to dt

    Input spikes between 0.0 and dt are not integrated over

    Parameters
    ----------
    n_spikes : int
    T : float
        Time interval (ms)
    t_min : float
        Lower bound on generated time value (ms)
    min_isi : float
        Minimum inter-spike separation : n_spikes*MIN_ISI << T (default 10 ms)

    Returns
    -------
    spike_times : pyNN.parameters.Sequence (float64)

    """

    spike_times = np.empty([0], dtype=float)
    while spike_times.size < n_spikes:
        timing = rng.uniform(t_min, T)

        # Ensure minimum separation w.r.t. existing spikes
        if (spike_times.size > 0 and
                np.min(np.abs(timing - spike_times)) < min_isi):
            continue
        else:
            spike_times = np.append(spike_times, timing)

    spike_times.sort()

    if rounding:
        return Sequence(np.floor(spike_times))
    else:
        return Sequence(spike_times)


def poisson(rate, T, rng=NumpyRNG(), rounding=False, t_min=1.0,
            min_isi=10.0):
    """
    Poisson distributed spikes between [t_min, T),
    with minimum inter-spike separation and optional rounding

    Parameters
    ----------
    rate : firing rate (Hz)
    T : float
        Time interval (ms)
    t_min : float
        Lower bound on generated time value (ms)
    min_isi : float
        Minimum inter-spike separation (ms)

    Returns
    -------
    spike_times : pyNN.parameters.Sequence (float64)

    """

    spike_times = t_min + rng.exponential(1000.0 / rate, 1)
    while spike_times[-1] < T - min_isi:
        timing = spike_times[-1] + (min_isi + rng.exponential(1000.0 / rate))

        spike_times = np.append(spike_times, timing)

    spike_times = spike_times[spike_times < T]

    if rounding:
        return Sequence(np.floor(spike_times))
    else:
        return Sequence(spike_times)
