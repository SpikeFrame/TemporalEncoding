#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  2 15:20:41 2016

@author: brian

Error metrics of (dis)similarity between spike trains
"""

from __future__ import division
import numpy as np

from pyNN.parameters import Sequence


def van_rossum(spike_times_out, spike_times_ref, tau_c=10.0):
    """
    Exactly computes the van Rossum distance between two spike trains

    Parameters
    ----------
    spike_times_out : sequence / array / list / value
    spike_times_ref : sequence / array / list / value
    tau_c : float
        van Rossum time constant (default 10 ms)

    Returns
    -------
    distance : numpy.float64
        Distance between spike trains

    """

    def convert_value(spike_times):
        if isinstance(spike_times, Sequence):
            spike_times = spike_times.value
        elif hasattr(spike_times, '__len__'):
            spike_times = np.asarray(spike_times)
        else:
            spike_times = np.asarray([spike_times])
        return spike_times

    try:
        # Convert spike trains to numpy arrays
        spike_times_out = convert_value(spike_times_out)
        spike_times_ref = convert_value(spike_times_ref)

        # Contribution from interaction between output spikes
        dist_out = 0.0
        for i in spike_times_out:
            for j in spike_times_out:
                dist_out += np.exp(-(2*max(i, j) - i - j) / tau_c)

        # Contribution from interaction between reference spikes
        dist_ref = 0.0
        for i in spike_times_ref:
            for j in spike_times_ref:
                dist_ref += np.exp(-(2*max(i, j) - i - j) / tau_c)

        # Contribution from interaction between output and reference spikes
        dist_out_ref = 0.0
        for i in spike_times_out:
            for j in spike_times_ref:
                dist_out_ref += np.exp(-(2*max(i, j) - i - j) / tau_c)

        return (dist_out + dist_ref - 2*dist_out_ref) / 2

    except TypeError:
        print "Invalid type of input argument(s)"

        return None


def van_rossum_spatio(spike_trains_out, spike_trains_ref, tau_c=10.0):
    """
    Exactly computes the van Rossum distance between two spatiotemporal spike
    patterns

    Parameters
    ----------
    spike_trains_out : list
    spike_trains_ref : list
    tau_c : float
        van Rossum time constant (default 10 ms)

    Returns
    -------
    distance : numpy.float64
        Distance between spatiotemporal spike patterns

    """

    n_trains = len(spike_trains_out)
    if len(spike_trains_ref) != n_trains:
        raise ValueError('Spike patterns differ spatially in size')

    spatio_dist = 0.0
    for i in xrange(n_trains):
        spatio_dist += van_rossum(spike_trains_out[i], spike_trains_ref[i],
                                  tau_c)

    return spatio_dist
