#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 27 16:17:53 2016

@author: brian

Useful plotting / data functions
"""

from __future__ import division
import numpy as np
import matplotlib.pyplot as plt

from pyNN.utility.plotting import Figure, Panel


def plot_spikepattern(spike_trains, sim_time):
    """Plot set of spike trains (spike pattern)"""
    plt.ioff()

    plt.figure()
    for i in xrange(len(spike_trains)):
        spike_times = spike_trains[i].value
        plt.plot(spike_times, np.full(len(spike_times), i,
                 dtype=np.int), 'k.')
    plt.xlim((0.0, sim_time))
    plt.ylim((0, len(spike_trains)))
    plt.xlabel('Time (ms)')
    plt.ylabel('Neuron index')
    plt.show()

    plt.ion()


def plot_signal(record, seg=-1):
    """
    Plot voltage trace for all output neurons
    (seg=-1 corresponds to last trial no.)
    """
    data = record.segments[seg]
    vm = data.filter(name="v")[0]
    Figure(Panel(vm, ylabel="Membrane potential (mV)", xlabel="Time (ms)",
                 xticks=True, yticks=True))


def plot_spiker(record, spike_trains_target, neuron_index=0):
    """Plot spikeraster and target timings for given neuron index"""
    plt.ioff()

    spike_trains = [np.array(i.spiketrains[neuron_index])
                    for i in record.segments]
    n_segments = record.size['segments']

    plt.figure()
    for i in xrange(len(spike_trains)):
        plt.plot(spike_trains[i], np.full(len(spike_trains[i]), i + 1,
                 dtype=np.int), 'k.')
    target_timings = spike_trains_target[neuron_index].value
    plt.plot(target_timings, np.full(len(target_timings), 1.025 * n_segments),
             'kx', markersize=8, markeredgewidth=2)
    plt.xlim((0., np.float(record.segments[0].t_stop)))
    plt.ylim((0, np.int(1.05 * n_segments)))
    plt.xlabel('Time (ms)')
    plt.ylabel('Trials')
    plt.title('Output neuron {}'.format(neuron_index))
    plt.show()

    plt.ion()


def plot_error(err):
    """Plot error with each epoch"""
    plt.ioff()

    plt.figure()
    plt.plot(1 + np.arange(len(err)), err, 'k-', linewidth=1)
    plt.xlabel('Epochs')
    plt.ylabel('van Rossum distance')
    plt.grid()
    plt.show()

    plt.ion()


def psp(s, param):
    """PSP for IF_curr_exp cell type"""
    tau_m, tau_s = param.cell_params['tau_m'], param.cell_params['tau_syn_E']
    psp_coeff = param.psp_coeff
    if s < 0.0:
        return 0.0
    else:
        return psp_coeff * (np.exp(-s / tau_m) - np.exp(-s / tau_s))


def lag_peak(param):
    """Lag time at which point the PSP assumes its peak value"""
    tau_m, tau_s = param.cell_params['tau_m'], param.cell_params['tau_syn_E']
    return tau_m * tau_s / (tau_m - tau_s) * np.log(tau_m / tau_s)


def psp_peak(param):
    """Peak value of the PSP"""
    return psp(lag_peak(param), param)
