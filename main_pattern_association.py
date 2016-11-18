#!/usr/bin/env python2
# -*- coding: utf-8 -*--
"""
Created on Mon Oct 24 16:09:17 2016

@author: brian

Dependencies:
    Python 2.7
    pyNN 0.8.1
    Numpy
    Matplotlib
    nest 2.10.0 (backend simulator used in this case)

PatternAssociation:
    A fully-connected, feed-forward network with a single layer is trained to
    associate arbitrarily generated input-output spike patterns through weight
    modifications. Weights are modified according to either INST or FILT
    learning method. C.f. Gardner & Gruening (2016).

Key network parameters:
    - n_classes:        No. of input classes
    - n_patterns_class: No. of input patterns assigned per class.
    - n_target spikes:  No. of target output spikes, at each output neuron,
                        that must be reproduced by the network. Each set of
                        target output spike trains is arbitrarily
                        (and uniquely) determined for each input class.
    - n_inputs:         No. of input spike trains in the network, with
                        feed-forward and all-to-all connectivity with the
                        output layer.
    - n_outputs:        No. of output layer neurons, each sharing the same,
                        received input pattern, but each tasked with
                        learning a different target spike train.
    - n_epochs:         No. of learning epochs, where one epoch corresponds to
                        the (sequential) presentation of each pattern
"""

import numpy as np
import matplotlib.pyplot as plt  # For plotting

import pyNN.nest as sim  # NEST is used here as the backend for simulations, although any choice should be possible

import values  # Parameters are defined via this module
from lib import task, snn, utility, distmetric
from pyNN.parameters import Sequence  # Spike trains should be of type Sequence


def main():
    # === Init. parameters ====================================================

    # These parameter choices recreate results in Fig. 5 of
    # Gardner & Gruening 2016 (although arbitrary target spikes and exact spike
    # precision are used here)
    # Param(n_classes, n_patterns_class, n_target_spikes, n_inputs, n_outputs,
    #       n_epochs)
    param = values.Param(1, 1, 4, 200, 1, 200)  # For 200 inputs and T = 200 ms: ~9 seconds wall time for 100 runs

    sim.setup(timestep=param.dt, min_delay=0.1, spike_precision='off_grid')  # Must have minimum conductance delay of 0.1 ms, 'off_grid' for exact spiking precision

    sim.nest.sr("M_FATAL setverbosity")  # Suppress most nest warnings - remove this for a different simulator

    rng = sim.NumpyRNG(42)  # Random number generator

    # === Learning task =======================================================

    # Specify task
    learn_task = task.PatternAssociation(param, rng)

    # Generate arbitrary input / target patterns
    learn_task.build_static_patterns()

    # === Build network =======================================================

    #net = snn.NetworkINST(param, sim, rng)  # Select INST learning rule
    net = snn.NetworkFILT(param, sim, rng)  # Select FILT learning rule

    # === Record ==============================================================

    class Record(object):
        """Simulation recordings container"""
        def __init__(self, learn_task, param):
            self.err = np.zeros(param.n_epochs)  # Network error: van Rossum distance
    #        if param.n_patterns == 1:
    #            self.w = np.zeros((param.n_inputs, param.n_epochs))  # Record weights per epoch for first output neuron (large slowdown)
    #            self.spikes = [None] * param.n_epochs  # Actual output spike trains
    #            self.spikes_target = learn_task.pattern_target[0].spike_trains  # Target output spike trains for first pattern

    rec = Record(learn_task, param)

    # === Run simulation ======================================================

    for i in xrange(param.n_epochs):
        rec.err[i] = net.learn(learn_task)
    #    if param.n_patterns == 1:
    #        rec.w[:, i] = net.connections_exc.get('weight', 'array')[:, 0]
    #        rec.spikes[i] = net.get_output_spikes()  # Actual output spike trains from last presented pattern
        epochs_completed = i + 1
        if epochs_completed % 50 == 0:
            print epochs_completed

    # === Gather and plot results =============================================

    #record_in = net.layer_in.get_data()
    record_out = net.layer_out.get_data()

    # Plots:
    utility.plot_error(rec.err)  # Performance metric
    #utility.plot_spikepattern(learn_task.pattern_input[0].spike_trains, param.T)  # Plot input pattern

    # Plots for one pattern:
    if param.n_patterns == 1:
        #utility.plot_signal(record_out)  # Voltage trace (default last epoch)
        #plt.plot(np.transpose(rec.w))  # Plot evolution of synaptic weights
        utility.plot_spiker(record_out, learn_task.pattern_target[-1].spike_trains, 0)  # Spike raster of given output neuron

    # === End simulation ======================================================

    sim.end()


if __name__ == '__main__':
    main()
