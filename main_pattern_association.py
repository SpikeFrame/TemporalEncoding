#!/usr/bin/env python2
# -*- coding: utf-8 -*--
"""
Created on Mon Oct 24 16:09:17 2016

@author: Brian

Dependencies:
    Python 2.7
    pyNN 0.8.1
    Numpy
    Matplotlib
    nest 2.10.0 (backend simulator used in this case)

A fully-connected, feed-forward network with a single layer is trained to
associate arbitrarily generated input spike patterns with target output signals
through weight modifications. Weights are modified according to either INST or
FILT learning method. C.f. Gardner & Gruening (2016).

PatternAssociation:
    Learning a prescribed target output spike pattern.
Parameters:
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

TutorMimicry:
    Learning a tutor signal.
Parameters:
    - n_patterns:       No. of input patterns.
    - n_target spikes:  Expected no. of target output spikes at tutor neurons
    - n_inputs:         No. of input spike trains in the network, with
                        feed-forward and all-to-all connectivity with the
                        output layer.
    - n_outputs:        No. of output layer neurons, each with a one-one
                        correspondence with output layer tutor neurons
    - n_epochs:         No. of learning epochs
"""

import numpy as np
import matplotlib.pyplot as plt  # For plotting

import pyNN.nest as sim  # NEST is used here as the backend for simulations
from pyNN.parameters import Sequence  # Spike trains should be of type Sequence

import values  # Parameters are defined via this module
from lib import task, snn, utility


def main(task_id=0):
    """
    Parameters
    ----------
    task_id : int
        Identifier for learning task selection:
            0 : Pattern association
            1 : Mimicking tutor neurons

    Returns
    -------
    rec : record struct
        Container for network output / input pattern recordings during
        simulation runs.
    """
    # === Init. parameters and specify task  ==================================

    # Seed for random number generator (use 'seed = None' for system clock)
    seed = 42

    if task_id == 0:
        # These parameter choices recreate results in Fig. 5 of
        # Gardner & Gruening 2016 (although arbitrary target spikes and exact
        # spike precision are used here)
        # For 200 inputs and T = 200 ms: ~8.7 seconds wall time for 100 runs
        param = values.PatternAssocParam(n_classes=1, n_patterns_class=1,
                                         n_target_spikes=4, n_inputs=200,
                                         n_outputs=1, n_epochs=100, seed=seed)
        learn_task = task.PatternAssociation(param)
    elif task_id == 1:
        param = values.MimicParam(n_patterns=1, n_target_spikes=4,
                                  n_inputs=200, n_outputs=1, n_epochs=100,
                                  seed=seed)
        learn_task = task.Mimicry(param)
    else:
        # Invalid input argument
        raise ValueError('Invalid main argument')

    # Initialise simulator
    # Must have minimum conductance delay of 0.1 ms, 'off_grid' for
    # exact spiking precision
    sim.setup(timestep=param.dt, min_delay=0.1, spike_precision='off_grid',
              verbosity='error')

    # === Record ==============================================================

    class Record(object):
        """Simulation recordings container"""
        def __init__(self, learn_task, param):
            # Network error: van Rossum distance
            self.err = np.zeros(param.n_epochs)
            # Network absolute timing displacements
            self.dt_max = np.full((param.n_epochs, param.n_patterns,
                                   param.n_outputs), np.inf)
            # Record weights per epoch for output layer
            self.w = np.zeros((param.n_epochs, param.n_inputs,
                               param.n_outputs))
            if task_id == 1:
                # Euclidean distance between actual and tutor weights per epoch
                self.w_err = np.empty(param.n_epochs)
    rec = Record(learn_task, param)

    # === Setup patterns and network ==========================================

    # Generate arbitrary input / target patterns
    # Parameter choices are: {'uniform', 'poisson'}
    learn_task.build_static_patterns('uniform')  # Select static input spikes
#    learn_task.build_dynamic_patterns('uniform')  # Select noisy input spikes

#    net = snn.NetworkINST(param, sim)  # Select INST learning rule
    net = snn.NetworkFILT(param, sim)  # Select FILT learning rule

    # === Run simulation ======================================================

    for i in xrange(param.n_epochs):
        # Simulate network
        rec.err[i], rec.dt_max[i] = net.learn(learn_task)
        # Record weights of output layer
        rec.w[i, :, :] = net.w
        if task_id == 1:
            rec.w_err[i] = np.linalg.norm(net.w - net.w_ref)
        # Print progress
        epochs_completed = i + 1
        if epochs_completed % 50 == 0:
            print epochs_completed

    # === Gather and plot results =============================================

    rec.input = learn_task
    if task_id == 1:
        rec.w_ref = net.w_ref

    # Classification performance
    accs = utility.accuracy(rec.dt_max)
    # Exponentially-weighted moving average
    accs_ewma = utility.ewma_vec(accs, len(accs) / 10.)
    # Plots:
    utility.plot_error(rec.err)  # Distance metric
    utility.plot_accuracy(accs_ewma)  # Performance metric

    # Plots for one pattern:
    if param.n_patterns == 1:
        rec.output = net.layer_out.get_data()
        if task_id == 1:
            rec.output_ref = net.layer_out_ref.get_data()
            rec.spikes_ref = [Sequence(np.array(spikes_ref)) for spikes_ref in rec.output_ref.segments[-1].spiketrains]
        else:
            rec.spikes_ref = learn_task.pattern_target[-1].spike_trains
#        utility.plot_signal(rec.output)  # Voltage trace (default last epoch)
#        plt.plot(rec.w[:, :, 0])  # Plot evolution of first output neuron's synaptic weights
#        plt.hist(rec.w[-1, :, 0])  # Plot histogram of first output neuron's synaptic weights
#        utility.plot_spikepattern(rec.input.pattern_input[0].spike_trains, rec.input.param.T)  # Plot an input pattern
        utility.plot_spiker(rec.output, rec.spikes_ref, neuron_index=0)  # Spike raster of given output neuron

#    if task_id == 1:
#        # Plot Euclidean distance between actual and tutor weights
#        plt.plot(rec.w_err)

    # === End simulation ======================================================

    sim.end()
    return rec


if __name__ == '__main__':
    rec = main(task_id=0)
