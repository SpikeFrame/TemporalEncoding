#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  4 15:07:21 2016

@author: brian

Spiking Neural Network (SNN) definitions, and custom supervised learning rules
"""

import numpy as np

from pyNN.random import NumpyRNG
from pyNN.parameters import Sequence

from lib.distmetric import van_rossum_spatio


class SingleLayer(object):
    """
    Template: single-layer feed-forward spiking neural network with all-to-all
    connections
    """

    def __init__(self, param, sim, rng=NumpyRNG()):
        # Internel reference to simulation parameters
        self.param = param
        # Backend simulator selection
        self.sim = sim
        # Random number generator reference
        self.rng = rng

        # Define neuron type
        self.neuron_type = sim.IF_curr_exp(**param.cell_params)
        # Connectivity type
        self.connectivity = sim.AllToAllConnector()
        # Initial weight values
        self.w = rng.uniform(*param.w_unif_init, size=(param.n_inputs,
                                                       param.n_outputs))
        # Fixed input spike delay
        self.d = sim.get_min_delay()

        self.built = False

    def build_network(self, spike_trains_input):
        """Initialise network using given input spike pattern"""

        # Setup layers: layer_in -> layer_out
        self.layer_in = self.sim.Population(self.param.n_inputs, self.sim.SpikeSourceArray(spike_times=spike_trains_input), label='input')
        self.layer_out = self.sim.Population(self.param.n_outputs, self.neuron_type, label='output')

        # Connect network
        syn = self.sim.StaticSynapse(weight=self.w)  # Weights are fixed during the simulation run
        self.connections_exc = self.sim.Projection(self.layer_in,
                                                   self.layer_out,
                                                   self.connectivity, syn,
                                                   receptor_type='excitatory')

        #  Recordings
#        self.layer_in.record('spikes')
        self.layer_out.record(['v', 'spikes'])
#        self.layer_out.record('spikes')

        self.built = True

    def get_output_spikes(self):
        """Return recorded output spike trains from latest run"""
        spike_trains_out = [Sequence(np.array(spike_times)) for spike_times in
                            self.layer_out.get_data('spikes').segments[-1].spiketrains]
        return spike_trains_out

    def learning_window(self, lag):
        """Overwritten by subclass: learning window for weight updates"""
        return None

    def weight_changes(self, spike_trains_input, spike_trains_output,
                       spike_trains_target):
        """Batch weight changes"""
        dw = np.zeros((len(spike_trains_input), len(spike_trains_output)))
        # Weight updates due to actual output spikes
        for i in xrange(len(spike_trains_output)):  # Output neurons
            for j in xrange(len(spike_trains_input)):  # Input neurons
                spike_times_output = spike_trains_output[i].value
                spike_times_input = spike_trains_input[j].value
                for t_out in spike_times_output:  # Output spikes
                    for t_in in spike_times_input:  # Input spikes
                        dw[j, i] -= self.learning_window(t_out - t_in - self.d)

        # Weight updates due to target output spikes
        for i in xrange(len(spike_trains_target)):  # Output neurons
            for j in xrange(len(spike_trains_input)):  # Input neurons
                spike_times_output = spike_trains_target[i].value
                spike_times_input = spike_trains_input[j].value
                for t_out in spike_times_output:  # Target output spikes
                    for t_in in spike_times_input:  # Input spikes
                        dw[j, i] += self.learning_window(t_out - t_in - self.d)

        return self.param.eta * dw

    def reset(self, spike_trains_input):
        """Reset network state for next input pattern"""
        self.sim.reset()
        self.layer_in.set(spike_times=spike_trains_input)

    def run(self, spike_trains_input):
        """Run the network for given pattern"""
        if not self.built:
            self.build_network(spike_trains_input)
        else:
            self.reset(spike_trains_input)

        # Run simulation
        self.sim.run(self.param.T)

    def learn(self, learn_task):
        """
        Simulate network for one epoch (each pattern), update weights and
        return average network error
        """

        # Initialise candidate weight changes
        dw = np.zeros((self.param.n_inputs, self.param.n_outputs))
        # Initial network error
        err = 0.0

        # Iterate over learning trials
        for p in learn_task.pattern_input:
            # Simulate
            self.run(p.spike_trains)
            # Get output / target spikes
            spike_trains_out = self.get_output_spikes()
            spike_trains_target = learn_task.pattern_target[p.class_n].spike_trains
            # Update candidate weight change
            dw += self.weight_changes(p.spike_trains, spike_trains_out,
                                      spike_trains_target)
            # Update network error
            err += van_rossum_spatio(spike_trains_out, spike_trains_target)

        # Update weights
        self.w = self.connections_exc.get('weight', 'array') + dw
        self.connections_exc.set(weight=self.w.ravel())
        # Return network error
        return err / len(learn_task.pattern_input)


class NetworkINST(SingleLayer):
    """INST learning rule defined"""

    def learning_window(self, lag):
        """Returns weight change based on lag time (pre to post)"""
        if lag <= 0.0:  # Acausal trace
            return 0.0
        else:  # Causal trace
            return self.param.psp_coeff * (np.exp(-lag / self.param.cell_params['tau_m']) -
                                           np.exp(-lag / self.param.cell_params['tau_syn_E']))


class NetworkFILT(SingleLayer):
    """FILT learning rule defined"""

    def learning_window(self, lag):
        """Returns weight change based on lag time (pre to post)"""
        if lag <= 0.0:
            return self.param.filt_coeff_acausal * np.exp(lag / self.param.tau_q)
        else:
            return self.param.filt_coeff_m * np.exp(-lag / self.param.cell_params['tau_m']) - self.param.filt_coeff_s * np.exp(-lag / self.param.cell_params['tau_syn_E'])
