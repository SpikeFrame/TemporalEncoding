#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  4 15:07:21 2016

@author: Brian

Spiking Neural Network (SNN) definitions, and custom supervised learning rules
"""

import numpy as np

from pyNN.parameters import Sequence

from lib.distmetric import van_rossum_spatio


class SingleLayer(object):
    """
    General structure and routines for single-layer feed-forward spiking neural
    network with all-to-all connections
    """

    def __init__(self, param, sim):
        # Internel reference to simulation parameters
        self.param = param
        # Backend simulator selection
        self.sim = sim

        # Define neuron type
        self.neuron_type = sim.IF_curr_exp(**param.cell_params)
        # Connectivity type
        self.connectivity = sim.AllToAllConnector()
        # Fixed input spike delay
        self.d = sim.get_min_delay()

        self.built = False

    def build_network(self, spike_trains_input, tutor_signal=False):
        """
        Initialise network structure and state using given input spike pattern
        Structure: input->output neurons[, referent output neurons]
        """
        if self.built:
            return
        # Setup layers: layer_in -> layer_out
        self.layer_in = self.sim.Population(self.param.n_inputs, self.sim.SpikeSourceArray(spike_times=spike_trains_input), label='input')
        self.layer_out = self.sim.Population(self.param.n_outputs, self.neuron_type, label='output')
        # Initial weight values
        self.w = self.param.rng.uniform(*self.param.w_unif_init,
                                        size=(self.param.n_inputs,
                                              self.param.n_outputs))
        # Connect network
        syn = self.sim.StaticSynapse(weight=self.w)  # Weights are fixed during the simulation run
        self.connections_exc = self.sim.Projection(self.layer_in,
                                                   self.layer_out,
                                                   self.connectivity, syn,
                                                   receptor_type='excitatory')
        self.layer_out.record(['v', 'spikes'])
#        self.layer_out.record('spikes')

        # (Optional) connect tutor neurons to form a reference output pattern:
        # one-one correspondence with each actual output
        if tutor_signal:
            # Setup referent output layer
            self.layer_out_ref = self.sim.Population(self.param.n_outputs,
                                                     self.neuron_type,
                                                     label='ref')
            # Referent weight matrix: mean value is sigma for ~16% -ve weights
            self.w_ref = self.param.rng.normal(self.param.w_ref_sigma,
                                               self.param.w_ref_sigma,
                                               (self.param.n_inputs,
                                                self.param.n_outputs))
            syn_ref = self.sim.StaticSynapse(weight=self.w_ref)
            self.connections_ref = self.sim.Projection(self.layer_in,
                                                       self.layer_out_ref,
                                                       self.connectivity,
                                                       syn_ref,
                                                       receptor_type='excitatory')
            self.layer_out_ref.record('spikes')
        self.built = True

    def get_spikes(self, layer):
        """Return recorded spike trains of a given layer, from latest run"""
        spike_trains_out = [Sequence(np.array(spike_times)) for spike_times in
                            layer.get_data('spikes').segments[-1].spiketrains]
        return spike_trains_out

    def reset(self, spike_trains_input):
        """Reset network state and set next input pattern"""
        self.sim.reset()
        self.layer_in.set(spike_times=spike_trains_input)

    def run(self, spike_trains_input, tutor_signal=False):
        """Run the network for given pattern"""
        if not self.built:
            self.build_network(spike_trains_input, tutor_signal)
        else:
            self.reset(spike_trains_input)
        # Run simulation
        self.sim.run(self.param.T)


class NetworkLearning(SingleLayer):
    """Template for training a network"""

    def learning_window(self, lag):
        """Overwritten by subclass: learning window for weight updates"""
        return 0.0

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

    def learn(self, learn_task):
        """
        Simulate network for one epoch (sequential pattern presentations),
        update weights based on target signal and return average network error
        """
        # Initialise candidate weight changes
        dw = np.zeros((self.param.n_inputs, self.param.n_outputs))
        # Initial network error
        err = 0.0
        # Record maximum timing displacements of actual w.r.t. target spike(s),
        # per pattern
        dt_max = np.full((self.param.n_patterns, self.param.n_outputs), np.inf)

        # Check if learning task requires tutor neurons
        if hasattr(learn_task, 'pattern_target'):
            tutor_signal = False
        else:
            tutor_signal = True

        # Jitter input spike patterns if applicable
        if hasattr(learn_task, 'pattern_input_ref'):
            learn_task.jittered_inputs()

        # Iterate over learning trials
        for i in xrange(self.param.n_patterns):
            p = learn_task.pattern_input[i]
            # Simulate
            self.run(p.spike_trains, tutor_signal)
            # Get output / target spikes
            spike_trains_out = self.get_spikes(self.layer_out)
            if tutor_signal:
                spike_trains_target = self.get_spikes(self.layer_out_ref)
            else:
                p_label = learn_task.labels[i]
                spike_trains_target = learn_task.pattern_target[p_label].spike_trains
            # Update candidate weight change
            dw += self.weight_changes(p.spike_trains, spike_trains_out,
                                      spike_trains_target)
            # Update network error
            err += van_rossum_spatio(spike_trains_out, spike_trains_target)
            # Record maximum timing displacement of output layer nrns;
            # Num. actual and target spikes per output nrn must match, and at
            # least one target spike must be present
            for nrn in xrange(self.param.n_outputs):
                spikes_out = spike_trains_out[nrn].value
                spikes_target = spike_trains_target[nrn].value
                if len(spikes_out) == len(spikes_target) and \
                   len(spikes_target > 0):
                    dt_max[i, nrn] = np.max(np.abs(spikes_out - spikes_target))

        # Update weights
        self.w = self.connections_exc.get('weight', 'array') + dw
        self.connections_exc.set(weight=self.w.ravel())
        # Return network error and max. timing displacements
        return err / len(learn_task.pattern_input), dt_max


class NetworkINST(NetworkLearning):
    """INST learning rule plasticity"""

    def learning_window(self, lag):
        """Returns weight change based on lag time (pre to post)"""
        if lag <= 0.0:  # Acausal trace
            return 0.0
        else:  # Causal trace
            return self.param.psp_coeff * (np.exp(-lag / self.param.cell_params['tau_m']) -
                                           np.exp(-lag / self.param.cell_params['tau_syn_E']))


class NetworkFILT(NetworkLearning):
    """FILT learning rule plasticity"""

    def learning_window(self, lag):
        """Returns weight change based on lag time (pre to post)"""
        if lag <= 0.0:
            return self.param.filt_coeff_acausal * np.exp(lag / self.param.tau_q)
        else:
            return self.param.filt_coeff_m * np.exp(-lag / self.param.cell_params['tau_m']) - self.param.filt_coeff_s * np.exp(-lag / self.param.cell_params['tau_syn_E'])
