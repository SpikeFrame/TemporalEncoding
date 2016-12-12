#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  1 14:57:56 2016

@author: Brian

Simulation parameter values

According to pyNN, PSC for IF_curr_exp type is not normalised and dimensionless
(i.e. PSC = exp(-s / tau_syn)), and weights have units of nA. Weight * tau_syn
gives total charge transferred due to one input spike for IF_curr_exp.
"""

from __future__ import division

from pyNN.random import NumpyRNG


class NetworkParam(object):
    """Common network parameters"""

    def __init__(self, n_patterns, n_target_spikes, n_inputs, n_outputs,
                 n_epochs, seed=None):
        self.n_patterns = n_patterns            # Total no. patterns
        self.n_target_spikes = n_target_spikes  # Desired no. target spikes
        self.n_inputs = n_inputs                # No. afferent spike trains
        self.n_outputs = n_outputs              # No. postsynaptic neurons
        self.n_epochs = n_epochs                # Total no. training epochs
        self.rng = NumpyRNG(seed)               # Random number generator
        # Initial weight bounds based on uniform distrib. (nA)
        self.w_unif_init = (0.0, 200.0 / self.n_inputs)

    # Time
    dt = 0.1  # Time step (ms)
    T = 200.0  # Simulation duration (ms)
    n_iterations = int(round(T / dt))

    # IF_curr_exp cell type
    cell_params = {'cm': 2.5,           # Capacitance (nF)
                   'i_offset': 0.0,     # Offset current (nA)
                   'tau_m': 10.0,       # Membrane time constant (ms)
                   'tau_refrac': 0.1,   # Absolute refractory time (ms)
                   'tau_syn_E': 5.0,    # Synaptic time constant (Exc) (ms)
                   'tau_syn_I': 5.0,    # Synaptic time constant (Inh) (ms)
                   'v_reset': -65.0,    # Reset voltage (mV)
                   'v_rest': -65.0,     # Resting voltage (mV)
                   'v_thresh': -50.0}   # Firing threshold (mV)

    # Input patterns
    n_input_spikes = 1  # No. of spikes from each input as unif. distrib.
    r_base = 5.0  # Input firing rate as Poisson distrib. (Hz)
    noise_stdev = 1.0  # Noise amplitude in jittering input spikes (ms)

    # Coefficient factor for INST & FILT learning rules
    psp_coeff = 1.0 / cell_params['cm'] * (cell_params['tau_m'] * cell_params['tau_syn_E']) / (cell_params['tau_m'] - cell_params['tau_syn_E'])
    # Coefficients for FILT rule
    tau_q = 10.0  # filter time constant (c.f. van Rossum distance)
    filt_coeff_m = psp_coeff * cell_params['tau_m'] / (cell_params['tau_m'] + tau_q)  # Fold psp_coeff and C_m into one constant
    filt_coeff_s = psp_coeff * cell_params['tau_syn_E'] / (cell_params['tau_syn_E'] + tau_q)  # Fold psp_coeff and C_s into one constant
    filt_coeff_acausal = filt_coeff_m - filt_coeff_s  # Coefficient for acausal trace


class PatternAssocParam(NetworkParam):
    """Pattern association parameters"""

    # Target patterns
    t_min = 40.0  # Min. first target spike time relative to pattern onset (ms)

    def __init__(self, n_classes, n_patterns_class, n_target_spikes, n_inputs,
                 n_outputs, n_epochs, seed=None):
        super(PatternAssocParam, self).__init__(n_patterns_class * n_classes,
                                                n_target_spikes, n_inputs,
                                                n_outputs, n_epochs, seed)
        self.n_classes = n_classes
        self.n_patterns_class = n_patterns_class
        # Learning rate (reduce for less patterns for increased stability)
        self.eta = 600.0 / (self.n_inputs * self.n_target_spikes *
                            self.n_patterns)


class MimicParam(NetworkParam):
    """Mimicry parameters"""

    def __init__(self, n_patterns, n_target_spikes, n_inputs, n_outputs,
                 n_epochs, seed=None):
        super(MimicParam, self).__init__(n_patterns, n_target_spikes, n_inputs,
                                         n_outputs, n_epochs, seed)
        
        # Standard deviation of initialised tutor output neuron weights
        self.w_ref_sigma = 130.0 / self.n_inputs
        # Learning rate
        self.eta = 300.0 / (self.n_inputs * self.n_target_spikes *
                            self.n_patterns)
