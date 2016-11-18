#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  1 14:57:56 2016

@author: brian

Simulation parameter values

According to pyNN, PSC for IF_curr_exp type is not normalised and dimensionless
(i.e. PSC = exp(-s / tau_syn)), and weights have units of nA. Weight * tau_syn
gives total charge transferred due to one input spike for IF_curr_exp.
"""

from __future__ import division


class Param(object):
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
    r_base = 10.0  # Input firing rate as Poisson distrib. (Hz)

    # Target patterns
    t_min = 40.0  # Min. timing of first target spike relative to pattern onset (ms)

    # Coefficient factor for INST & FILT learning rules
    psp_coeff = 1.0 / cell_params['cm'] * (cell_params['tau_m'] * cell_params['tau_syn_E']) / (cell_params['tau_m'] - cell_params['tau_syn_E'])
    # Coefficients for FILT rule
    tau_q = 10.0  # filter time constant (c.f. van Rossum distance)
    filt_coeff_m = psp_coeff * cell_params['tau_m'] / (cell_params['tau_m'] + tau_q)  # Fold psp_coeff and C_m into one constant
    filt_coeff_s = psp_coeff * cell_params['tau_syn_E'] / (cell_params['tau_syn_E'] + tau_q)  # Fold psp_coeff and C_s into one constant
    filt_coeff_acausal = filt_coeff_m - filt_coeff_s  # Coefficient for acausal trace

    def __init__(self, n_classes, n_patterns_class, n_target_spikes, n_inputs,
                 n_outputs, n_epochs):
        self.n_classes = n_classes
        self.n_patterns_class = n_patterns_class
        self.n_patterns = n_patterns_class * n_classes  # Total no. patterns
        self.n_target_spikes = n_target_spikes
        self.n_inputs = n_inputs  # No. afferent spike trains
        self.n_outputs = n_outputs  # No. postsynaptic neurons
        self.n_epochs = n_epochs  # Maximum number of training epochs

        self.w_unif_init = (0.0, 200.0 / self.n_inputs)  # (nA) default: 200.0, reduce for less patterns
        self.eta = 600.0 / (self.n_inputs * self.n_target_spikes *
                            self.n_patterns)  # Default factor: 600.0, reduce for less patterns
