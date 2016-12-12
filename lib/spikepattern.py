#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  1 16:36:53 2016

@author: Brian

Input / target spike train containers
"""

from __future__ import division
import numpy as np

from pyNN.parameters import Sequence

import spikegen


class Pattern(object):
    """
    Generic spike pattern
    """
    def __init__(self, n_trains, param):
        self.n_trains = n_trains  # No. spike trains
        self.param = param
        self.spike_trains = [None] * n_trains  # Spike trains

        self.built = False

    def jitter(self, pattern_ref, noise_stdev):
        """
        Builds pattern based on jittered copy of pattern_ref
        """
        for i in xrange(self.n_trains):
            # Copy spike times from reference (unjittered) pattern
            spike_times = pattern_ref.spike_trains[i].value.copy()
            self.spike_trains[i] = Sequence(self.param.rng.normal(spike_times,
                                                                  noise_stdev,
                                                                  spike_times.size))
        self.built = True


class Input(Pattern):
    """
    Input spike pattern
    """
    def __init__(self, param):
        super(Input, self).__init__(param.n_inputs, param)

    def build(self, spike_distrib='uniform'):
        """Build input spike pattern according to a given distribution"""
        if self.built:
            return
        if spike_distrib == 'uniform':
            for i in xrange(self.n_trains):
                self.spike_trains[i] = spikegen.unif(self.param.n_input_spikes,
                                                     self.param.T,
                                                     self.param.rng, False,
                                                     self.param.dt)
        elif spike_distrib == 'poisson':
            for i in xrange(self.n_trains):
                self.spike_trains[i] = spikegen.poisson(self.param.r_base,
                                                        self.param.T,
                                                        self.param.rng,
                                                        False, self.param.dt)
        else:
            raise ValueError('Invalid spiking distribution')
        self.built = True


class Target(Pattern):
    """
    Target spike pattern
    """
    def __init__(self, param):
        super(Target, self).__init__(param.n_outputs, param)

    def build(self, spike_distrib='uniform'):
        """Build target spike pattern according to a given distribution"""
        if self.built:
            return
        if spike_distrib == 'uniform':
            for i in xrange(self.n_trains):
                self.spike_trains[i] = spikegen.unif(self.param.n_target_spikes,
                                                     self.param.T,
                                                     self.param.rng, True,
                                                     self.param.t_min)
        elif spike_distrib == 'poisson':
            # No spikes for target output rate of zero
            if self.param.n_target_spikes == 0:
                self.spike_trains = [Sequence(np.array([]))
                                     for i in xrange(self.n_trains)]
            else:
                # Interpret n_target_spikes as expected number of spikes
                target_rate = self.param.n_target_spikes / self.param.T * 1000.
                for i in xrange(self.n_trains):
                    while True:
                        self.spike_trains[i] = spikegen.poisson(target_rate,
                                                                self.param.T,
                                                                self.param.rng,
                                                                True, self.param.t_min)
                        # Ensure at least one target spike to classify
                        if len(self.spike_trains[i].value) > 0:
                            break
        else:
            raise ValueError('Invalid spiking distribution')
        self.built = True
