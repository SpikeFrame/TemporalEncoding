#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  1 16:36:53 2016

@author: brian

Input / target pattern containers
"""

from pyNN.random import NumpyRNG

import spikegen


class Pattern(object):
    """Generic spike pattern"""

    def __init__(self, n_trains):
        self.n_trains = n_trains  # No. spike trains
        self.spike_trains = [None] * n_trains  # Spike trains
        self.built = False


class Input(Pattern):
    """Input spike pattern"""

    def build_unif(self, param, rng=NumpyRNG()):
        """Build input spike pattern as unif. distrib."""
        for i in xrange(self.n_trains):
            self.spike_trains[i] = spikegen.unif(param.n_input_spikes, param.T,
                                                 rng, False, param.dt)
        self.built = True


class Target(Pattern):
    """Target spike pattern"""

    def build_unif(self, param, rng=NumpyRNG()):
        """Build target spike pattern as unif. distrib."""
        for i in xrange(self.n_trains):
            self.spike_trains[i] = spikegen.unif(param.n_target_spikes,
                                                 param.T, rng, True,
                                                 param.t_min)
        self.built = True
