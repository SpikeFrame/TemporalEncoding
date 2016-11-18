#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  2 17:42:58 2016

@author: brian

Templates for learning tasks
"""

from __future__ import division
import numpy as np

from pyNN.random import NumpyRNG

from lib import spikepattern, distmetric


class GenericTask(object):
    """Abstract learning class"""

    def __init__(self, param, rng=NumpyRNG()):
        self.param = param
        self.rng = rng
        self.pattern_input = [None] * param.n_patterns
        self.pattern_target = [None] * param.n_classes

        self.built = False


class PatternAssociation(GenericTask):
    """Arbitrary input-output pattern association task"""

    def __init__(self, param, rng=NumpyRNG()):
        super(PatternAssociation, self).__init__(param, rng)
        # Setup static patterns : uniform distrib.
        #  self.build_static_patterns()

    def build_static_patterns(self):
        # Create input patterns
        for i in xrange(self.param.n_patterns):
            self.pattern_input[i] = spikepattern.Input(self.param.n_inputs)
            self.pattern_input[i].build_unif(self.param, self.rng)
            self.pattern_input[i].class_n = int(np.floor(i /
                                                self.param.n_patterns_class))

        # Create unique target patterns
        dist_min = self.param.n_target_spikes / 2
        for i in xrange(self.param.n_classes):
            self.pattern_target[i] = spikepattern.Target(self.param.n_outputs)
            while not self.pattern_target[i].built:
                self.pattern_target[i].build_unif(self.param, self.rng)
                for j in xrange(i):
                    if distmetric.van_rossum_spatio(self.pattern_target[i].spike_trains,
                                                    self.pattern_target[j].spike_trains) < dist_min:
                        self.pattern_target[i].built = False
                        break

        self.built = True
