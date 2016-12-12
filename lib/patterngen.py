#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  9 21:04:42 2016

@author: Brian

Routines for generating different types of spike patterns
"""

from __future__ import division

from lib import spikepattern, distmetric


def input_patterns(param, spike_distrib='uniform'):
    """
    Create a set of input patterns with spikes following spike_distrib
    """
    pattern_input = [None] * param.n_patterns
    for i in xrange(param.n_patterns):
        pattern_input[i] = spikepattern.Input(param)
        pattern_input[i].build(spike_distrib)
    return pattern_input


def jittered_patterns(pattern_ref, param):
    """
    Copy patterns in pattern_ref and jitter spikes with given noise amplitude
    """
    pattern_input = [None] * param.n_patterns
    for i in xrange(len(pattern_ref)):
        pattern_input[i] = spikepattern.Input(param)
        pattern_input[i].jitter(pattern_ref[i], param.noise_stdev)
    return pattern_input


def target_patterns(param, spike_distrib='uniform'):
    """
    Create a set of unique target spike patterns with spikes following
    spike_distrib
    """
    pattern_target = [None] * param.n_classes
    dist_min = param.n_target_spikes / 2  # Min. separation between spike patterns
    for i in xrange(param.n_classes):
        pattern_target[i] = spikepattern.Target(param)
        while not pattern_target[i].built:
            pattern_target[i].build(spike_distrib)
            # Reject target patterns similar to any previous one
            for j in xrange(i):
                if distmetric.van_rossum_spatio(pattern_target[i].spike_trains,
                                                pattern_target[j].spike_trains) < dist_min:
                    pattern_target[i].built = False
                    break
    return pattern_target
