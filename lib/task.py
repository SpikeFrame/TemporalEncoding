#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  2 17:42:58 2016

@author: Brian

Definition of learning tasks: a for container for input[, target] patterns and,
if applicable, their class labels.

The following spike timing distributions for generated patterns are supported:
    - 'uniform'
    - 'poisson'
"""

from __future__ import division
import numpy as np

from lib import patterngen


class GenericTask(object):
    """
    General learning task and methods for creating typical patterns
    """
    def __init__(self, param):
        self.param = param  # Internal reference to parameter set

        self.built = False

    def build_static_patterns(self, spike_distrib=None):
        """
        Create noiseless input[, target] patterns.
        Overwritten by subclass.
        """
        pass

    def build_dynamic_patterns(self, spike_distrib=None):
        """
        Create input patterns subject to noise between trials.
        Overwritten by subclass.
        """
        pass

    def jittered_inputs(self):
        """
        Create jittered version of reference input patterns
        """
        if not hasattr(self, 'pattern_input_ref'):
            return
        self.pattern_input = patterngen.jittered_patterns(self.pattern_input_ref, self.param)


class PatternAssociation(GenericTask):
    """
    Arbitrary input-output spike pattern association task.
    """
    def assign_classes(self):
        # Assign class labels to input patterns
        self.labels = np.floor(np.arange(self.param.n_patterns) /
                               self.param.n_patterns_class)
        self.labels = self.labels.astype(int)

    def build_static_patterns(self, spike_distrib='uniform'):
        """Build noiseless input / target patterns"""
        if self.built:
            return
        # Initialise input patterns
        self.pattern_input = patterngen.input_patterns(self.param,
                                                       spike_distrib)
        self.assign_classes()
        # Initialise target patterns
        self.pattern_target = patterngen.target_patterns(self.param,
                                                         spike_distrib)
        self.built = True

    def build_dynamic_patterns(self, spike_distrib='uniform'):
        """Create reference input patterns to generate jittered copies"""
        if self.built:
            return
        # Initialise reference input patterns
        self.pattern_input_ref = patterngen.input_patterns(self.param,
                                                           spike_distrib)
        self.assign_classes()
        # Initialise (fixed) target patterns
        self.pattern_target = patterngen.target_patterns(self.param,
                                                         spike_distrib)
        self.built = True


class Mimicry(GenericTask):
    """
    Input patterns are to be mapped to the output of a tutor neuron
    """
    def build_static_patterns(self, spike_distrib='uniform'):
        """Build noiseless input / target patterns"""
        if self.built:
            return
        # Initialise input patternss
        self.pattern_input = patterngen.input_patterns(self.param,
                                                       spike_distrib)
        self.built = True

    def build_dynamic_patterns(self, spike_distrib='uniform'):
        """Create reference input patterns to generate jittered copies"""
        if self.built:
            return
        # Initialise reference input patterns
        self.pattern_input_ref = patterngen.input_patterns(self.param,
                                                           spike_distrib)
        self.built = True
