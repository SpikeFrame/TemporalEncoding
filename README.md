# TemporalEncoding
Supervised learning in spiking neural networks for precise temporal encoding

Code written in pyNN for training a single-layer, feed-forward spiking network with all-to-all connectivity to form associations between arbitrary input and target output spike patterns. Full details in: Gardner, B. & Grüning, A. (2016). Supervised learning in spiking neural networks for precise temporal encoding. PLoS ONE 11(8): e0161335. doi:10.1371/journal.pone.0161335.

## Dependencies:
- Python 2.7
- pyNN 0.8.1
- Numpy
- Matplotlib
- nest 2.10.0 (backend simulator used in this case)

## Example usage:
In main_pattern_association.py: Param(1, 1, 4, 200, 1, 200) initialises a network with the following parameters:
- 1 input class
- 1 input pattern assigned to the class
- 4 target spikes assigned to each output neuron
- 200 input spike trains
- 1 output neuron
- 200 learning trials

Simulation is run with random initial network input & target spike times and weight values.

## License
Code released under the GNU General Public License v3.
