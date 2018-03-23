# Variational Inference with Normalizing Flows

Reimplementation of Variational Inference with Normalizing Flows (https://arxiv.org/abs/1505.05770)

The idea is to approximate a complex multimodal probability density with a simple probability density followed by a sequence of invertible nonlinear transforms. Inference in such model requires a computation of multiple Jacobian determinants, which can be computationaly expensive. Authors propose a specific form of the transformation that reduces the cost of computing the Jacobians from approximately ![](/assets/cubic_d.svg) to ![](/assets/linear_d.svg) where ![](/assets/simple_d.svg) is the dimensionality of the data.

I got the following results:

<p align="left">
  <img src="/assets/density.png" width="320"/>
  <img src="/assets/flow_2.png" width="320"/>
</p>


<p align="left">
  <img src="/assets/flow_4.png" width="320"/>
  <img src="/assets/flow_16.png" width="320"/>
</p>

As can be seen, the approximation quality indeed increases as flow length gets higher.

### Reproducing my results

To reproduce my results, you will need to install [pytorch](http://pytorch.org/).

Then you will need to install other dependencies from ```requirements.txt```. If you are using ```pip```, simply run ```pip install -r requirements.txt```.

After you have installed the dependencies, run ```python run_experiment.py``` and collect the results in the ```experiments``` folder.
