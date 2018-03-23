# Variational Inference with Normalizing Flows

Reimplementation of Variational Inference with Normalizing Flows (https://arxiv.org/abs/1505.05770)

I got the following results:

<p align="left">
  <img src="/assets/density.png" width="320"/>
  <img src="/assets/flow_2.png" width="320"/>
</p>


<p align="left">
  <img src="/assets/flow_4.png" width="320"/>
  <img src="/assets/flow_16.png" width="320"/>
</p>

### Reproducing my results

To reproduce my results, you will need to install [pytorch](http://pytorch.org/).

Then you will need to install other dependencies from ```requirements.txt```. If you are using ```pip```, simply run ```pip install -r requirements.txt```.

After you have installed the dependencies, run ```python run_experiment.py``` and collect the results in the ```experiments``` folder.
