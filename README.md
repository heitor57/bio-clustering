# Solving Clustering with Bioinspired Algorithms

## Genetic Algorithm

### Configuration

Just edit src/config.yaml for settings.

### Usage

You can see the usage passing the argument --help in ga.py

```sh
usage: ga.py [-h] [--config_file CONFIG_FILE]

optional arguments:
  -h, --help            show this help message and exit
  --config_file CONFIG_FILE, -c CONFIG_FILE
                        Configuration file.
```

## Particle Swarm Optimization

### Configuration

Just edit src/config.yaml for settings.

### Usage

You can see the usage passing the argument --help in pso.py
```sh
usage: pso.py [-h] [--config_file CONFIG_FILE]

optional arguments:
  -h, --help            show this help message and exit
  --config_file CONFIG_FILE, -c CONFIG_FILE
                        Configuration file.
```
## Settings

Change the yaml config file to change policies, variables and others things of algorithms.

```yaml
cross_policy:
  TwoPointCross: null
  PointCross: null
  BLXa:
    alpha: 0.5
  BLXab:
    alpha: 0.75
    beta: 0.25
algorithms:
  ga:
    mutation_policy: UniformRandomSolutionSpace
    cross_policy: BLXab
    selection_policy: Roulette
    population_size: 50
    num_generations: 100
    cross_rate: 1
    elitism: true
    mutation_rate: 0.001
  pso:
    w: 0.4
    c_1: 0.5
    c_2: 0.5
    num_particles: 50
    num_iterations: 100
    topology: VonNeumannTopology
general:
  logging_level: INFO
parameters:
  instance_name: iris.data
  eid: 1
```
## Experiments

To execute factorial experiments, we use these following commands:

```sh
pso_make_configs.py
pso_grid.py
pso_grid_plot.py
pso_plot_mean_and_best.py
ga_make_configs.py
ga_grid.py
ga_grid_plot.py
ga_plot_mean_and_best.py
```
