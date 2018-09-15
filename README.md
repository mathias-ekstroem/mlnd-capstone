# mlnd-capstone

## Pre processing
All feature layers that contain categorical values are embedded into a continuous space
followed by a 1x1 convolution.

Numerical features are logarithmically transformed (minerals and vespene might get much higher values than other
numerical features)


## TODO

- Add replay buffer
- OU noise

```shell
$ python -m pysc2.bin.agent --map CollectMineralShards --agent pysc2.agents.scripted_agent.CollectMineralShards
```

```shell
$ python -m pysc2.bin.play --map Simple64
```
