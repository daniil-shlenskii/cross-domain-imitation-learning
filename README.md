# cross-domain-imitation-learning

**Agent training example**
```bash
python train.py --config_path configs/sac_config.yaml -w --from_scratch
python train.py --config_path configs/gail_config.yaml -w --from_scratch
python train.py --config_path configs/dida_config.yaml -w --from_scratch
```

**Agent saving and loading**

During training, the agent is periodically saved.

It saved under the path that set in the config in the following way:
```yaml
archive:
    agent_save_dir: archive/agents/sac
    agent_buffer_save_dir: archive/agents/sac
	...
```
The agent will be stored under the following directory path:
```bash
archive/agents/sac/<env_name>
```

In order to load the agent, form config in the following way:
```yaml
archive:
	...
	agent_load_dir: archive/agents/sac
	agent_buffer_load_dir: archive/agents/sac
```

*NOTE: if those directory paths are not set, the default ones will be used.*

**Agent's trajectories collecting example**
```bash
python -m utils.collect_expert --archive_agent_dir archive/agents/sac/<env_name> --num_episodes 5
```
Collected rollouts with be under the following path:
```bash
archive/agents/sac/<env_name>/collected_rollouts/buffer_state.pickle
```

**Extra elaborations**
1. `--from_scratch`
   If not set,  the training script will try to load the agent
2. `-w`
   Option for removing some Deprecation Warning