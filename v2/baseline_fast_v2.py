import sys
from os.path import exists
from pathlib import Path
from red_gym_env_v2 import RedGymEnv
from stream_agent_wrapper import StreamWrapper
from stable_baselines3 import PPO
from stable_baselines3.common import env_checker
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
from tensorboard_callback import TensorboardCallback

from rlhf import  rlhf
import queue
import torch

def make_env(rank, env_conf, seed=0,):
    """
    Utility function for multiprocessed env.
    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environments you wish to have in subprocesses
    :param seed: (int) the initial seed for RNG
    :param rank: (int) index of the subprocess
    """
    def _init():
        env = RedGymEnv(env_conf)

        if 'db_path' in env_conf and env_conf['db_path']:
            db_path = Path(env_conf['db_path'])
            db_path.mkdir(parents=True, exist_ok=True)
            env = rlhf.EpisodeRecorderWrapper(env, db_path, env_idx=rank)


        if 'reward_path' in env_conf and env_conf['reward_path']:
            metrics_queue = queue.Queue(maxsize=0)
            dev = torch.device(env_conf['device'])
            reward_path = Path(env_conf['reward_path'])
            env = rlhf.RewardModelWrapper(env, reward_path, dev=dev, metrics_queue=metrics_queue)

        env = StreamWrapper(
            env, 
            stream_metadata = { 
                "user": "v2-default",
                "env_id": rank, 
                "color": "#447799",
                "extra": "",
            }
        )
        
        env.reset(seed=(seed + rank))
        return env
    set_random_seed(seed)
    return _init

if __name__ == "__main__":


    use_wandb_logging = False
    ep_length = 2048 * 80
    sess_id = "runs"
    sess_path = Path(sess_id)

    env_config = {
                'headless': True, 'save_final_state': False, 'early_stop': False,
                'action_freq': 24, 'init_state': '../has_pokedex_nballs.state', 'max_steps': ep_length, 
                'print_rewards': True, 'save_video': False, 'fast_video': True, 'session_path': sess_path,
                'gb_path': '../PokemonRed.gb', 'debug': False, 'reward_scale': 0.5, 'explore_weight': 0.25,
                'db_path': 'db_rlhf','reward_path': 'rw/reward-v0-rw.dat','device':'cuda','reduced_lr':'0.00003'
            }
    
    print(env_config)
    
    num_cpu = 14 # Also sets the number of episodes per training iteration
    env = SubprocVecEnv([make_env(i, env_config) for i in range(num_cpu)])
    
    checkpoint_callback = CheckpointCallback(save_freq=ep_length//2, save_path=sess_path,
                                     name_prefix="poke")
    
    callbacks = [checkpoint_callback, TensorboardCallback(sess_path)]

    if use_wandb_logging:
        import wandb
        from wandb.integration.sb3 import WandbCallback
        wandb.tensorboard.patch(root_logdir=str(sess_path))
        run = wandb.init(
            project="pokemon-train",
            id=sess_id,
            name="v2-a",
            config=env_config,
            sync_tensorboard=True,  
            monitor_gym=True,  
            save_code=True,
        )
        callbacks.append(WandbCallback())

    #env_checker.check_env(env)

    # put a checkpoint here you want to start from    
    if sys.stdin.isatty():
        file_name = ""
    else:
        file_name = sys.stdin.read().strip() #"runs/poke_26214400_steps"

    train_steps_batch = ep_length // 64
    if 'reward_path' in env_config and env_config['reward_path']:
        lr = float(env_config['reduced_lr'])
        if exists(file_name + ".zip"):
            print("\nloading checkpoint")
            model = PPO.load(file_name, env=env,custom_objects={"learning_rate": lr})
            model.n_steps = train_steps_batch
            model.n_envs = num_cpu
            model.rollout_buffer.buffer_size = train_steps_batch
            model.rollout_buffer.n_envs = num_cpu
            model.rollout_buffer.reset()
            print("learning rate now is :", model.learning_rate)
        else:
            model = PPO("MultiInputPolicy", env, verbose=1, n_steps=train_steps_batch, batch_size=512, n_epochs=1, gamma=0.997, ent_coef=0.01, tensorboard_log=sess_path,learning_rate=lr)
    
    else : 
        if exists(file_name + ".zip"):
            print("\nloading checkpoint")
            model = PPO.load(file_name, env=env)
            model.n_steps = train_steps_batch
            model.n_envs = num_cpu
            model.rollout_buffer.buffer_size = train_steps_batch
            model.rollout_buffer.n_envs = num_cpu
            model.rollout_buffer.reset()
        else:
            model = PPO("MultiInputPolicy", env, verbose=1, n_steps=train_steps_batch, batch_size=512, n_epochs=1, gamma=0.997, ent_coef=0.01, tensorboard_log=sess_path)
            
    
    print(model.policy)

    model.learn(total_timesteps=(ep_length)*num_cpu*10000, callback=CallbackList(callbacks), tb_log_name="poke_ppo")

    if use_wandb_logging:
        run.finish()
