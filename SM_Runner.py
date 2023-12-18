from gym.envs.registration import register as gym_register
from gym import make as gym_make
from gym.wrappers import GrayScaleObservation
from JoypadSpace import JoypadSpace

from os import makedirs
from os import listdir
from os.path import join as join_path
from os.path import exists as path_exists

from time import strftime, time
from datetime import datetime

from random import choice as random_choice

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback


class _TrainAndLoggingCallback(BaseCallback):

    def __init__(self, check_freq, save_path, verbose=1):
        super(_TrainAndLoggingCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path

    def _init_callback(self):
        if self.save_path is not None:
            makedirs(self.save_path, exist_ok=True)

    def _on_step(self):
        if self.n_calls % self.check_freq == 0:
            model_path = join_path(self.save_path, 'best_model_{}'.format(self.n_calls))
            self.model.save(model_path)
        return True

class SM_Runner:

    _DEFAULT_MOVEMENT = [ ['NOOP'], ['right'], ['right', 'A'], ['right', 'B'], ['right', 'A', 'B'], ['A'], ['left'] ]

    def _register_mario_env(self, rom_location, entry_point, downsample, rectangle, version, **kwargs):
        if isinstance(version, str) and version != '': version = int(version)
        elif version == '': version = 1

        s = 'super-mario-bros'
        if version != 1: s += '-' + str(version)
        if downsample: s += '-downsample'
        if rectangle: s += '-rectangle'
        s += '.nes'

        kwargs.update({'rom_path': join_path(rom_location, s)})
        gym_register(
            id=self.env_name,
            entry_point=entry_point,
            max_episode_steps=9_999_999,
            reward_threshold=9_999_999,
            kwargs=kwargs,
            nondeterministic=True
        )

    def __init__(self, 
                rom_location='./roms', entry_point='SMEnv:SMBEnv', downsample=False, rectangle=False, version='',
                env_name='SMBEnv',
                movements=[], check_freq=10000, learning_rate=1e-6, verbose=1, total_timesteps=1_000_000, n_steps=512, model_path=None,
                n_envs=1, use_subproc=False,
                **kwargs):
        _DATE = strftime("%y_%m_%d_%H_%M_%S")
        self.log_dir = join_path('./logs/', _DATE)
        self.checkpoint_dir = join_path('./train', _DATE)

        self.movements = SM_Runner._DEFAULT_MOVEMENT if movements == [] else movements
        self.check_freq = check_freq
        self.learning_rate = learning_rate
        self.verbose = verbose
        self.total_timesteps = total_timesteps
        self.n_steps = n_steps

        self.model_path = ('model' + '_' + _DATE + '.pkl') if model_path is None else model_path
        self.env_name = env_name
        self.env = None
        self.n_envs = n_envs
        self.use_subproc = use_subproc

        self.rom_location = rom_location
        self.entry_point = entry_point
        self.downsample = downsample
        self.rectangle = rectangle
        self.version = version
        self.kwargs = kwargs

    def make_env(self, obs=True):
        if self.env is not None:
            self.env.close()

        def _make_env(obs=True):
            rqgy = obs
            def _init():
                self._register_mario_env(self.rom_location, self.entry_point, self.downsample, self.rectangle, self.version, **self.kwargs)
                env = gym_make(self.env_name)
                env = JoypadSpace(env, self.movements)
                if rqgy:
                    env = GrayScaleObservation(env, keep_dim=True)
                return env
            return _init

        if not obs:
            print("Creating playing environment")
            self.env = DummyVecEnv([_make_env(False) for _ in range(self.n_envs)])
            return self.env

        print("Creating observable environment")
        def _gen_env():
            return [_make_env() for _ in range(self.n_envs)]
        def _gen_env_subproc():
            return SubprocVecEnv(_gen_env())
        def _gen_env_dummy():
            return DummyVecEnv(_gen_env())

        self.env = _gen_env_subproc() if self.use_subproc else _gen_env_dummy()
        self.env = VecFrameStack(self.env, 4, channels_order='last')
        return self.env

    def get_env(self):
        if self.env is None:
            return self.make_env()
        return self.env

    def play(self, model=None):
        self.make_env(False)

        env = self.get_env()
        model = PPO.load(self.model_path) if model is None else model
        state = env.reset()
        while True:
            action, _states = model.predict(state)
            state, reward, done, info = env.step(action)
            env.render()
            if done.any():
                env.reset()

    def record(self, model=None):
        self.make_env()

        env = self.get_env()
        model = PPO.load(self.model_path) if model is None else model
        states = env.reset()

        actions = [[] for _ in range(env.num_envs)]
        while True:
            actions_, _states = model.predict(states)
            for i, action in enumerate(actions_):
                actions[i].append(action)
            states, rewards, dones, infos = env.step(actions_)
            for i, done in enumerate(dones):
                if not done:
                    continue
                world, level, x_position = infos[i]['world'], infos[i]['stage'], infos[i]['x_pos']
                print(f"Environment {i} considered done at level {world}-{level} with distance {x_position} and reward {rewards[i]}")
                if world > 0 or level > 0 or x_position > 2600:
                    date = datetime.now().strftime("%y_%m_%d_%H_%M_%S")
                    record_path = join_path('./records', f'record_{date}_level_{world}_{level}_distance_{x_position}_reward_{rewards[i]}.meta')
                    makedirs('./records', exist_ok=True)
                    with open(record_path, 'w') as f:
                        actions[i] = list(map(lambda x: [x], actions[i]))
                        f.write("[" + ", ".join(map(str, actions[i])) + "]")
                    print(f"Recorded {len(actions[i])} actions for environment {i}")
                actions[i] = []
            if dones.all():
                env.reset()
                print('Environment reset!')

    def play_record(self, fps=60):
        self.make_env(False)
        while True:
            try:
                validate = lambda f: f.startswith('record') and f.endswith('.meta')
                record_base = './records'
                record_path = join_path(record_base, random_choice(list(filter(validate, listdir(record_base)))))

                if not path_exists(record_path):
                    raise Exception("No record found")

                print(record_path)
                with open(record_path, 'r') as f:
                    env = self.get_env()
                    state = env.reset()
                    actions = eval(f.read())
                current_time = time()
                def _frame_advance():
                    nonlocal current_time
                    target_time = current_time + 1/fps
                    while time() < target_time: pass
                    current_time = time()

                _frame_advance()
                for action in actions:
                    state, reward, done, info = env.step(action)
                    _frame_advance()
                    env.render()
            except Exception as e:
                print(e)
                #with open(record_path, 'w') as f:
                #    actions = map(lambda x: [x], actions)
                #    f.write("[" + ", ".join(map(str, actions)) + "]")

    def train(self):
        self.make_env()

        callback = _TrainAndLoggingCallback(check_freq=self.check_freq, save_path=self.checkpoint_dir)
        model = PPO('CnnPolicy', self.get_env(), verbose=self.verbose, tensorboard_log=self.log_dir, learning_rate=self.learning_rate, n_steps=self.n_steps)
        model.learn(total_timesteps=self.total_timesteps, callback=callback)
        model.save(self.model_path)
        return self.env, model

def parse_sm_runner_arguments(args=None):
    from argparse import ArgumentParser
    parser = ArgumentParser() if args is None else ArgumentParser(args)
    parser.add_argument('--rom_location', type=str, default='./roms/')
    parser.add_argument('--entry_point', type=str, default='SMEnv:SMBEnv')
    parser.add_argument('--downsample', type=bool, default=False)
    parser.add_argument('--rectangle', type=bool, default=False)
    parser.add_argument('--version', type=int, default=1)
    parser.add_argument('--env_name', type=str, default='SMBEnv')
    #parser.add_argument('--movements', type=list, default=[])
    parser.add_argument('--check_freq', type=int, default=10000)
    parser.add_argument('--learning_rate', type=float, default=1e-6)
    parser.add_argument('--verbose', type=int, default=1)
    parser.add_argument('--total_timesteps', type=int, default=1_000_000)
    parser.add_argument('--n_steps', type=int, default=512)
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--n_envs', type=int, default=1)
    parser.add_argument('--train_use_subproc', type=bool, default=False)
    args = parser.parse_args()
    return args

def make_sm_runner_from_args(args=None):
    args = parse_sm_runner_arguments(args)
    return SM_Runner(
        rom_location=args.rom_location,
        entry_point=args.entry_point,
        downsample=args.downsample,
        rectangle=args.rectangle,
        version=args.version,
        env_name=args.env_name,
        #movements=args.movements,
        movements=[],
        check_freq=args.check_freq,
        learning_rate=args.learning_rate,
        verbose=args.verbose,
        total_timesteps=args.total_timesteps,
        n_steps=args.n_steps,
        model_path=args.model_path,
        n_envs=args.n_envs,
        use_subproc=args.train_use_subproc
    )