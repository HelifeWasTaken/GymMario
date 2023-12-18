import numpy as np
from collections import defaultdict
from nes_py import NESEnv

_BOWSER = 0x2D
_FLAGPOLE = 0x31

_STATUS_MAP = defaultdict(lambda: 'fireball', {0:'small', 1: 'tall'})
_BUSY_STATES = [0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x07] 
_ENEMY_TYPE_ADDRESSES = [0x0016, 0x0017, 0x0018, 0x0019, 0x001A]
_STAGE_OVER_ENEMIES = np.array([_BOWSER, _FLAGPOLE])

_SCORE = (0x07DD, 6)
_TIME = (0x07F8, 3)
_COINS = (0x07ED, 2)
_LIFE = (0x075A, 1)

_X_POSITION = (0x86, 1) # Actual X position but scroll adjusted
_X_POSITION_ = (0x06D, 1) # Pixel precision X position - need to be multiplied by 0x100
_X_SCROLL = (0x071, 1) # Current X scroll

_Y_POSITION = (0x03B8, 1) # Y pixel position on screen
_Y_VIEWPORT = (0x00B5, 1) # Current Y viewport

_PLAYER_STATUS = (0x0756, 1) # Player status
_PLAYER_STATE = (0x000E, 1) # Player state

_PLAYER_DYING = 0x0b
_PLAYER_DEAD = 0x06

_GAME_OVER_LIFE = 0xFF

_WORLD_STATE = (0x0770, 1)
_WORLD_STATE_OVER = 0x02

_STAGE_STATE = (0x001D, 1)
_STAGE_STATE_OVER = 0x3

class SMBEnv(NESEnv):

    reward_range = (-15, 15)

    def __init__(self, rom_path):
        self._x_position_last = 0
        self._time_last = 0
        super(SMBEnv, self).__init__(rom_path)
        self.reset()
        self._skip_start_screen()
        self._backup()

    def read_mem_range(self, address, length):
        return int(''.join(map(str, self.ram[address:address + length]))) if length > 1 else self.ram[address]

    @property
    def score(self): return self.read_mem_range(*_SCORE)

    @property
    def time(self): return self.read_mem_range(*_TIME)

    @property
    def coins(self): return self.read_mem_range(*_COINS)

    @property
    def life(self): return self.read_mem_range(*_LIFE)

    @property
    def x_position(self): return 0x100 * self.read_mem_range(*_X_POSITION_) + self.read_mem_range(*_X_POSITION)

    @property
    def x_scroll(self): return np.uint8(int(self.ram[0x86]) - int(self.ram[0x071c])) % 256

    @property
    def y_pixel(self): return self.read_mem_range(*_Y_POSITION)

    @property
    def y_viewport(self): return self.read_mem_range(*_Y_VIEWPORT)

    @property
    def y_position(self): return (255 + 255 - self.y_pixel) if self.y_viewport < 1 else self.y_viewport - self.y_pixel

    @property
    def player_status(self): return _STATUS_MAP[self.read_mem_range(*_PLAYER_STATUS)]

    @property
    def player_state(self): return self.read_mem_range(*_PLAYER_STATE)

    @property
    def is_dying(self): return self.player_state == _PLAYER_DYING

    @property
    def is_dead(self): return self.player_state == _PLAYER_DEAD

    @property
    def is_game_over(self): return self.life == _GAME_OVER_LIFE

    @property
    def is_busy(self): return self.player_state in _BUSY_STATES

    @property
    def is_world_over(self): return self.read_mem_range(*_WORLD_STATE) == _WORLD_STATE_OVER

    @property
    def is_stage_over(self):
        for address in _ENEMY_TYPE_ADDRESSES:
            if self.ram[address] in _STAGE_OVER_ENEMIES:
                return self.read_mem_range(*_STAGE_STATE) == _STAGE_STATE_OVER
        return False

    @property
    def flag_get(self): return self.is_world_over or self.is_stage_over

    @property
    def stage(self): return self.read_mem_range(0x075f, 1)

    @property
    def world(self): return self.read_mem_range(0x075c, 1)

    def _skip_change_area(self):
        change_area_timer = self.ram[0x06DE]
        if change_area_timer > 1 and change_area_timer < 255:
            self.ram[0x06DE] = 1

    def _skip_occupied_states(self):
        while self.is_busy or self.is_world_over:
            self._runout_prelevel_timer()
            self._frame_advance(0)

    def _write_stage(self):
        self.ram[0x075f] = 0
        self.ram[0x075c] = 0
        self.ram[0x0760] = 0

    def _runout_prelevel_timer(self):
        self.ram[0x07A0] = 0

    def _skip_start_screen(self):
        self._frame_advance(8)
        self._frame_advance(0)
        while self.time == 0:
            self._frame_advance(8)
            self._write_stage()
            self._frame_advance(0)
            self._runout_prelevel_timer()
        self._time_last = self.time
        while self.time >= self._time_last:
            self._time_last = self.time
            self._frame_advance(8)
            self._frame_advance(0)

    def _skip_end_of_world(self):
        if self.is_world_over:
            time = self.time
            while self.time == time:
                self._frame_advance(0)

    def _will_reset(self):
        self._time_last = 0
        self._x_position_last = 0

    def _did_reset(self):
        self._time_last = self.time
        self._x_position_last = self.x_position

    def _did_step(self, done):
        if done:
            return
        if self.is_dying:
            self._kill_mario()

        #if not False:
        self._skip_end_of_world()

        self._skip_change_area()
        self._skip_occupied_states()

    def _get_done(self):
        #if False:
        #    return self._is_dying or self._is_dead or self._flag_get
        return self.is_game_over

    def _kill_mario(self):
        self.ram[0x000e] = 0x06
        self._frame_advance(0)

    @property
    def _x_reward(self):
        _reward = self.x_position - self._x_position_last
        self._x_position_last = self.x_position
        if _reward < -5 or _reward > 5:
            return 0
        return _reward

    @property
    def _time_penalty(self):
        _reward = self.time - self._time_last
        self._time_last = self.time
        if _reward > 0:
            return 0
        return _reward

    @property
    def _death_penalty(self): return -25 if self.is_dying or self.is_dead else 0

    def _get_reward(self):
        return self._x_reward + self._time_penalty + self._death_penalty

    def _get_info(self):
        """Return the info after a step occurs"""
        return dict(
            coins=self.coins,
            flag_get=self.flag_get,
            life=self.life,
            score=self.score,
            stage=self.stage,
            status=self.player_status,
            time=self.time,
            world=self.world,
            x_pos=self.x_position,
            y_pos=self.y_position,
        )

    def reset(self, **kwargs):
        obs = super(SMBEnv, self).reset(**kwargs)
        return obs