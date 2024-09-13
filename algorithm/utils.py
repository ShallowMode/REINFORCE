import random
import numpy as np

from collections import namedtuple, deque

def epi_done(done):
    '''
    - 단일 환경 : done 값은 True 또는 False로 표시됨.
    - 벡터화된 환경 : done 은 여러 개의 값을 포함한 배열 또는 리스트로 나타날 수 있다.
    벡터화된 환경에서는 모든 에피소드가 동시에 끝나지 않기 때문에 자연스러운 종료 경계가 없다.
    -np.isscalar(done) = 스칼라 값, 즉 단일값인지 확인
    -and done = np.isscalar(done)이 True 일 때만 done 값 반환
    -즉, done이 스칼라이면서 동시에 True 일때만 최종적으로 True 반환
    '''
    return np.isscalar(done) and done

class OnpolicyReplay:
    def __init__(self):
        super(OnpolicyReplay, self).__init__()
        self.is_episodic = True
        self.size = 0
        self.seen_size = 0
        self.training_frequency = 300
        self.data_keys = ['states', 'actions', 'rewards', 'next_states', 'dones', 'next_actions']
        self.reset()

    def reset(self):
        for k in self.data_keys:
            setattr(self, k, [])
        self.cur_epi_data = {k: [] for k in self.data_keys}
        self.most_recent = (None,) * len(self.data_keys)
        self.size = 0

    def update(self, state, action, reward, next_state, done):
        self.add_experience(state, action, reward, next_state, done)
    
    def add_experience(self, state, action, reward, next_state, done, next_action=None):
        self.most_recent = (state, action, reward, next_state, done, next_action)
        for idx, k in enumerate(self.data_keys):
            self.cur_epi_data[k].append(self.most_recent[idx])
        if epi_done(done):
            for k in self.data_keys:
                getattr(self, k).append(self.cur_epi_data[k])
            self.cur_epi_data = {k:[] for k in self.data_keys}
            if len(self.states) == self.training_frequency:
                self.to_train = 1
    
    def sample(self):
        batch = {k: getattr(self, k) for k in self.data_keys}
        self.reset()
        # 'next_actions'가 None일 경우 적절한 기본값으로 설정
        if 'next_actions' in batch:
            if batch['next_actions'] is None:
                batch['next_actions'] = [0] * len(batch['states'])
        return batch

class OnpolicyBatchReplay(OnpolicyReplay):
    def __init__(self):
        super(OnpolicyBatchReplay, self).__init__()
        self.is_episodic = False
    
    def add_experience(self, state, action, reward, next_state, done, next_action=None):
        # next_action이 제공되지 않았을 때 기본값을 설정
        if next_action is None:
            next_action = 0  # 적절한 기본값 설정
        self.most_recent = [state, action, reward, next_state, done, next_action]
        for idx, k in enumerate(self.data_keys):
            getattr(self, k).append(self.most_recent[idx])
        self.size += 1
        self.seen_size += 1
        if len(self.states) == self.training_frequency:
            self.to_train = 1
    
    def sample(self):
        batch = super().sample()
        # 'next_actions'가 None일 경우 적절한 기본값으로 설정
        if 'next_actions' in batch:
            if batch['next_actions'] is None:
                batch['next_actions'] = [0] * len(batch['states'])
        return batch

class ReplayMemory:
    def __init__(self, max_size, Transition):
        self.max_size = max_size
        self.Transition = Transition
        self.memory = deque(maxlen = max_size)

    def push(self, *args):
        self.memory.append(self.Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)