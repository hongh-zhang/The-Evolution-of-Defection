from copy import deepcopy
from network import NeuralNetwork
from collections import namedtuple, deque

Transition = namedtuple('Transition', 
                        ('state', 'action', 'next_state', 'reward'))

class DQN():
    
    def __init__(self, layers):
        
        # define networks
        self.policy_net = NeuralNetwork(layers)
        self.target_net = deepcopy(self.policy_net)
    
    def query(self, state):
        """make decision from given state"""
        return self.policy_net(state, mode='classification')
    
    def update_target(self):
        self.target_net = deepcopy(self.policy_net)
        
        
        
    def learn(self, memory, param, verbosity=0):
        
        
        length = len(memory)
        batch_size = param['batch']
        assert length >= batch_size
        
        param['epoch'] += 1
        param['mode'] = 'train'
        self.policy_net.set_loss_func('mse')
        
        
        # get batch
        
        transitions = 
        
        
        
        batch = Transition(*zip(*memory.sample(length)))
        state_batch = np.array(batch.state)
        next_batch = np.array(batch.next_state)

            
        action_batch = batch.action
        action_batch = np.array([[True, False] if a==C else [False, True] for a in action_batch])
        reward_batch = np.array(batch.reward, ndmin=2).T

        # calculate q values
        # Q value = value of current state = value of most suitable action
        Q_values = self.policy_net(state_batch, param=param, mode='rg') * action_batch

        # E(Q value of next state) = reward + value of most suitable action next state
        Q_values_ = np.max(self.target_net(next_batch, mode='rg'), axis=1, keepdims=True)
        E_values = self.gamma*Q_values_ + reward_batch

        # feedback
        loss, _ = self.policy_net.loss_fn(E_values, Q_values)
        loss = loss * action_batch
        if verbosity:
            print(Q_values)
            print(loss)
        if not self.loss:
            self.loss = np.mean(np.max(np.abs(loss),axis=1))
        else:
            self.loss = 0.9*self.loss + 0.1*np.mean(np.max(np.abs(loss),axis=1))  # track training loss
        self.policy_net.backprop(loss, param)