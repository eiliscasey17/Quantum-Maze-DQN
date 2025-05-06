import numpy as np
import scipy.special as sp
import matplotlib.pyplot as plt
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import collections

Transition = collections.namedtuple('Experience',
                                    field_names=['state', 'action',
                                                 'next_state', 'reward',
                                                 'is_game_on'])


class Agent:
    def __init__(self, maze, memory_buffer, use_softmax = True):
        self.env = maze
        self.buffer = memory_buffer # this is actually a reference
        self.num_act = 5
        self.use_softmax = use_softmax
        self.total_reward = 0
        self.min_reward = -self.env.maze.size
        self.isgameon = True

       
      


        
    def make_a_move(self, qvalues, epsilon, state_update, action, current_state, device = 'cpu'):
        #action = self.select_action(qvalues, epsilon, device)
        #current_state = self.env.state()
        #next_state, reward, self.isgameon = self.env.state_update(action)
        next_state, reward, self.isgameon = state_update

        self.total_reward += reward
        
        if self.total_reward < self.min_reward:
            self.isgameon = False
        if not self.isgameon:
            self.total_reward = 0
        
        transition = Transition(current_state, action,
                                next_state, reward,
                                self.isgameon)
        
        self.buffer.push(transition)
            

    def select_action(self, qvalues, epsilon, device = 'cpu'):
        
        qvalues = qvalues.detach().cpu().numpy()

        # softmax sampling of the qvalues
        if self.use_softmax:
            p = sp.softmax(qvalues/epsilon).squeeze()
            p /= np.sum(p)
            action = np.random.choice(self.num_act, p = p)
            
        # else choose the best action with probability 1-epsilon
        # and with probability epsilon choose at random
        else:
            if np.random.random() < epsilon:
                action = np.random.randint(self.num_act, size=1)[0]
            else:         
                action = np.argmax(qvalues)
                action = int(action)
        
        return action
    
    def save_model(self, net, filename):
        # Save the model under the subfolder 'models'
        torch.save(net.state_dict(), filename)
    
    def plot_policy_map(self, get_qvalues ,net, filename, offset):
        # Load the model

        with torch.no_grad():
            fig, ax = plt.subplots()
            ax.imshow(self.env.maze, 'Greys')

            for free_cell in self.env.allowed_states:
                self.env.current_position = np.array(free_cell)
                qvalues = 0
                if self.agent_id == 'Start':
                    qvalues, _ = get_qvalues(net, torch.Tensor(self.env.state()).view(1,-1).to('cpu'),torch.Tensor(self.env.state()).view(1,-1).to('cpu'))
                else:
                    _, qvalues = get_qvalues(net, torch.Tensor(self.env.state()).view(1,-1).to('cpu'),torch.Tensor(self.env.state()).view(1,-1).to('cpu'))
                action = int(torch.argmax(qvalues).detach().cpu().numpy())
                policy = self.env.directions[action]

                ax.text(free_cell[1]-offset[0], free_cell[0]-offset[1], policy)
            ax = plt.gca()

            plt.xticks([], [])
            plt.yticks([], [])

            #ax.plot(self.env.goal[1], self.env.goal[0],
            #        'bs', markersize = 4)
            plt.savefig(filename, dpi = 300, bbox_inches = 'tight')
            plt.show()

 
