import numpy as np
import scipy.special as sp

from IPython.display import display, clear_output
import matplotlib.pyplot as plt
import copy 
import time
import random

import torch
import torch.nn as nn
import torch.optim as optim
import collections

# Necessary imports

import numpy as np
import matplotlib.pyplot as plt
from torch.nn import Linear

from qiskit import QuantumCircuit
#from qiskit.utils import algorithm_globals
from qiskit.circuit import Parameter
from qiskit.circuit.library import RealAmplitudes, ZFeatureMap
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_machine_learning.connectors import TorchConnector

from qiskit.circuit.library import ZZFeatureMap, EfficientSU2

from environment import MazeEnvironment


torch.backends.cudnn.benchmark = True

num_qubits =4

def main():

    
    print("QNN Training Parameters: ", qnn.num_weights)
    print("QNN-Entangled Training Parameters: ", entanglement.num_weights)

    maze_env.draw('./results/maze_10.pdf')


Transition = collections.namedtuple('Experience',
                                    field_names=['state', 'action',
                                                 'next_state', 'reward',
                                                 'is_game_on'])

class ExperienceReplay:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = collections.deque(maxlen=capacity)

    def __len__(self):
        return len(self.memory)

    def push(self, transition):
        self.memory.append(transition)

    def sample(self, batch_size, device = 'cpu'):
        indices = np.random.choice(len(self.memory), batch_size, replace = False)
        
        states, actions, next_states, rewards, isgameon = zip(*[self.memory[idx] 
                                                                for idx in indices])
        
        return torch.Tensor(states).type(torch.float).to(device), \
               torch.Tensor(actions).type(torch.long).to(device), \
               torch.Tensor(next_states).to(device), \
               torch.Tensor(rewards).to(device), torch.tensor(isgameon).to(device)



def create_qnn():
    n_qubits = int(num_qubits/2)

    qc = QuantumCircuit(n_qubits)
    feature_map = ZFeatureMap(n_qubits)
    ansatz = RealAmplitudes(num_qubits=n_qubits, reps=1)

    qc.compose(feature_map, inplace=True)
    qc.compose(ansatz, inplace=True)

    # REMEMBER TO SET input_gradients=True FOR ENABLING HYBRID GRADIENT BACKPROP
    qnn = SamplerQNN(
        circuit=qc,
        input_params=feature_map.parameters,
        weight_params=ansatz.parameters,
        input_gradients=True,
    )
    
    return qnn

def create_entanglement():
    qc = QuantumCircuit(num_qubits)
    feature_map =ZZFeatureMap(num_qubits)
    ansatz = EfficientSU2(num_qubits= num_qubits, reps = 1)

    qc.compose(feature_map, inplace=True)
    qc.compose(ansatz, inplace = True)

    qnn = SamplerQNN(
        circuit=qc,
        input_params=feature_map.parameters,
        weight_params=ansatz.parameters,
        input_gradients=True,
    )
    
    return qnn

qnn = create_qnn()
entanglement = create_entanglement()

class conv_nn(nn.Module):
    
    channels = [16, 32, 64]
    kernels = [3, 3, 3]
    strides = [1, 1, 1]
    in_channels = 1
    
    def __init__(self, rows, cols, n_act):
        super().__init__()
        self.rows = rows
        self.cols = cols

        self.conv_start= nn.Sequential(nn.Conv2d(in_channels = self.in_channels,
                                            out_channels = self.channels[0],
                                            kernel_size = self.kernels[0],
                                            stride = self.strides[0]),
                                  nn.ReLU(),
                                  nn.Conv2d(in_channels = self.channels[0],
                                            out_channels = self.channels[1],
                                            kernel_size = self.kernels[1],
                                            stride = self.strides[1]),
                                  nn.ReLU()
                                 )
        self.conv_finish= nn.Sequential(nn.Conv2d(in_channels = self.in_channels,
                                            out_channels = self.channels[0],
                                            kernel_size = self.kernels[0],
                                            stride = self.strides[0]),
                                  nn.ReLU(),
                                  nn.Conv2d(in_channels = self.channels[0],
                                            out_channels = self.channels[1],
                                            kernel_size = self.kernels[1],
                                            stride = self.strides[1]),
                                  nn.ReLU()
                                 )


        size_out_conv = self.get_conv_size(rows, cols)

        self.linear_start = nn.Sequential(nn.Linear(size_out_conv, 2),
                                    TorchConnector(qnn),
                                    nn.Linear(4, n_act),
                                   )

        self.linear_finish = nn.Sequential(nn.Linear(size_out_conv, 2),
                                    TorchConnector(qnn),
                                    nn.Linear(4, n_act),
                                   )
        
        self.linear_entangle = nn.Sequential(nn.LazyLinear(num_qubits),
                                             TorchConnector(entanglement),
                                             nn.LazyLinear(n_act )
                                             )
        
    def forward(self, x1, x2):

        x1 = x1.view(len(x1), self.in_channels, self.rows, self.cols)
        out_conv_start = self.conv_start(x1).view(len(x1),-1)
        out_lin_start = self.linear_start(out_conv_start)
    

        x2 = x2.view(len(x2), self.in_channels, self.rows, self.cols)
        out_conv_finish = self.conv_finish(x2).view(len(x2),-1)
        out_lin_finish = self.linear_finish(out_conv_finish)

        out_lin = torch.cat((out_lin_start, out_lin_finish))

    

        out_lin = self.linear_entangle(out_lin)
        
        
        return out_lin
    
    def get_conv_size(self, x, y):
        out_conv = self.conv_start(torch.zeros(1,self.in_channels, x, y))
        return int(np.prod(out_conv.size()))
    

    
def get_qvalues(net, state_start, state_finish):
    state_start = torch.Tensor(state_start).to("cpu")
    state_finish = torch.Tensor(state_finish).to("cpu")
    qvalues = net(state_start, state_finish)
    qvalues_start, qvalues_finish = torch.chunk(qvalues, 2, dim =0)
   
    return qvalues_start, qvalues_finish



def plot_policy_map(agent_start, agent_finish, get_qvalues, net, filename, offset):
    with torch.no_grad():
        # Create two independent figures
        fig_start, ax_start = plt.subplots()
        ax_start.imshow(agent_start.env.maze, 'Greys')

        fig_finish, ax_finish = plt.subplots()
        ax_finish.imshow(agent_finish.env.maze, 'Greys')

        for free_cell in agent_start.env.allowed_states:
            agent_start.env.current_position = np.array(free_cell)
            agent_finish.env.current_position = np.array(free_cell)

            # Get Q-values for both agents
            qvalues_start, qvalues_finish = get_qvalues(
                net,
                torch.Tensor(agent_start.env.state()).view(1, -1).to('cpu'),
                torch.Tensor(agent_finish.env.state()).view(1, -1).to('cpu')
            )

            # Get best actions
            action_start = int(torch.argmax(qvalues_start).detach().cpu().numpy())
            policy_start = agent_start.env.directions[action_start]

            action_finish = int(torch.argmax(qvalues_finish).detach().cpu().numpy())
            policy_finish = agent_finish.env.directions[action_finish]

            # Add policy text to both figures
            ax_start.text(free_cell[1] - offset[0], free_cell[0] - offset[1], policy_start)
            ax_finish.text(free_cell[1] - offset[0], free_cell[0] - offset[1], policy_finish)

        # Remove axis ticks
        ax_start.set_xticks([])
        ax_start.set_yticks([])
        ax_finish.set_xticks([])
        ax_finish.set_yticks([])

        # Save images
        plt.figure(fig_start.number)  # Activate first figure
        plt.savefig(f"{filename}_start.png", dpi=300, bbox_inches='tight')

        plt.figure(fig_finish.number)  # Activate second figure
        plt.savefig(f"{filename}_finish.png", dpi=300, bbox_inches='tight')

        # Save images with different filenames
        fig_start.savefig(f"{filename}_start.png", dpi=300, bbox_inches='tight')
        fig_finish.savefig(f"{filename}_finish.png", dpi=300, bbox_inches='tight')
        
        # Show both figures
        plt.show()


def Qloss(batch_start, batch_finish, net, gamma=0.99, device="cpu"):
    #(states, actions, next_states, rewards, _ ), ()= batch

    states_start, actions_start, next_states_start, rewards_start, _ = batch_start

    states_finish, actions_finish, next_states_finish, rewards_finish, _ = batch_finish   
    # Move data for both agents to the desired device
    
    states_start = states_start.to(device)
    actions_start = actions_start.to(device)
    next_states_start = next_states_start.to(device)
    rewards_start = rewards_start.to(device)
    

    states_finish = states_finish.to(device)
    actions_finish = actions_finish.to(device)
    next_states_finish = next_states_finish.to(device)
    rewards_finish = rewards_finish.to(device)
    
    # Convert states and next_states to NumPy arrays
    states_start = states_start.cpu().detach().numpy()
    next_states_start = next_states_start.cpu().detach().numpy()

    states_finish = states_finish.cpu().detach().numpy()
    next_states_finish = next_states_finish.cpu().detach().numpy()

    # Create PyTorch tensors from the NumPy arrays
    states_tensor_start = torch.from_numpy(states_start).to(device)
    next_states_tensor_start = torch.from_numpy(next_states_start).to(device)

    states_tensor_finish = torch.from_numpy(states_finish).to(device)
    next_states_tensor_finish = torch.from_numpy(next_states_finish).to(device)


    # Perform the rest of the computation on the device
    lbatch_start = len(states_tensor_start)
    lbatch_finish = len(states_tensor_finish)

   
    #####################################

    state_action_values_start, state_action_values_finish  = get_qvalues(net,states_tensor_start.view(lbatch_start,-1), states_tensor_finish.view(lbatch_finish,-1))
    #state_action_values_start = state_action_values_start.transpose(0,1)
    
    #state_action_values_start = state_action_values_start.view(lbatch_start, 1)
    #state_action_values_start = state_action_values_start.view(-1, 1)
    
    state_action_values_start = state_action_values_start.gather(1, actions_start.unsqueeze(-1))
    state_action_values_start = state_action_values_start.squeeze(-1)

    state_action_values_finish = state_action_values_finish.gather(1, actions_finish.unsqueeze(-1))
    state_action_values_finish = state_action_values_finish.squeeze(-1)

    state_action_values_combined = torch.cat((state_action_values_start, state_action_values_finish))

    #####################################

    next_state_values_start, next_state_values_finish  = get_qvalues(net,next_states_tensor_start.view(lbatch_start, -1), next_states_tensor_finish.view(lbatch_finish, -1))


    next_state_values_start = next_state_values_start.max(1)[0]
    next_state_values_start = next_state_values_start.detach()
   
    expected_state_action_values_start = next_state_values_start * gamma + rewards_start


    next_state_values_finish = next_state_values_finish.max(1)[0]
    next_state_values_finish = next_state_values_finish.detach()

    
    expected_state_action_values_finish = next_state_values_finish * gamma + rewards_finish

    expected_state_action_values_combined = torch.cat((expected_state_action_values_start, expected_state_action_values_finish))


    return nn.MSELoss()(state_action_values_combined, expected_state_action_values_combined)


maze = np.load('./maze_generator/maze_10x10.npy')

start_position = [0,0]
end_position = [len(maze)-1, len(maze)-1]

maze_env = MazeEnvironment(maze, start_position, end_position)

maze_finish = np.load('./maze_generator/maze_10x10.npy')
maze_env_finish = MazeEnvironment(maze_finish, end_position, start_position)

main()