# Basic Q Learning example without a policy
# Credits: http://mnemstudio.org/path-finding-q-learning-tutorial.htm

import numpy as np

reward_matrix = np.loadtxt('environment.csv', delimiter = ',')
Q_matrix = np.zeros((6, 6))
states = np.array(range(6))
actions = np.array(range(6))
num_episodes = 1000
gamma = 0.8
alpha = 0.3
epsilon = 0.2

for episode in range(num_episodes):
	print(episode)
	initial_state = np.random.choice(states)
	current_state = initial_state
	while current_state != 5:
		available_actions = list({i: j for i, j in enumerate(reward_matrix[current_state]) if j != -1}.keys())
		possible_rewards = {action: round(Q_matrix[current_state][action]) for action in available_actions}
		best_reward = max(possible_rewards.values())
		if np.random.random() <= epsilon:
			action = np.random.choice(available_actions)
		else:
			if list(possible_rewards.values()).count(best_reward) == 1:
				action = max(possible_rewards, key = possible_rewards.get)
			else:
				action = np.random.choice([action for action in available_actions if possible_rewards[action] == best_reward]) 
		possible_actions = list({i: j for i, j in enumerate(reward_matrix[action]) if j != -1}.keys())
		maxq = max([Q_matrix[action][possible_action] for possible_action in possible_actions])
		Q_matrix[current_state][action] = alpha * (Q_matrix[current_state][action]) + (1 - alpha) * (reward_matrix[current_state][action] + gamma * maxq)
		current_state = action
	print(Q_matrix)
