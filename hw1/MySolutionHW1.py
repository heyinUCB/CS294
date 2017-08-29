import tensorflow as tf
import tf_util
import numpy as np
import gym
import load_policy
# learning_rate = 

# class Policies():
# 	def __init__(self, env):
# 		optimizer = tf.train.AdamOptimizer(learning_rate)

def get_training_data(num_running, stop_iter, policy_fn, env):
	with tf.Session():
		tf_util.initialize() # TODO: try to understand how does initialize work

		reward_total_list = []
		observs = []
		actions = []
		for _ in range(num_running):
			obs = env.reset()
			done = False
			reward_total = 0
			num_iter = 0
			while not done:
				action = policy_fn(obs[None, :])
				# append the last observation caused by last action
				observs.append(obs)
				actions.append(action)
				obs, reward, done, info = env.step(action)
				reward_total += reward
				num_iter += 1
				env.render()
				if num_iter >= stop_iter:
					break
			reward_total_list.append(reward_total)
		train_data_expert = {'observations': np.stack(observs, axis = 0), 
		                     'actions': np.squeeze(np.stack(actions, axis = 0)),
		                     'reward_total_list': np.array(reward_total_list)}
        return train_data_expert

def Behavioral_cloning(num_running = 2, stop_iter = 200, env_name = 'Hopper-v1', expert_policy_file = 'experts/Hopper-v1.pkl'):
	expert_policy = load_policy.load_policy(expert_policy_file)
	env = gym.make(env_name)
	training_data = get_training_data(num_running, stop_iter, expert_policy, env)



