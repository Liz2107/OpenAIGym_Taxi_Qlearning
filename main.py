import gym
import numpy as np
import random
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline

# e: env variable used, same as the training variable
# q_table: stored q_table
#
def evaluate_policy(e, q_table):
    total_rewards = []
    
    for i in range(100):
        state = e.reset()[0]
        total_reward = 0
        done = False
        
        while not done:
            action = np.argmax(q_table[state])
            next_state, reward, terminated, truncated , info = e.step(action)

            done = truncated or terminated
            total_reward += reward
        
        total_rewards.append(total_reward)
    
    average_reward = np.mean(total_rewards)
    # print(f"Average reward over 100 episodes: {average_reward}")





def render_policy(e, q_table, num_episodes):
    rewards = []
    for _ in range(num_episodes):
        state = e.reset()[0]
        done = False
        total_r = 0
        while not done:
            e.render()  # Render the environment
            action = np.argmax(q_table[state])  # Choose the best action
            state, reward, t, tt, _ = e.step(action)
            done = t or tt
            # print(f"State: {state}, Action: {action}, Reward: {reward}")
            total_r += reward
        rewards.append(total_r)
        # print("Episode finished. Total reward: ", total_r, ".")
    print(f"Average Reward: {np.mean(rewards)}")
    e.close()


def run_taxi_random(n_episodes, render_mode):
    if render_mode == "h":
        e = gym.make('Taxi-v3', render_mode='human')
    else:
        e = gym.make('Taxi-v3', render_mode='rgb_array')
        
    rewards_to_plot = []
    
    for episode in range(n_episodes):
        state = e.reset()
        total_reward = 0
        state = state[0]
        step_max = 100
        for step in range(step_max):
            action = e.action_space.sample() 
            
            next_state, reward, terminated, truncated , info = e.step(action)
            done = truncated or terminated
            
            state = next_state
            total_reward += reward
            
            if done:
                break
        
        rewards_to_plot.append(total_reward)
        # print(episode, reward)

    print("Training finished.")
    plt.plot(rewards_to_plot, label="Learning Curve")
    plt.title("Rewards over time for Random Policy")
    plt.xlabel("Episodes")
    plt.ylabel("Reward")
    plt.show()
    # won't evaluate the policy bc it will always score very badly, and we can just see this from the graph. We can't really calculate how "bad" it is because it goes on for too long to be worth waiting for

# alpha: learning rate
# gamma: discount factor
# epsilon: exploration rate
# min_epsilon: point at which to stop decay
# decay_rate: how fast to decay epsilon, decided to just use a standard number here for easiest tuning
# n_episodes: how many training episodes there will be, should be lower if using human rendering
# render_mode: use 'h' if you want to actually see the results, but it will be very slow watching the training

def run_taxi_epsilon_greedy(alpha, gamma, epsilon, min_epsilon, decay_rate, n_episodes, render_mode):
    epsilon_over_time = [epsilon]
    if render_mode == "h":
        e = gym.make('Taxi-v3', render_mode='human')
    else:
        e = gym.make('Taxi-v3', render_mode='rgb_array')
        
    q_table = np.zeros((e.observation_space.n, e.action_space.n))
    rewards_to_plot = []
    for episode in range(n_episodes):
        state = e.reset()
        total_reward = 0
        state = state[0]
        step_max = 100
        for step in range(step_max):
            if random.uniform(0, 1) < epsilon:
                action = e.action_space.sample() 
            else:
                action = np.argmax(q_table[state])        
            
            next_state, reward, terminated, truncated , info = e.step(action)

            done = truncated or terminated
            best_next_action = np.argmax(q_table[next_state])
            td_target = reward + gamma * q_table[next_state][best_next_action]
            # print(step,state)

            td_error = td_target - q_table[state][action]
            q_table[state][action] += alpha * td_error
            
            total_reward += reward
            state = next_state
            
            if done:
                break
        
        epsilon = max(min_epsilon, epsilon * decay_rate)
        epsilon_over_time.append(epsilon)
        rewards_to_plot.append(total_reward)
        # print(episode, reward)

    print("Training finished.")

    # Plot learning curve
    fit = np.polyfit(np.linspace(0,n_episodes,n_episodes),rewards_to_plot, 7)
    curve1 = [fit[0]*r**7 + fit[1]*r**6 + fit[2]*r**5 + fit[3]*r**4 + fit[4]*r**3 + fit[5]*r**2 + fit[6]*r + fit[7] for r in range(n_episodes)]
    # curve = make_interp_spline(np.linspace(0,n_episodes,n_episodes), rewards_to_plot)
       
    
    fig, ax1 = plt.subplots()

    color = 'tab:blue'
    ax1.set_xlabel('time (s)')
    ax1.set_ylabel('Total Reward', color=color)
    ax1.plot(rewards_to_plot, color=color, label="Learning Curve")
    ax1.plot(curve1, color='tab:red', label="Learning Trend")
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second Axes that shares the same x-axis

    color = 'tab:green'
    ax2.set_ylabel("Epsilon", color=color)  # we already handled the x-label with ax1
    ax2.plot(epsilon_over_time, color=color, label="Epsilon")
    ax2.tick_params(axis='y', labelcolor=color)
    
    plt.title('Learning Curve')
    ax1.legend()
    ax2.legend()
    fig.tight_layout()  
    plt.show()

    np.save("q_table.npy", q_table)
    
    evaluate_policy(e, q_table)
    render_policy(e, q_table, 3)


def run_taxi_always_explore(alpha, gamma, n_episodes, render_mode):
    if render_mode == "h":
        e = gym.make('Taxi-v3', render_mode='human')
    else:
        e = gym.make('Taxi-v3', render_mode='rgb_array')
        
    q_table = np.zeros((e.observation_space.n, e.action_space.n))
    rewards_to_plot = []
    for episode in range(n_episodes):
        state = e.reset()
        total_reward = 0
        state = state[0]
        step_max = 100
        for step in range(step_max):
            action = e.action_space.sample() 
            
            next_state, reward, terminated, truncated , info = e.step(action)

            done = truncated or terminated
            best_next_action = np.argmax(q_table[next_state])
            td_target = reward + gamma * q_table[next_state][best_next_action]
            # print(step,state)

            td_error = td_target - q_table[state][action]
            q_table[state][action] += alpha * td_error
            
            total_reward += reward
            state = next_state
            
            if done:
                break
        
        
        rewards_to_plot.append(total_reward)
        # print(episode, reward)

    print("Training finished.")

    # Plot learning curve
    fit = np.polyfit(np.linspace(0,n_episodes,n_episodes),rewards_to_plot, 7)
    curve1 = [fit[0]*r**7 + fit[1]*r**6 + fit[2]*r**5 + fit[3]*r**4 + fit[4]*r**3 + fit[5]*r**2 + fit[6]*r + fit[7] for r in range(n_episodes)]
    plt.plot(rewards_to_plot, label="Learning Curve")
    plt.plot(curve1, label="Learning Trend")
    plt.title("Rewards Over Time")
    plt.xlabel("Episodes")
    plt.ylabel("Reward")
    plt.legend()
    plt.show()
    
    evaluate_policy(e, q_table)
    render_policy(e, q_table, 3)

# Set some basic params. We can refine these and explore more combinations as we go

alpha = 0.1
gamma = 0.9
epsilon = 1.0
min_epsilon = 0.05
decay_rate = 0.995
episodes = 2000
render_mode = 'r'

#First, just take a look at what the random policy looks like
# run_taxi_random(episodes, render_mode)

#Basic test: use the params above to complete a test. We will check the paramters and various approaches later
# run_taxi_epsilon_greedy(alpha, gamma, epsilon, min_epsilon, decay_rate, episodes, render_mode)

# alpha test: Run 6 tests for alpha values of 0.05, 0,1, 0.2, 0.4, 0.5, 0.75

# run_taxi_epsilon_greedy(0.05, gamma, epsilon, min_epsilon, decay_rate, episodes, render_mode)
# run_taxi_epsilon_greedy(0.1, gamma, epsilon, min_epsilon, decay_rate, episodes, render_mode)
# run_taxi_epsilon_greedy(0.2, gamma, epsilon, min_epsilon, decay_rate, episodes, render_mode)
# run_taxi_epsilon_greedy(0.4, gamma, epsilon, min_epsilon, decay_rate, episodes, render_mode)
# run_taxi_epsilon_greedy(0.5, gamma, epsilon, min_epsilon, decay_rate, episodes, render_mode)
# run_taxi_epsilon_greedy(0.75, gamma, epsilon, min_epsilon, decay_rate, episodes, render_mode)

# Result: best option is alpha = 0.5
alpha = 0.5

# gamma test: Run the same style of test, but with changing gamma values
# run_taxi_epsilon_greedy(alpha, 0.5, epsilon, min_epsilon, decay_rate, episodes, render_mode)
# run_taxi_epsilon_greedy(alpha, 0.6, epsilon, min_epsilon, decay_rate, episodes, render_mode)
# run_taxi_epsilon_greedy(alpha, 0.7, epsilon, min_epsilon, decay_rate, episodes, render_mode)
# run_taxi_epsilon_greedy(alpha, 0.8, epsilon, min_epsilon, decay_rate, episodes, render_mode)
# run_taxi_epsilon_greedy(alpha, 0.9, epsilon, min_epsilon, decay_rate, episodes, render_mode)
# run_taxi_epsilon_greedy(alpha, 1, epsilon, min_epsilon, decay_rate, episodes, render_mode)

#Result: Best option is gamma = 0.8 
gamma = 0.8

# Test different values of epsilon to explore what happens if we abandon the technique of epsilon-greedy approach

#if we choose any extreme value for epsilon, we will obtain a model that is self-evidently bad. But if we only explore and never exploit, what happens?

# Only explore, remove epsilon and choose random actions, but still learn from them
# run_taxi_always_explore(alpha, gamma, episodes, render_mode)
 
# On the other hand, if we want to exploit, we can choose to set decay = 1 and lower the value of epsilon:
# run_taxi_epsilon_greedy(alpha, gamma, 0.01, min_epsilon, 1, episodes, render_mode)

# With this baseline, we know that exploiting tends to be a better strategy
# We can test out some different decay rates with a preference for faster rates (smaller numbers)

# run_taxi_epsilon_greedy(alpha, gamma, epsilon, min_epsilon, .999, episodes, render_mode)
# run_taxi_epsilon_greedy(alpha, gamma, epsilon, min_epsilon, .995, episodes, render_mode)
# run_taxi_epsilon_greedy(alpha, gamma, epsilon, min_epsilon, .95, episodes, render_mode)
# run_taxi_epsilon_greedy(alpha, gamma, epsilon, min_epsilon, .9, episodes, render_mode)
# run_taxi_epsilon_greedy(alpha, gamma, epsilon, min_epsilon, .8, episodes, render_mode)
# run_taxi_epsilon_greedy(alpha, gamma, epsilon, min_epsilon, .7, episodes, render_mode)

# a rate of .95 seems to perform the best
decay_rate = .95

#next, we will test the minimum value of epsilon, so we will repeat the same process

# run_taxi_epsilon_greedy(alpha, gamma, epsilon, 0.01, decay_rate, episodes, render_mode)
# run_taxi_epsilon_greedy(alpha, gamma, epsilon, 0.02, decay_rate, episodes, render_mode)
# run_taxi_epsilon_greedy(alpha, gamma, epsilon, 0.05, decay_rate, episodes, render_mode)
# run_taxi_epsilon_greedy(alpha, gamma, epsilon, 0.1, decay_rate, episodes, render_mode)
# run_taxi_epsilon_greedy(alpha, gamma, epsilon, 0.15, decay_rate, episodes, render_mode)
# run_taxi_epsilon_greedy(alpha, gamma, epsilon, 0.2, decay_rate, episodes, render_mode)

min_epsilon = 0.01

#final model, with optimized params and the balance of exploitation and exploration:
run_taxi_epsilon_greedy(alpha, gamma, epsilon, min_epsilon, decay_rate, episodes, render_mode)
