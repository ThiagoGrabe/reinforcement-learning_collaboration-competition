### Solution Walkthrough


#### The environment

The goal of this project is to train two RL agents to play tennis. As in real tennis, the goal of each player is to keep the ball in play. And, when you have two equally matched opponents, you tend to see fairly long exchanges where the players hit the ball back and forth over the net.

![](./images/tennis.png)

TThe observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation. Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping.

The task is episodic, and in order to solve the environment, your agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). Specifically,

* After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores.
* This yields a single score for each episode.


#### Solution components

##### Algorithm

The algorithm chosen in this project is [Multi-Agent Deep Deterministic Policy Gradient MADDPG](https://arxiv.org/abs/1706.02275). MADDPG is based on DDPG agent. MADDPG, or Multi-agent DDPG, extends DDPG into a multi-agent policy gradient algorithm where decentralized agents learn a centralized critic based on the observations and actions of all agents.

##### Network

[_Actor_](model.py) 
* First fully connected layer with input size __24__ and output size __256__
* Second fully connected layer with input size __256__ and output size __128__
* Third fully connected layer with input size __128__ and output size __2__


[_Critic_](model.py)
* First fully connected layer with input size __24__ and output size __256__
* Second fully connected layer with input size (256 + 2) = __258__ and output size __128__
* Third fully connected layer with input size __128__ and output size __1__
* Batch Normalization layer between first and second layers



##### Experience Replay

The solution is using shared Replay Buffer to enable the agents to learn from each others experiences. Random sampling of past experience breaks correlation between sequential experiences and alos allows the agent to learn from the same experience multiple times.


##### Hyperparameters

```
_batch_size = 256       # minibatch size
_buffer_size = int(1e5) # replay buffer size
_gamma = 0.99           # discount factor
_lr_actor = 1e-4        # learning rate of the actor 
_lr_critic = 1e-4       # learning rate of the critic
_tau = 3e-1             # soft update interpolation
```


#### Training Results

##### Training results without noise decay

###### Agent training logs

```
...
Episode 700, Average Score: 0.08, Max: 0.60, Min: 0.00, Avg: 0.14, Time: 1.66
Episode 710, Average Score: 0.09, Max: 0.30, Min: 0.00, Avg: 0.07, Time: 1.02
Episode 720, Average Score: 0.09, Max: 0.60, Min: 0.00, Avg: 0.12, Time: 0.47
Episode 730, Average Score: 0.10, Max: 0.40, Min: 0.00, Avg: 0.15, Time: 5.10
Episode 740, Average Score: 0.10, Max: 0.30, Min: 0.00, Avg: 0.08, Time: 0.47
Episode 750, Average Score: 0.11, Max: 0.40, Min: 0.00, Avg: 0.16, Time: 3.63
Episode 760, Average Score: 0.12, Max: 0.50, Min: 0.00, Avg: 0.24, Time: 6.50
Episode 770, Average Score: 0.14, Max: 0.60, Min: 0.00, Avg: 0.29, Time: 3.61
Episode 780, Average Score: 0.19, Max: 2.60, Min: 0.00, Avg: 0.61, Time: 0.53
Episode 790, Average Score: 0.36, Max: 2.70, Min: 0.10, Avg: 1.79, Time: 32.40
Episode 800, Average Score: 0.41, Max: 2.70, Min: 0.00, Avg: 0.59, Time: 32.69


Environment solved in 804 episodes!	Moving Average Score: 0.514
```


![](./images/results.png)


 ### Exploration vs Exploitation
 
 As was mentioned earlier OU Noise function was introduced to enable environment exploration by the agent. There is a particular challenge related to this known as exploration vs. exploitation dilemma i.e. choosing which action to take while the agent is still learning the optimal policy. Should the agent choose an action based on the rewards observed so far. Alternatively should the agent explor other actions in hope and that will lead to pententially higher rewards later on? In this implementaion I investigated an impact of noise decay that reduces exploartion compoennt over time. I trained one model without noise decay and another one with ```_noise_decay = 0.999``` applied during the agent traing. 
 
The model without noise decay demostrated steady average score improment and was able to converge around 500 episode. I tried multiple traning rounds and there was relatively small variation with the environment solved within 500-650 episodes. Introducing noise decay changed the training pattern. It took the agent longer to get average score to improve but at some point the model converged very fast in just 20-30 episodes. Doing multiple trainng rounds showed singificant vartion in when the model started to converge (between 400-800 episodes).
 
Model with noise decay had higer testing scores but bigger variation. The model without decay had lower testing scores with smaller variatioin. My interpretaton of the results is that without noise decay the agent spends more time exploring the environment and this leads to a model trained over a broad action/state space that makes it work reasonably well in most scenarios. With noise decay the agent spends less time exploring and this results in a model biased towards a subset of action/state space. It performs really well in this subset and fares poorly in the scenarios the agent has not had opportunity to explore.

This enviroment seems to favor more aggressive exploration and applying a simplistic noise decay doesn't work particularly well. There is definitely a better way to find exploaration vs explotaion balance.

### Future work ideas

* Experiment different approches for finding better exploaration vs explotaion balance. E.g. apply noise later in the traing by setting episode threshold. QU Noise hyperameters coulld be fine tuned as well. E.g. reducing noise volitility may help the model to converge better. Another idea is to increse OU Noise output early in the training process to allow the agent to do more aggressiv initial exploration. The noise level could be reduced later in the training using noise decay.  
* Implement [Prioritized Experienece Replay](https://arxiv.org/abs/1511.05952). This can improve learning by increasing the probability that rare or important experiences can be sampled and explored.
* Address model stability to get more consistent results. One approach could be to implement [Gradient Clipping](https://machinelearningmastery.com/exploding-gradients-in-neural-networks/).
* Implement Play Soccer challenge.
