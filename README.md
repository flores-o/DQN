Adapted from:
https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
https://github.com/jmichaux/dqn-pytorch

# Deep Q Learning

Q networks take as input some representation of the state of the environment
Q networks output one Q value per action

How does Deep Q learning work?
The core of the algorithm involves the computation of the temporal difference (TD) error for transition (Si, Ai, Ri, Si+1) sampled from taking actions in teh env:

  DELTAi = Yi - Q_curr(Si, Ai)
  
Where Yi = Ri + gamma * max(Q_prev(Si+1, a))

We minimize the loss:
L = 1/N * SUM i=1..n (DELTAi)


DQN ALGORITHM
[...]

# References:
https://web.stanford.edu/class/psych209/Readings/MnihEtAlHassibis15NatureControlDeepRL.pdf
