Notes are based on Sutton's textbook chapter 3, 4, 5, 6, 13, and codes from
https://github.com/dennybritz/reinforcement-learning

## Model-based
When we have an environment model (p(s',r| s,a) is given).

Q[s][a] = sum_{s', r} p(s',r| s,a)(r + gamma \* V[s'])

V[s] = sum_{a} pi[s][a] Q[s][a]

pi[s][a] is the prob of selecting action a in state s given by the policy

Policy evalution:
```
# Update V[s] by summing over V[s']
V = np.zeros(env.nS)
while True:
    delta = 0
    for s in range(env.nS):
        v = 0
        for a, action_prob in enumerate(policy[s]):
            for  prob, next_state, reward, done in env.P[s][a]: # p(s',r| s,a)
                v += action_prob * prob * (reward + discount_factor * V[next_state])
        delta = max(delta, np.abs(v - V[s]))
        V[s] = v
    if delta < theta:
        break
```

Policy improvement:
```
# For each state s, compute Qs by V. Update policy by selecting the highest Q.
def policy_improvement(env, discount_factor=1.0):

    def compute_Q(state, V):
        Q = np.zeros(env.nA)
        for a in range(env.nA):
            for prob, next_state, reward, done in env.P[state][a]:
                Q[a] += prob * (reward + discount_factor * V[next_state])
        return Q
    
    # Start with a random policy
    policy = np.ones([env.nS, env.nA]) / env.nA
    
    while True:
        policy_stable = True
        for s in range(env.nS):
            chosen_a = np.argmax(policy[s])
            
            action_values = compute_Q(s, V)
            best_a = np.argmax(action_values)
            
            if chosen_a != best_a:
                policy_stable = False
            policy[s] = np.eye(env.nA)[best_a]
        
        if policy_stable:
            return policy, V
```

Policy Iteration:

Repeat these steps:

	policy evaluation

	policy improvement

Generalized Policy Iteration:

Generally involve the interaction between policy evaluation and policy improvement.

Value Iteration:
First perform policy evalutation, update V[s] as max_a Q[s][a]. V converges to V\*.

V\*[s] = max_pi V_pi[s] for all s.

Q\*[s][a] = max_pi Q_pi[s][a] for all s and a.

then update policy by selecting max Q[s][a] (use V[s] to compute Q[s][a]).

Two tasks in RI:

Prediction, estimating value function.

Control, finding an optimal policy.

## Monte Carlo

No environment model, learn from experience.

The first-visit MC method for estimation:

Estimates v_{pi}(s) as the average of the returns following first visits to s.
```
def mc_prediction(policy, env, num_episodes, discount_factor=1.0):
    returns_sum = defaultdict(float)
    returns_count = defaultdict(float)
    
    V = defaultdict(float)
    
    for i_episode in range(1, num_episodes + 1):
        # An episode is an array of (state, action, reward) tuples
        episode = []
        state = env.reset()
        for t in range(100):
            action = policy(state)
            next_state, reward, done, _ = env.step(action)
            episode.append((state, action, reward))
            if done:
                break
            state = next_state

        # We convert each state to a tuple so that we can use it as a dict key
        states_in_episode = set([tuple(x[0]) for x in episode]) # All possible states as a set
        for state in states_in_episode:
        	# Only compute return G after the first visit to each state
            first_occurence_idx = next(i for i,x in enumerate(episode) if x[0] == state)
            # Sum up all rewards since the first occurance
            G = sum([x[2]*(discount_factor**i) for i,x in enumerate(episode[first_occurence_idx:])])
            returns_sum[state] += G
            returns_count[state] += 1.0 # Only takes return after the first visit
            V[state] = returns_sum[state] / returns_count[state]

    return V
```

On-policy first-visit MC control

Update Q values with MC, which gives policy.

On-policy: 

Use the same policy for exploration and learning. Apply epsilon greedy to ensure exploring all states.
```
def make_epsilon_greedy_policy(Q, epsilon, nA):
    def policy_fn(observation):
        A = np.ones(nA, dtype=float) * epsilon / nA
        best_action = np.argmax(Q[observation])
        A[best_action] += (1.0 - epsilon)
        return A
    return policy_fn

def mc_control_epsilon_greedy(env, num_episodes, discount_factor=1.0, epsilon=0.1): 
    returns_sum = defaultdict(float)
    returns_count = defaultdict(float)
    
    # A nested dictionary that maps state -> (action -> action-value).
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    
    # The policy we're following
    policy = make_epsilon_greedy_policy(Q, epsilon, env.action_space.n)
    
    for i_episode in range(1, num_episodes + 1):
        episode = []
        state = env.reset()
        for t in range(100):
            probs = policy(state)
            action = np.random.choice(np.arange(len(probs)), p=probs)
            next_state, reward, done, _ = env.step(action)
            episode.append((state, action, reward))
            if done:
                break
            state = next_state

        # Find all (state, action) pairs we've visited in this episode
        s_a_in_episode = set([(tuple(x[0]), x[1]) for x in episode])
        for state, action in s_a_in_episode:
            sa_pair = (state, action)
            first_occurence_idx = next(i for i,x in enumerate(episode) if x[0] == state and x[1] == action)
            G = sum([x[2]*(discount_factor**i) for i,x in enumerate(episode[first_occurence_idx:])])
            returns_sum[sa_pair] += G
            returns_count[sa_pair] += 1.0
            Q[state][action] = returns_sum[sa_pair] / returns_count[sa_pair]  
        # The policy is improved implicitly by changing the Q dictionary
    
    return Q, policy
```
Off-policy MC control and importance sampling

Off-policy: 

Use a separate behavior policy for exploratin. Learn a target policy as the optimal policy. Compared with on-pollicy, greater variance and slower converging. 

Importance sampling:

Use the estimation from the behavior policy to get the estimation of the target policy.
Importance-sampling ratio, the relative probability of the trajectory under the target and behavior policies, is

rho_{t: T - 1} = prod_{k = t}^{T - 1} (pi(A_k|S_k) / b(A_k|S_k))

pi is the target policy, and b is the behavior policy.

Use the returns from behavior policy to estimate value of target policy

V(s) = (sum_t rho_{t:T(t) - 1} G_t)/(sum_t rho_{t:T(t) - 1})

T(t) is the first time of termination following time t, and G_t denote the return after t up through T(t). Each t is a time step that state s is visited.

C_n is the cumulative sum of the weights given to the first n returns.

C_{n + 1} = C_n + W_{n + 1},

where W_n = rho_{t_i: T(t_i) - 1}

The update rule for V_n is

V_{n + 1} = V_n + (W_n / C_n)[G_n - V_n].

Here C starts from C_0 = 0, and V starts from a random V_1.

For off-policy MC control, choose a soft policy (positive prob for any state) as the behavior policy.
```
def create_random_policy(nA):
    A = np.ones(nA, dtype=float) / nA
    def policy_fn(observation):
        return A
    return policy_fn

def create_greedy_policy(Q):
    def policy_fn(state):
        A = np.zeros_like(Q[state], dtype=float)
        best_action = np.argmax(Q[state])
        A[best_action] = 1.0
        return A
    return policy_fn

def mc_control_importance_sampling(env, num_episodes, discount_factor=1.0):

    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    C = defaultdict(lambda: np.zeros(env.action_space.n))
    
    behavior_policy = create_random_policy(env.action_space.n)
    target_policy = create_greedy_policy(Q)
        
    for i_episode in range(1, num_episodes + 1):
        episode = []
        state = env.reset()
        for t in range(100):
            probs = behavior_policy(state)
            action = np.random.choice(np.arange(len(probs)), p=probs)
            next_state, reward, done, _ = env.step(action)
            episode.append((state, action, reward))
            if done:
                break
            state = next_state
        
        G = 0.0
        W = 1.0
        for t in range(len(episode))[::-1]:
            state, action, reward = episode[t]
            G = discount_factor * G + reward
            C[state][action] += W
            Q[state][action] += (W / C[state][action]) * (G - Q[state][action])
            if action !=  np.argmax(target_policy(state)):
                break # In this case, W will be 0. So break here.
            W = W * 1./behavior_policy(state)[action] # This step is by the definition of ratio.
        
    return Q, target_policy
```
Q, policy = mc_control_importance_sampling(env, num_episodes=500000)

## Temporal Difference Learning
TD method doesn't wait until the end of the episode to perform update like MC. It updates value after each time step.

V[s_t] += learning_rate \* (R_{t + 1} + gamma \* V[s_{t+1}] - V[s_t])

because 

v(s) = E[R_{t+1} + gamma \* v(S_{t+1}) | S_t = s],

here v(s) is the truth.

TD error is

R_{t + 1} + gamma \* V[s_{t+1}] - V[s_t]

TD methods update their estimates based in part on other estimates. They learn a guess from a guess—they bootstrap.

Sarsa: On-policy TD control

TD_target = R[t+1] + discount_factor \* Q[next_state][next_action]
```
def make_epsilon_greedy_policy(Q, epsilon, nA):
    def policy_fn(observation):
        A = np.ones(nA, dtype=float) * epsilon / nA
        best_action = np.argmax(Q[observation])
        A[best_action] += (1.0 - epsilon)
        return A
    return policy_fn

def sarsa(env, num_episodes, discount_factor=1.0, alpha=0.5, epsilon=0.1):
    Q = defaultdict(lambda: np.zeros(env.action_space.n))

    policy = make_epsilon_greedy_policy(Q, epsilon, env.action_space.n)
    
    for i_episode in range(num_episodes):
        state = env.reset()
        action_probs = policy(state)
        action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
        
        for t in itertools.count():
            next_state, reward, done, _ = env.step(action)
            
            next_action_probs = policy(next_state)
            next_action = np.random.choice(np.arange(len(next_action_probs)), p=next_action_probs)

            # TD Update.
            td_target = reward + discount_factor * Q[next_state][next_action]
            td_delta = td_target - Q[state][action]
            Q[state][action] += alpha * td_delta
    
            if done:
                break
                
            action = next_action
            state = next_state   
    return Q
```
Q-learning: Off-policy TD Control

TD_target = R[t+1] + discount_factor \* max(Q[next_state])

Q approximated q\*, which is independent of the policy followed.

```
def make_epsilon_greedy_policy(Q, epsilon, nA):
    def policy_fn(observation):
        A = np.ones(nA, dtype=float) * epsilon / nA
        best_action = np.argmax(Q[observation])
        A[best_action] += (1.0 - epsilon)
        return A
    return policy_fn

def q_learning(env, num_episodes, discount_factor=1.0, alpha=0.5, epsilon=0.1):
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    # This is the behavior policy
    policy = make_epsilon_greedy_policy(Q, epsilon, env.action_space.n)
    
    for i_episode in range(num_episodes):
        state = env.reset()
        
        for t in itertools.count():
            action_probs = policy(state)
            action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
            next_state, reward, done, _ = env.step(action)

            # TD Update
            best_next_action = np.argmax(Q[next_state])    
            td_target = reward + discount_factor * Q[next_state][best_next_action]
            td_delta = td_target - Q[state][action]
            Q[state][action] += alpha * td_delta
                
            if done:
                break
                
            state = next_state
    
    return Q
```

## Policy gradient
