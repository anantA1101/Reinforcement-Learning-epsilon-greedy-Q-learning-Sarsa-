import numpy as np
import matplotlib.pyplot as plt

#Board Parameters 
Board_width = 4
Board_length = 12

# initial Cur_Pos action pair values
Start = [3, 0]
Goal = [3, 11]

#Constant Parameters
Epsilon = 0.1
Learning_rate = 0.1
Discount_factor = 1

# Actions
Up = 0
Down = 1
Left = 2
Right = 3

Actions= [Up, Down, Left, Right]



def Get_Nxt_Pos_n_Reward(Cur_Pos, action):
    m,n = Cur_Pos

    if action == Up:
        Nxt_pos = [max(m - 1, 0), n]

    elif action == Left:
        Nxt_pos = [m, max(n - 1, 0)]

    elif action == Right:
        Nxt_pos = [m, min(n + 1, 11)]

    elif action == Down:
        Nxt_pos = [min(m + 1, 3), n]

    reward = -1

    if (action == 1 and m == 2 and 1 <= n <= 10) or (action == 3 and Cur_Pos == Start):
        
        reward = -100
        Nxt_pos = Start

    return Nxt_pos, reward

def Eps_Greedy_Action_Selection(Cur_Pos, Q_values):

    p  = np.random.random()

    if  p <= Epsilon:
        return np.random.choice(Actions)

    else:
        values = Q_values[Cur_Pos[0], Cur_Pos[1], :]
        list=[]
        for a, v in enumerate(values):
            if v == np.max(values):
                list.append(a)
        return np.random.choice(list)


def sarsa(Q_values, Lr = Learning_rate):

    Cur_Pos = Start
    action = Eps_Greedy_Action_Selection(Cur_Pos, Q_values)
    rewards = 0

    while Cur_Pos != Goal:

        Nxt_pos, reward = Get_Nxt_Pos_n_Reward(Cur_Pos, action)
        next_action = Eps_Greedy_Action_Selection(Nxt_pos, Q_values)
        rewards += reward
        Next_Q_Value = Q_values[Nxt_pos[0], Nxt_pos[1], next_action]

        Q_values[Cur_Pos[0], Cur_Pos[1], action] +=  Lr * (reward + Discount_factor * Next_Q_Value - Q_values[Cur_Pos[0], Cur_Pos[1], action])

        Cur_Pos = Nxt_pos
        action = next_action

    return rewards

def q_learning(Q_values, Lr=Learning_rate):
    Cur_Pos = Start
    rewards = 0

    while Cur_Pos != Goal:

        action = Eps_Greedy_Action_Selection(Cur_Pos, Q_values)
        Nxt_pos, reward = Get_Nxt_Pos_n_Reward(Cur_Pos, action)
        rewards += reward

        Q_values[Cur_Pos[0], Cur_Pos[1], action] += Lr * (reward + Discount_factor * np.max(Q_values[Nxt_pos[0], Nxt_pos[1], :]) - Q_values[Cur_Pos[0], Cur_Pos[1], action])
        Cur_Pos = Nxt_pos

    return rewards



def Rolling_avg_(a, n=10) :
    apend= []
    for i in range(9):
        apend.append(np.mean(a[:i+1]))

    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    b= ret[n - 1:]/n
    z= np.insert(b,0,apend)
    return z 


def Play():

    ep = 500
    runs = 10

    rewards_sarsa = np.zeros(ep)
    rewards_q_learning = np.zeros(ep)

    q_sarsa = np.zeros((Board_width, Board_length, 4))
    q_q_learning = np.zeros((Board_width, Board_length, 4))

    for r in range(runs):

        for i in range(0, ep):

            rewards_sarsa[i] += sarsa(q_sarsa)
            rewards_q_learning[i] += q_learning(q_q_learning)
            print("Episode: ", i, " Runs: ", r)

    rewards_sarsa /= runs
    rewards_q_learning /= runs

    Rewards_sarsa_Avg= Rolling_avg_(rewards_sarsa, n=10)
    Rewards_q_learning = Rolling_avg_(rewards_q_learning, n=10)

    plt.plot(Rewards_sarsa_Avg, label='Alg: Sarsa')
    plt.plot(Rewards_q_learning, label='Alg: Q Learning')
    plt.xlabel('Episode')
    plt.ylabel('Sum of rewards per Episode')
    plt.ylim([-100, 0])
    plt.legend()
    plt.show()

Play()
