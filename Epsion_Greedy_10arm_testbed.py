import numpy as np 
import matplotlib.pyplot as plt


class epsilon_greedy():

    def __init__(self,k, epsilon, iter,plays):
        
        self.epsilon= epsilon
        self.iter= iter
        self.k = k
        self.plays= plays

        self.avg_reward_axis=np.zeros((self.plays,len(self.epsilon)))
        self.Optimal_actions_axis =np.zeros((self.plays,len(self.epsilon)))
        
        for j in range(self.iter):
            
            self.action_values= np.random.normal(0,1,self.k)
            self.Q_A= np.zeros((len(self.epsilon) , self.k))

            self.optimal_action= np.argmax(self.action_values)

            self.Times_Action_Chosen= np.zeros((len(self.epsilon),self.k))
            self.reward_sum= np.zeros((len(self.epsilon),self.k))
            self.n= 0
            
            for t in range(self.plays):

                for i in range(0,len(self.epsilon)):

                    prob= np.random.random()

                    if prob < self.epsilon[i]:

                        action, rr =self.explore(i)

                        if action == self.optimal_action:
                            self.Optimal_actions_axis[t,i] += 1
                        
                    else:

                        action, rr = self.exploit(i)

                        if action == self.optimal_action:
                            self.Optimal_actions_axis[t,i] += 1
                    

                    self.avg_reward_axis[t,i] += rr

                    print("FOR Epsilon: ", self.epsilon[i] ," For Iter : ", j , " play: ", t )

        self.avg_reward_axis = self.avg_reward_axis/self.iter
        self.Optimal_actions_axis= self.Optimal_actions_axis/self.iter


        plt.title("10 Arm Bandit (Rewards)")
        plt.plot(self.avg_reward_axis)
        plt.xlabel(' Pulls ')
        plt.ylabel('Average Reward')
        plt.legend(self.epsilon)
        plt.show()
        
        plt.title("10 Arm Bandit (Optimal Actions %)")
        plt.plot(self.Optimal_actions_axis*100)
        plt.ylim(0, 100)
        plt.xlabel('Pulls')
        plt.ylabel("Times Optimal Action Chosen (in %)")
        plt.legend(self.epsilon)
        plt.show()
    
     

    def explore(self,eps):
            
            action = np.random.choice(range(len(self.Q_A[eps])))
 
                        
            self.Times_Action_Chosen[eps,action]+= 1
            self.n += 1

            r= np.random.normal(self.action_values[action],1)

            self.reward_sum[eps,action] += r
            self.Q_A[eps,action]= self.reward_sum[eps,action]/self.Times_Action_Chosen[eps,action]
            
            return action , r 
           
    def exploit(self,eps):

            maxaction = np.argmax(self.Q_A[eps])
            action_array= np.where(self.Q_A[eps] == np.argmax(self.Q_A[eps]))[0]

            if len(action_array) == 0:
                action = maxaction
            else:
                action = np.random.choice(action_array)
        

            self.Times_Action_Chosen[eps, action]+= 1
            self.n += 1

            r= np.random.normal(self.action_values[action],1)


            self.reward_sum[eps, action] += r
            self.Q_A[eps,action]= self.reward_sum[eps, action]/self.Times_Action_Chosen[eps, action]

            return action , r 


e= epsilon_greedy(10, [0.01,0.1,0] ,2000,1000)



