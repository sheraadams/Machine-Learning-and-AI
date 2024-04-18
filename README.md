# AI and Machine Learning Algorithms

## Table of Contents:
<p align="center">
  <h2 style="font-family: 'Architects Daughter', 'Helvetica Neue', Helvetica, Arial, sans-serif;">
      <a href="https://github.com/sheraadams/AI-ML-Algorithms/#hand-written-digits-recognition-algorithm">Hand Written Digit Recognition</a><br>
      <a href="https://github.com/sheraadams/AI-ML-Algorithms/#image-recognition-algorithm">Image Recognition</a><br>
      <a href="https://github.com/sheraadams/AI-ML-Algorithms/#treasure-hunt-game">Treasure Hunt Game</a><br>
      <a href="https://github.com/sheraadams/AI-ML-Algorithms/#cartpole">Cartpole</a><br>
  </h2>
</p>

## Hand Written Digits Recognition Algorithm

This code is an image recognition algorithm from Deep Learning With Keras by Gulli and Pal. The algorithm is trained on the CIFAR-10 dataset that contains 60000 images that span 10 categorical classes (some examples of the categories include: airplane, automobile, bird, cat, deer, etc.) and it aims to train the machine to categorize novel images based on previous learning of attributes shared by other images that category (Gulli & Pal, 2017).

There are ethical and privacy considerations that should be taken into account when working with image recognition AI, especially if the technology is used for more significant uses including facial recognition, for example. Image recognition algorithms can result in privacy concerns when the subject has not explicitly given permission for their image to be used with the technology. If the algorithm was instead trained on faces, permission would be required and the context of the use of the images should be explained to the subject so that they can make an informed decision on whether they wish to participate to comply with GDPR transparency requirements. Specifically, The GDPR requires that software accessed by the public must clearly state the context and intended use of personal information in clear language (Recital 58 - the Principle of Transparency - General Data Protection Regulation (GDPR), 2019).  According to this law, without explicit permission from the subject, private and personally identifying data including photography may breach privacy or copyright laws.

Another ethical issue that arises is the potential for copyright infringement. Images such as photography featuring famous actors or celebrities are often copyrighted. Using copyrighted images requires additional permission and licensing for use by artificial intelligence for these purposes as well. To ensure compliance with copyright laws, open-source images or licensed images should be used in accordance with the license agreement. 

Finally, image recognition algorithms can also pose ethical concerns including discrimination if they are not properly implemented, trained, and maintained (Johnson, 2023). If data is not large and diverse, discriminatory judgments maybe made and inferred by artificial intelligence that categorizes by gender, race, or age, for example (SITNFlash, 2020). AI companies and institutions that use poorly trained algorithms that lead to discriminatory practices may find themselves liable for discriminatory actions that are made on these bases and efforts should always be made to provide holistic and well-rounded data to prevent these inequities in AI. 

**References**

Gulli, A., & Pal, S. (2017). *Deep Learning with Keras.* Packt Publishing Limited. ISBN: 978-1-78712-842-2.

Johnson, A. (2023, May 25). Racism and AI: Here’s how it’s been criticized for amplifying bias. *Forbes.* https://www.forbes.com/sites/ariannajohnson/2023/05/25/racism-and-ai-heres-how-its-been-criticized-for-amplifying-bias/?sh=4b623d1d269d

*Recital 58 - The Principle of Transparency - General Data Protection Regulation (GDPR)*. 
(2019, September 3). General Data Protection Regulation (GDPR). https://gdpr-
info.eu/recitals/no-58/

SITNFlash. (2020, October 26). *Racial discrimination in face recognition technology - Science in the news.* Science in the News. https://sitn.hms.harvard.edu/flash/2020/racial-discrimination-in-face-recognition-technology/#:~:text=Face%20recognition%20algorithms%20boast%20high,and%2018%2D30%20years%20old.

## Image Recognition Algorithm

The data for 20 epochs: 

- accuracy: 0.9461
- Test score: 0.19061198830604553
- Test accuracy: 0.9460999965667725
- 60000 train samples
- 10000 test samples

The data for for 60 epochs: 

- accuracy: 0.9674
- Test score: 0.10819873213768005
- Test accuracy: 0.9674000144004822
- 60000 train samples
- 10000 test samples

For 120 epochs: 

- accuracy: 0.9752
- Test score: 0.08114946633577347
- Test accuracy: 0.9751999974250793
- 60000 train samples
- 10000 test samples

This code was provided in the text, Deep Learning With Keras (Gulli & Pal, 2017). The code is an algorithm used to train the computer to recognize handwritten digits using supervised learning. According to IBM, supervised learning uses labeled datasets to train models to classify data or reach a certain conclusion (What Is Supervised Learning?  | IBM, n.d.). 

In this code, we can observe that with each iteration or epoch, the test accuracy increases as the machine becomes experienced. As expected, we see that as we increase the number of epochs, we observe that the accuracy increases. Though I only tested up to 120 epochs, Gulli and Pal explain that as we test up to a certain point, the accuracy increases, but after we test beyond around 100 epochs, increases in accuracy begin to slow. In the figure on page 28, we see that as we train the data past approximately 100 epochs, there is a point at which the accuracy benefits with additional training begin to diminish (Gulli & Pal, 2017). 

![1](https://github.com/sheraadams/AI-ML-Algorithms/assets/110789514/826dd858-b3ef-4c14-ab10-b562ec62327a)


Image credit: (Gulli & Pal, 2017)

In these cases, the model may be more likely to overfit the data, making generalizations that are not beneficial for predicting digits correctly. According to *Do Machine Learning Models Memorize or Generalize?* from the website Google PAIR, a “model overfits the training data when it performs well on the training data but poorly on the test data” (n.d.). Similarly, Investopedia defines overfitting as a “modeling error in statistics that occurs when a function is too closely aligned to a limited set of data points” (Twin, 2021) Overfitting may also involve making generalizations based on characteristics that are not appropriate, which is what we see in this model. 

In this model, accuracy rates increase as the training sample size and validation are kept constant. This happens as the model improves and "learns" patterns over many iterations. As we manipulate the number of epochs that the model is trained over, we increase the iterations that the model will execute the full dataset over and for each iteration, the model gains experience. As the data grows, the model becomes more and more experienced and efficient with its prediction capabilities. The benefits, however, begin to taper off and only modest improvements can be achieved after a certain number of epochs (around 100). After this point, the model begins overfitting and making mistakes and generalizations that prevent it from significantly improving in accuracy any further. 

The data set used to train image recognition can be found here: 

 https://www.cs.toronto.edu/~kriz/cifar.html
 
**References**

CIFAR-10 and CIFAR-100 datasets. (n.d.). https://www.cs.toronto.edu/~kriz/cifar.html

*Do machine learning models memorize or generalize?* (n.d.). https://pair.withgoogle.com/explorables/grokking/#:~:text=A%20model%20overfits%20the%20training,required%20to%20make%20more%20generalizations.

Gulli, A., & Pal, S. (2017). *Deep Learning with Keras. Packt Publishing Limited.* ISBN: 978-1-78712-842-2.

Twin, A. (2021, October 22). *Understanding overfitting and how to prevent it.* Investopedia. https://www.investopedia.com/terms/o/overfitting.asp

What is Supervised Learning?  | *IBM.* (n.d.). https://www.ibm.com/topics/supervised-learning

## Cartpole

**SUMMARY INFO AND VALUES USED**

Values for the original code
**Solved in 414 runs, 514 total runs.**

GAMMA = 0.95  
LEARNING_RATE = 0.001  
MEMORY_SIZE = 1000000
BATCH_SIZE = 20  
EXPLORATION_MAX = 1.0  
EXPLORATION_MIN = 0.01  
EXPLORATION_DECAY = 0.995  

Values for modifying exploration 
Exploration factor = comb EXPLORATION_MAX, EXPLORATION_MIN, and #EXPLORATION_DECAY
**Solved in 38 runs, 138 total runs.**

GAMMA = 0.95  
LEARNING_RATE = 0.001  
MEMORY_SIZE = 1000000  
BATCH_SIZE = 20  
**EXPLORATION_MAX = .5 # decreased from 0.01**
EXPLORATION_MIN = .01 
**EXPLORATION_DECAY = .999  # decreased from 0.995**

Values for modifying discount factor 
Discount factor GAMMA =.92 # decreased from .95 I tried .99 it was too high
**Solved in 105 runs, 205 total runs.**

**GAMMA =.92 # decreased from .95 I tried .99 it was too high**
LEARNING_RATE = 0.001  
MEMORY_SIZE = 1000000  
BATCH_SIZE = 20  
EXPLORATION_MAX = 1.0  
EXPLORATION_MIN = 0.01  
EXPLORATION_DECAY = 0.995  

Values for modifying learning rate
LEARNING_RATE = 0.004 # increased from .001
**Solved in 914 runs, 1014 total runs.**

GAMMA = 0.95  
**LEARNING_RATE = 0.004  # increased from .001**
MEMORY_SIZE = 1000000
BATCH_SIZE = 20  
EXPLORATION_MAX = 1.0  
EXPLORATION_MIN = 0.01  
EXPLORATION_DECAY = 0.995  

The cartpole problem, also known as the inverted pendulum problem uses the deep Q-learning algorithm to maintain balance. The Cartpole problem essentially is solved by positioning the pivot point under the center of mass (Surma, 2021). Using reinforcement learning, the application improves performance over time and stores information related to various states to determine the best course of action.

Explain how reinforcement learning concepts apply to the cartpole problem.
•	What is the goal of the agent in this case?
•	What are the various state values?
•	What are the possible actions that can be performed?
•	What reinforcement algorithm is used for this problem?

In this code, the goal of the agent is to balance a pole on top of a cart by moving right or left based on the current state. The states in this problem include the cart position, the cart velocity, the angle, and the pole angular velocity. Valid moves allowed in the code include moving right and left. The reinforcement algorithm involved in this code is the deep-Q network or DQN, which is a Q-learning algorithm.

Analyze how experience replay is applied to the cartpole problem.
•	How does experience replay work in this algorithm?
•	What is the effect of introducing a discount factor for calculating the future rewards?

The remember() method is responsible for adding experiences to update the network. Introducing a discount factor causes the machine to consider long-term rewards as well as short-term rewards. The discount factor can be used to balance the importance of short-term versus long-term rewards and can encourage strategizing. A high discount factor, for example, places more emphasis on long-term rewards versus short-term rewards. 

Analyze how neural networks are used in deep Q-learning.
•	Explain the neural network architecture that is used in the cartpole problem.
•	How does the neural network make the Q-learning algorithm more efficient?
•	What difference do you see in the algorithm performance when you increase or decrease the learning rate?

 The neural network architecture consists of an input layer, hidden layers that process, and an output layer. The neural network makes the Q-learning network our effective by learning from training over time and storing past experiences. When we modify the learning rate by increasing it, the machine makes larger adjustments toward the optimal position. While I expected that moderately increasing the learning rate from .001 to .004 would allow the algorithm to solve the problem quicker, I found that it drastically increased the time for the model to solve the problem. 

The cartpole problem can be solved using the Reinforce (also known as the Monte Carlo Policy Gradients) algorithm computing probability distribution of actions for each state such that where actions that have high reward for a given state are chosen (Yoon, 2021). As the model completes the problem, observed rewards and states are stored and their values contribute to probability distributions that inform future behavior where a given distribution refers to the likelihood of success. The pseudocode is provided from the University of Toronto lecture slides, and it is shown below:

![2](https://github.com/sheraadams/AI-ML-Algorithms/assets/110789514/524ac06d-93dc-49e9-94db-1f2ec3b97205)

Image source: (University of Toronto, n.d.)

The A2C can also be used to solve the cartpole problem using information from the “Critic” that estimates the Q-value (or the action value) and information from the “Actor” that estimates the V-value (the state value). We can see the pseudocode for the Q-Actor-critic method provided by Lillian Weng in her post, “Policy Gradient algorithms”.

![3](https://github.com/sheraadams/AI-ML-Algorithms/assets/110789514/76e43fe6-9a3c-4814-808f-2310a958bfa9)

Image Credit: (Weng, 2018).

Value based methods, policy-based methods, or Actor-Critic methods can all both be used to solve the cartpole problem, and each has advantages and disadvantages. There are differences between value-based methods and policy based methods like reinforce. Value based methods like Q-learning assess the value of each action for a given state and choose the highest estimated value for a given state. Policy gradient approaches determine the best action for a given state given the probability distribution of actions in each state. 
Actor critic approaches differ from value based and policy-based approaches. Value based Q-learning is based on action values alone. For each state, an action with the highest estimated value is chosen in this method.  Policy based approaches like Reinforce use statistical inference to assign actions given the current state. Reinforce methods select the favorable action based on their probably of success for a given state. Finally, Actor critic methods combine these two approaches where the actor is similar to the policy-based method, and the critic is similar to the value based method. 


**References**

Surma, G. (2021, October 13). Cartpole - Introduction to Reinforcement Learning (DQN - Deep Q-Learning). Medium. https://gsurma.medium.com/cartpole-introduction-to-reinforcement-learning-ed0eb5b58288

University of Toronto. (n.d.). Learning Reinforcement Learning by Learning REINFORCE [Slide show; Lecture Slide]. http://www.cs.toronto.edu/~tingwuwang/REINFORCE.pdf

Weng, L. (2018, April 8). Policy gradient algorithms. Lil’Log. https://lilianweng.github.io/posts/2018-04-08-policy-gradient/

Yoon, C. (2021a, December 7). Deriving policy gradients and implementing REINFORCE. Medium. https://medium.com/@thechrisyoon/deriving-policy-gradients-and-implementing-reinforce-f887949bd63

Yoon, C. (2021b, December 7). Understanding actor critic methods and A2C - towards data science. Medium. https://towardsdatascience.com/understanding-actor-critic-methods-931b97b6df3f

## Treasure Hunt Game

For this project, I was given starter code and a sample environment where the intelligent agent pirate is placed. I was tasked with developing a Q-learning algorithm that would enable the pirate to find the treasure. I was also given pseudocode to use as a reference to guide me in this development process. I developed code that led the pirate to the treasure, completing the mission successfully.

Computer scientists use reasoning and code to solve real-world problems. This matters in the everyday world in fields like AI, marketing, business, physics, and architecture among others. Computer scientists solve problems and provide efficient and practical solutions using technology saving businesses time and money while increasing precision and reliability. Technology can be used to automate processes, reduce workflows, and optimize performance. 

When I approach software development that involves problem solving, I first trying to understand the problem and the needs of the end user. This can be done through documentation and good communication. I also use research, testing, and iterative development to ensure that each of the user's requirements are addressed, objectively supporting each finding with evidence through testing. 

My ethical responsibilities in software development in general and in AI and Machine Learning include accuracy, confidentiality, transparency, and responsible use of data. The GDPR provides definitions of each of these terms and we can use these definitions to guide us whenever we work with client data to ensure that we meet and maintain high standards of responsibility at all times.

-	**Transparency:** The GDPR requires that websites clearly state the context and intended use of personal information in language that is clear and understood by the target audience (Recital 58 - the Principle of Transparency - General Data Protection Regulation (GDPR), 2019).
-	**Confidentiality:** the GDPR defines the “Right to Be Forgotten” as a rule protecting consumers from data sharing retroactively. According to the GDPR, consumers have the right to withdraw consent for their data to be used and shared publicly at any time with written notice.
-	**Accuracy:** EU data protection laws require that data is accurate and up-to-date and that this data should be “erased or rectified” in the case that the data changes over time (Law, 2022).  
-	**Responsible Data Use:** Data should be assigned reasonable expiration dates (for example, six months to one year) at which time, it should be scheduled for deletion or archival.

**References**

Law, G. D. (2022, February 15). What does “accuracy” mean under EU Data Protection law? Medium. https://medium.com/golden-data/what-does-accuracy-mean-under-eu-data-protection-law-dbb438fc8e95
Recital 58 - The Principle of Transparency - General Data Protection Regulation (GDPR). 

(2019, September 3). General Data Protection Regulation (GDPR). https://gdpr-
info.eu/recitals/no-58/

Right to be Forgotten - General Data Protection Regulation (GDPR). (2021, October 22). 
General Data Protection Regulation (GDPR). https://gdpr-info.eu/issues/right-to-be-
forgotten/

## Design Defense for the Treasure Hunt Game

This program is a game that uses artificial intelligence in which a pirate follows a maze, avoids obstacles, and searches for treasure. The pirate’s behavior is defined by the qtrain() method in which the pirate uses exploration and exploitation to train and learn the best way to complete the maze. The way in which artificial intelligence solves this problem is very different from how a human would solve this problem. While a human would solve this problem using their senses (like sight, touch, or sound) to make their next move, artificial intelligence, lacking these senses, solves this problem quite differently. The intelligent agent pirate in this code instead uses reinforcement learning and rewards to guide them throughout the maze. In this code, the pirate uses a combination of random actions and a greedy epsilon algorithm that maximizes reward to guide it through the maze. 

There are some similarities between how humans and artificial intelligence would solve this problem. The similarity is that both humans and artificial intelligence rely on trial and error, and both may occasionally use random or illogical actions to guide them from time to time. Also, both humans and artificial intelligence are more significantly motivated by reinforced actions. Reinforced actions in the case of a human are moves that avoid obstacles according to their memory and senses. Reinforced actions in the case of the pirate are those moves that avoid collisions and progress through the maze. 

The difference between the pirate and the agent approach in this solution may be that humans are less likely to rely on truly random actions in the same way that the pirate chooses random actions. Unlike artificial intelligence, humans may become distracted or make wrong moves that contradict prior learning. Also, unlike artificial intelligence, humans may forget obstacles that we have already encountered. If we instead think of the “random actions” of humans and artificial intelligence as categorically “illogical actions”, we can then see more similarities between human behavior and the artificial intelligence agent.

In this code, the pirate is guided by a combination of exploration and exploitation. According to Lindwurm, “the dilemma for a decision-making system that only has incomplete knowledge of the world is whether to repeat decisions that have worked well so far (exploit) or to make novel decisions, hoping to gain even greater rewards (explore)” (2021). Du and colleagues also summarize the machine learning exploration strategy as a combination of occasional random actions and optimizing rewards (Du et al., n.d.). In this AI system, exploitation refers to the tendency for the pirate to maximize reward while exploration refers to the pirate’s tendency to randomly explore regardless of reward or punishment. 

In reinforcement learning, the epsilon represents the tendency to explore. It also makes sense that without a sufficient epsilon value, the intelligent agent will not be motivated to move toward the treasure. We know that reinforcement learning depends on a combination of exploration and exploitation proportions to complete the task, with reinforcement from exploitation comprising the majority of the motivation (and a value equal to 1- epsilon). I found that the pirate solved the problem effectively (in 41 minutes) with an epsilon exploration value of .1, and I found that it was able to solve the problem much faster (22 minutes) with an epsilon value of .3. This demonstrates the importance of exploration in machine learning and the dilemma that arises when balancing the reward-seeking mechanism with exploration. 

This code demonstrates that the correct balance between exploration and exploitation is essential to the system’s ability to solve the problem. While a minimum amount of exploration is required to allow the pirate to move along the path even in the absence of rewards, the pirate is most influenced by Q-learning, or reinforcement and reward-seeking to “learn” to navigate. Reinforcement learning rewards the pirate for progressing through the maze toward the treasure while avoiding collision. The agent then “remembers”, and associates moves with reward or punishment Q-values and it progresses by maximizing rewards. In essence, the agent learns to avoid collisions that it has already encountered, progressing toward the treasure along the path associated with rewards.

This code uses a deep Q-learning algorithm that trains the neural network to achieve the desired outcome. The pirate progresses through the maze as the Q-learning algorithm updates the neural network with expected reward values, or Q-values associated with states and actions as the agent traverses the maze. As the training is executed, the network grows, and the agent “learns” to maximize rewards and minimize punishments following a greedy epsilon algorithm. 

 
**References**

Du, Y., Kosoy, E., Dayan, A., Rufova, M., Abbeel, P., & Gopnik, A. (n.d.). What can AI learn from human exploration? Intrinsically-Motivated Humans and Agents in Open-World Exploration. In Open Review. https://openreview.net/pdf?id=UKb6aHxs1f

Lindwurm, E. (2021, December 12). Intuition: Exploration vs Exploitation - Towards Data Science. Medium. https://towardsdatascience.com/intuition-exploration-vs-exploitation-c645a1d37c7a
