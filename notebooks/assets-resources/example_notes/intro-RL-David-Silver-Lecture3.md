#raw 

Introduces [[Dynamic Programming]]

[[Dynamic Programming]] assumes full knowledge of the MDP.
It is used for planning in MDP.
(in RL you have two big problems: the [[Exploration & Exploitation in RL]] and the [[Prediction and Control in RL]])

### [[Policy Evaluation]]

### [[Policy Iteration]]

### Example: "Jack Rental's"
![[Pasted image 20210301150828.png | 400]]

![[Pasted image 20210301150925.png | 400]]

In this example, we see how we can use the value function which here is how much you get from having x amount of cars in the first location and x' amount on the other, and use it to calculate a better policy by acting greedily (improving on v):

![[Pasted image 20210301151110.png | 500]]

![[Pasted image 20210301153007.png]]

![[Pasted image 20210301153858.png]]
So the next policy $\pi'$ is chosen as the action that maximizes $q_{\pi}(s,a)$

If you improve on the first step and then follow the regular policy, it's better than always following the regular policy, because you are going to get at least as much reward as before.

And now we iterate:
![[Pasted image 20210301153825.png]]

Here we are unrolling the greedy for all steps and showing that taking this greedy policy is better for all steps, it will yield **at least** as much reward as the previous policy so it's better.

It is not guaranteed to keep getting better and find the optimal value!

If improvements stop, this can be described as saying:
![[Pasted image 20210301154241.png]]

Here we have the Bellman optimality equation satisfied:

![[Pasted image 20210301154326.png]]
This is the one step look ahead of the [[Bellman optimality equation]].
If this equation is satisfied then that means:
$$v_{\pi}(s) = v_*(s)  \text{ }\forall s \in S$$
He states in minute [54:20](https://www.youtube.com/watch?v=Nd1-UUMVfz4). that is it stops improving when we are following this greedy approach, that means that the Bellman optimality equation is satisfied and we found our optimal policy. It solves the MDP because it satisfied the Bellman optimality Equation.

Think of policy improvement as [[partial ordering]] of policies. 

So we don't want to evaluate policies when we are already in the optimal policy, however in the current version of the loop for policy evaluation, we have to. Can we truncate this process to reach the best policy faster and stop?

### Modified Policy Iteration

Basic idea: stop early.

One thing is to have a stopping condition:
![[Pasted image 20210301160111.png]]

Or simply stop after k iteration of iterative policy evaluation?

### [[Principle of Optimality]]
So the optimal policy corresponds to an optimal first action $A_*$ followed by an optimal policy for how to act from state $S'$ onwards...

### [[Value Iteration]]


Summary:
![[Pasted image 20210301164804.png]]
This final table is basically, given the problem of planning divided into its 2 subproblems: Prediction and Control, we have so far 1 approach to solve Prediction and 2 to solve Control, we use:
- Iterative policy evaluation with the bellman expectation equation to get the best possible prediction **(solves Prediction)**
- Policy Iteration with bellman expectation equation + greedy policy improvement **(solves Control)**
- Value Iteration with [[Bellman optimality equation]] **(solves Control)**

<span style="color: red"> Both prediction and control aim to find best policy...no? Evaluating the future and optimizing the future are interchangeable goals no?Should elucidate the difference in the lecture</span> 

### Assynchronous Dynamic Programming

Not using all states to perform the computations
![[Pasted image 20210301165757.png]]

#### Three ideas for asynchronous dynamic programming
##### In-place dynamic programming
![[Pasted image 20210301165954.png]]

Value function is updated immediately.

We plug in the latest information on the $v(s')$. 
How do you order states to compute things in the most efficient way?
##### Prioritized sweeping
Here you wanna come up with some measure of how important it is to update any state in your MDP, so you keep a priority queue that tells us which states are better than others.
![[Pasted image 20210301170221.png]]
We use the magnitude of the difference between the value of a state now compared to before as a guide to tell us which states to update
##### Real-time dynamic programming
Select the states that the agent actually visits. Update around these real samples that the agent actually visited.
Real experience as a guide 
![[Pasted image 20210301170727.png]]

### Full-Width Backups
Using full-width backups is expensive!
![[Pasted image 20210301170835.png]]
Sometimes you can't do one backup because there are too many states, so you sample.
![[Pasted image 20210301170956.png]]

![[Pasted image 20210301171015.png]]

### Gridworld Appendix
![[Pasted image 20210402170211.png]]

![[Pasted image 20210402170217.png]]

![[Pasted image 20210402170231.png]]

### Contraction Mapping

![[Pasted image 20210301171043.png]]

### Next

[[intro_RL_David_Silver_Lecture4]]

### References
- https://www.youtube.com/watch?v=Nd1-UUMVfz4
