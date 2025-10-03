# Fairness-and-Explainability-CFA
This repository is the implementation of our proposed Comprehensive Fairness Algorithm (CFA) and our two explanation fairness evaluation metrics proposed in "Fairness and Explainability: Bridging the Gap Towards Fair Model Explanations".

## Motivation

<img width="550" alt="Screen Shot 2022-11-29 at 7 40 26 PM" src="https://user-images.githubusercontent.com/58016786/204686597-9d8b3dbd-b00d-47ce-8f6d-fa717819357c.png">

Although various fairness metrics have been proposed, almost all of them are directly computed based on the outputs. Accordingly, the quantified unfairness could only reflect the result-oriented bias while ignoring the potential bias caused by the decision-making procedure. The neglect of such procedure-oriented bias would restrict the debiasing methods to provide a comprehensive solution that is fair in both prediction and procedure. Therefore, we aim to bridge the gap between fairness and explaninability towards fair model explanations.

## Framework
<img width="550" alt="Screen Shot 2022-11-29 at 7 40 46 PM" src="https://user-images.githubusercontent.com/58016786/204686631-b9635ab1-05c4-4a29-9f89-8f89a10a35e7.png">


We propose a Comprehensive Fairness Algorithm (CFA) with multiple objectives, aiming to achieve good performance on utility, traditional and explanation fairness. The framework is built upon the traditional training with extra distance loss to minimize the hidden representation distances from two groups based on their original features and masked features. The former helps improve traditional fairness while two components together improve explanation fairness.


## Configuration
The default version of python we use is 3.8.8. The versions of pytorch and geometric are as follows:
```linux
- Pytorch 1.10.1 with Cuda 11.3
- Pytorch-geometric 2.0.3
```
For other packages, please import the environment from CFA_environment.yml with command:
```linux
conda env create -f CFA_environment.yml
```

## File structure
- For ease of understanding, the directory structure and their functions are as follows:
```linux
├── dataset
│   ├── bail (called Recidivism in the paper)
│   ├── german
│   ├── math
│   ├── por
├── mlp.py
├── parse.py
├── train.py (train CFA and record the validation records)
├── test.py (test CFA on the test dataset with the obtained best hyperparameters)
├── utils.py
├── explanation_metrics.py (explanation fairness evaluation)
├── run_german.sh (command to train CFA on german dataset, similar for other datasets)
├── run_best.sh (command to test CFA)
├── scripts
│   ├── Best Validation.ipynb (obtain the best hyperparameter)
│   ├── Test Result.ipynb (obtain the test result)
```
## If you are interested in the explanation fairness evaluation
Two proposed explanation fairness metrics are in _explanation_metrics.py_ where we include the explainer (graphlime) and the interpreter for evaluation. Since our framework is model-agonostic, here the main input to the interpreter is the inputs and outputs from the model and also ground truth label for calculating utility performance and sensitive label for calculating fairness. Feel free to play around with other models once they have the same input and output formats.

## If you are interested in the CFA algorithm
1. Use _train.py_ to obtain validation logs (refer to _run_german.sh_ of how to run train.py)
2. Use _Best Valiation.ipynb_ to obtain the best validation hyperparameters
3. Use the obtained hyperparameters and _test.py_ to obtain results in test dataset (refer to _run_best.sh_ of how to run _test.py_)
4. Use _Test Result.ipynb_ to print the test results


## Acknowledgement: The code is developed based on part of the code in the following papers:
```linux
[1] Huang, Qiang, Makoto Yamada, Yuan Tian, Dinesh Singh, and Yi Chang. "Graphlime: Local interpretable model explanations for graph neural networks." IEEE Transactions on Knowledge and Data Engineering (2022).
[2] Agarwal, Chirag, Himabindu Lakkaraju, and Marinka Zitnik. "Towards a unified framework for fair and stable graph representation learning." In Uncertainty in Artificial Intelligence, pp. 2114-2124. PMLR, 2021.
```

Define a fairness score s = (DP+EO)/2 ,the average of DP and EO score
We will measure the fairness score of training set/testing set/ and model predict result

Measurement matrix
Define a fair score: s = (dp + eo)/2 (lower the better)
Define a fairness improvement of a ml model:
s1(training) = fair score of training set
s2(testing) = fair score of testing set
s3(predict) = fair score of predicting model
fairImprovement(I score) = (s1-s3)/(s1-s2) . We use methods to ensure s1>=s2, which means the training set is more baised than the testing set. 
Ratio discussion: the I score is range from -\inf to \inf. When we cap the ratio to (0,\inf), because a negative score indicates the model did worse on fairness.A score ranges from 0-1 means the model improve some fairness, but didn’t reach the test set standard. A score equal 1 means the model can improve fair reaching the same fair level of test set. A score larger than 1 means the model improve fair level much more than the test set. 
The fairImprovement score: the larger the better.


Define a new method to split the train/test data.
assume we define t1: training, t2: validiation, t3: testing. t1+t2+t3 = 1
given a fair noise range (0-1)
given a seed value
Each dataset has a senstive label. 
Our approch: first split all data into two groups based on senstive label (group s1 and s2).
based on the fair noise, like 10%, then there is a 10% chance we random select two data into test set (these two data may from same group or different group)
For each data point in s1, find the most similar point in s2 (most similar means besides from the predict label, all other labels are similar)
And if this data point in s1 is similar with data point in s2, and their predict label are the same, then sort them into testing dataset
Then we keep looping this process, util the testing dataset capacity has reach the threshold (>=t2+t3)
Then randomly split the testing dataset into test and validation based on t2 and t3. 
And then, we have successfully split all the dataset into train/validation/test
After split the dataset, we need to print out the fairness score of each part
we ensure the training set fair score is larger than test and validation, otherwise retry with a different seed value.
fair score lower is better


We define a final score
final score = (auc+f1+acc)/3-(dp+eo)/2-(vef+ref)/2+log(I)
the first part measure utility, second part measure result fairness, third part mesure procedure fairness, the lass part measure debias ability of the model.
let's implment this score system during the training process, print out the final score along the training process


python train.py --epoch 450 --use_similarity_split
python train.py --epoch 500 --dataset german


We will import a new matrix called GAD

first during the training process, we will measure the attention vector
for each data, we will collect the attention vector as (a1,a2,a3...) the sume equal 1.

then we group them by the senstive label (female/male) (white/nonwhite)
Then compute the average attention per group
we will get something like this male group = [a1,a2,a3..]
femail group [b1,b2,b3...]
Then next step we will compare the group distrubution
compare the two average attention distrubutions with jensen-shannon divergence also called JSD

since the current training process doesn't contain a attention mechism, we need to use a SHAP value to measure it

python -m moe_expert.run_moe


We treat the gate as a policy network that selects or weights experts. Reward is your proposed relative score:

𝑅
=
(
𝑢
2
−
𝑢
1
)
−
(
𝑓
2
−
𝑓
1
)
R=(u
2
	​

−u
1
	​

)−(f
2
	​

−f
1
	​

)
🔎 RL Setup
State

Input features 
𝑥
x (same as experts).

Optionally: disagreement signals between experts (e.g., variance of predictions).

Action

Gate outputs a distribution over experts (via softmax).

Sample an expert (or weighted mixture with Gumbel-softmax).

Reward

Compute:

Baseline scores from Expert1 → 
𝑢
1
,
𝑓
1
u
1
	​

,f
1
	​

.

Mixture scores from gate output → 
𝑢
2
,
𝑓
2
u
2
	​

,f
2
	​

.

Reward:

𝑅
=
(
𝑢
2
−
𝑢
1
)
−
(
𝑓
2
−
𝑓
1
)
R=(u
2
	​

−u
1
	​

)−(f
2
	​

−f
1
	​

)
Policy Gradient

Compute log-prob of chosen expert:

m = torch.distributions.Categorical(probs_gate)
action = m.sample()
log_prob = m.log_prob(action)


Loss (REINFORCE):

advantage = R - baseline
loss = -advantage * log_prob


Update baseline with EMA of rewards to reduce variance.

Add entropy bonus to encourage exploration.

🧩 Pseudocode (training loop)
for epoch in range(num_epochs):
    Xb, yb, sb = get_batch()

    # ---- Expert 1 baseline ----
    _, p1 = expert1(Xb)
    u1, f1 = compute_scores(p1, yb, sb)  # utility ↑, fairness ↓

    # ---- Gate decision ----
    probs_gate = gate(Xb)   # (batch, 3)
    dist = torch.distributions.Categorical(probs_gate)
    action = dist.sample()  # pick expert index per sample

    # ---- Mixture result ----
    outputs = []
    _, p2 = expert2(Xb); outputs.append(p2)
    _, p3 = expert3(Xb); outputs.append(p3)
    experts = [p1, p2, p3]
    chosen_outputs = torch.stack([experts[a][i] for i,a in enumerate(action)])

    u2, f2 = compute_scores(chosen_outputs, yb, sb)

    # ---- Reward ----
    R = (u2 - u1) - (f2 - f1)
    reward = torch.tensor(R, device=device)

    # ---- Baseline and advantage ----
    baseline = momentum * baseline + (1 - momentum) * reward.item()
    advantage = reward - baseline

    # ---- REINFORCE loss ----
    log_prob = dist.log_prob(action)
    entropy = dist.entropy().mean()
    loss = -advantage * log_prob - beta * entropy

    opt_g.zero_grad()
    loss.backward()
    opt_g.step()

🛠️ Notes

Per-batch reward: compute 
𝑢
,
𝑓
u,f averaged over batch (reduces noise).

Baseline: EMA over last rewards.

Entropy bonus: keeps gate from collapsing to uniform / single expert too early.

Variance reduction: normalize rewards in batch (z-score).

⚡ This way, the gate is pure RL: it learns a policy that chooses experts only when doing better than Expert1 baseline.