# Competition_3v3snakes

### Environment

<!-- ![image](https://github.com/jidiai/Competition_3v3snakes/blob/master/assets/snakesdemo.gif) -->

<img src="https://github.com/jidiai/Competition_3v3snakes/blob/master/assets/snakesdemo.gif" alt="Competition_3v3snakes" width="500" height="250" align="middle" />

Check details in Jidi Competition [RLChina2021智能体竞赛](http://www.jidiai.cn/compete_detail?compete=6)

---

### Dependency

You need to create competition environment.

> conda create -n snake3v3 python=3.7.5

> conda activate snake3v3

> pip install -r requirements.txt

---

### How to train rl-agent

> python rl_trainer/main.py

By default-parameters, the total reward of training is shown below.

![image](https://github.com/jidiai/Competition_3v3snakes/blob/master/assets/training.png)

You can edit different parameters, for example

> python rl_trainer/main.py --algo "qmix" --exp_name dev --epsilon 0.5

- important arguments
  - --algo: ddpg/bicnet/sac/qmix
  - --exp_name: path to save models and logs
  - --epsilon: for epsilon-greedy algorithms

You can locally evaluation your model.

> python evaluation_all.py --my_ai ddpg/sac/qmix/random--opponent ddpg/sac/qmix/random

![image](https://github.com/jidiai/Competition_3v3snakes/blob/master/assets/baseline.png)

---

### How to test submission

You can locally test your submission. At Jidi platform, we evaluate your submission as same as **run_log.py**

Once you run this file, you can locally check battle logs in the folder named "logs".

For example,

> python run_log.py --my_ai "random" --opponent "rl"

---

### Ready to submit

1. Random policy --> **agent/random/submission.py**
2. RL policy --> **all files in agent/rl/**

---

### Watch reply locally

1. Open reply/reply.html in any browser.
2. Load a log.
3. Reply and watch ^0^.
