# Understanding TRPO: An Intuitive Guide

## The Core Problem: Learning to Walk Without Falling

Imagine you're teaching a robot to walk. It's currently stumbling around, but it's making *some* progress. Now you want to improve its walking policy.

With basic policy gradient methods (like REINFORCE), you compute a gradient that says "adjust the policy in this direction to get better." Then you take a step in that direction. Simple, right?

**Here's the problem:** How big should that step be?

- **Too small:** Learning takes forever. You'll need millions of iterations.
- **Too large:** You might completely destroy what the robot already learned. It was stumbling before, but now it just falls flat on its face every time.

This isn't just a theoretical concern—it happens constantly in practice. One bad update can send your policy into a terrible region of parameter space, and it may never recover. Your training curve looks great for 1000 episodes, then suddenly crashes to zero and stays there.

## The Intuition Behind Trust Regions

Here's the key insight that makes TRPO work:

**The gradient tells you which direction is uphill, but it only tells you the truth *locally*.**

Think of it like hiking in fog. You can feel which direction is uphill right where you're standing. But if you sprint 100 meters in that direction without checking, you might run off a cliff. The "uphill" information was only valid for the immediate area around you.

TRPO's solution: **Define a "trust region"—a small area around your current policy where you trust the gradient information.** Only update within that region.

But how do you define "small area" for a policy? Policies are probability distributions over actions, not points in physical space.

### Why Is This Worse in RL Than Supervised Learning?

You might wonder: doesn't supervised learning have the same problem? You're following gradients there too. Why don't we need trust regions for pretraining LLMs or training image classifiers?

The step size problem *does* exist in supervised learning—that's why we use learning rate schedulers, Adam, warmup, etc. But RL has it **much worse** for several reasons:

**1. Non-Stationary Data Distribution**

In supervised learning, your training data is fixed. The images don't change based on how well your classifier is doing.

In RL, **your policy determines what data you see**. If you update the policy, you visit different states, which changes your training distribution. A bad update doesn't just hurt this batch—it corrupts all future data collection.

**2. Feedback Loops / Distribution Shift**

This creates a vicious cycle in RL:
- Bad policy update → visit bad states → collect bad data → even worse policy update → ...

In supervised learning, even if you make a bad update, the next batch is still sampled from the same fixed distribution. You can recover. In RL, a bad update can send you into a region of state space you never escape from.

**3. No Ground Truth Labels**

In supervised learning, every sample has a correct answer. The gradient signal is (relatively) clean.

In RL, you're estimating value functions from noisy returns, computing advantages from imperfect baselines, and dealing with sparse or delayed rewards. The gradient estimates have much higher variance.

**4. Credit Assignment Over Time**

When classifying an image, the loss directly tells you how wrong you were about *this* image.

In RL, a reward at timestep 100 might be due to an action at timestep 3. The gradient has to propagate through this long chain, accumulating noise and uncertainty.

**5. On-Policy Data Becomes Stale**

In supervised learning, you can reuse data forever. Run 100 epochs over the same dataset.

In on-policy RL (like TRPO), once you update the policy, old data is from the "wrong" distribution. You have to throw it away and collect new data. This makes each update precious—you can't easily undo mistakes by training more on the same data.

**6. The Loss Landscape Is Nastier**

Supervised learning loss landscapes have been extensively studied and tamed:
- BatchNorm smooths things out
- Residual connections help gradients flow
- The cross-entropy loss is convex in the output logits

RL objectives can have sharp cliffs. A tiny parameter change might flip a critical action, completely changing the trajectory and resulting in a very different return.

**The Bottom Line**

Supervised learning: "If I take too big a step, I might need a few more epochs to recover."

RL: "If I take too big a step, I might never recover because I've destroyed my data collection process."

This is why RL needed special techniques like trust regions (TRPO), clipping (PPO), or very conservative updates (DQN's target networks). The problem isn't just optimization—it's that the optimization process itself affects the problem you're trying to solve.

## Measuring Policy Distance with KL Divergence

This is where KL divergence comes in. KL divergence measures how different two probability distributions are.

If your old policy says "go left with 80% probability, go right with 20%" and your new policy says "go left with 79% probability, go right with 21%", these policies are very similar—low KL divergence.

If the new policy says "go left with 20% probability, go right with 80%", that's a dramatic change—high KL divergence.

TRPO constrains the *average* KL divergence across all states to be less than some small threshold δ (delta). But what exactly *is* KL divergence?

---

## Understanding KL Divergence

KL divergence (Kullback-Leibler divergence) is fundamental to TRPO, so let's understand it properly.

### What KL Divergence Measures

KL divergence answers the question: **"If I think the world works according to distribution P, but it actually works according to distribution Q, how surprised will I be on average?"**

More precisely, it measures the expected extra "information" (in bits or nats) needed to encode samples from P using a code optimized for Q.

### The Formula

For discrete distributions P and Q over the same set of outcomes:

$$
D_{\text{KL}}(P \| Q) = \sum_x P(x) \log \frac{P(x)}{Q(x)}
$$

For continuous distributions:

$$
D_{\text{KL}}(P \| Q) = \int P(x) \log \frac{P(x)}{Q(x)} \, dx
$$

### Intuitive Breakdown

Let's unpack the formula $D_{\text{KL}}(P \| Q) = \sum_x P(x) \log \frac{P(x)}{Q(x)}$:

1. **$\log \frac{P(x)}{Q(x)}$**: For each outcome $x$, this measures how much more likely $x$ is under P compared to Q
   - If $P(x) > Q(x)$: positive (P assigns more probability than Q)
   - If $P(x) < Q(x)$: negative (P assigns less probability than Q)
   - If $P(x) = Q(x)$: zero (they agree)

2. **Weighted by $P(x)$**: We weight each term by $P(x)$ because we care about outcomes that *actually happen* under P. If P rarely produces outcome $x$, we don't care much if Q gets it wrong.

### Why Logarithm Instead of Just the Ratio?

You might wonder: why not just use $\sum_x P(x) \cdot \frac{P(x)}{Q(x)}$ without the log? There are several deep reasons:

**1. Information Theory Foundation**

The log comes from information theory. The "information content" or "surprise" of an event with probability $p$ is defined as $-\log(p)$:
- Rare events ($p$ small) → high surprise ($-\log(p)$ large)
- Common events ($p$ close to 1) → low surprise ($-\log(p)$ small)

When we see outcome $x$ that we expected with probability $Q(x)$ but actually had probability $P(x)$, our "extra surprise" is:

$$
-\log Q(x) - (-\log P(x)) = \log \frac{P(x)}{Q(x)}
$$

KL divergence is the *expected* extra surprise.

**2. Additivity for Independent Events**

Logs convert multiplication to addition. For independent events A and B:

$$
\log \frac{P(A,B)}{Q(A,B)} = \log \frac{P(A)P(B)}{Q(A)Q(B)} = \log \frac{P(A)}{Q(A)} + \log \frac{P(B)}{Q(B)}
$$

This means KL divergence of independent distributions adds up nicely:

$$
D_{\text{KL}}(P_{AB} \| Q_{AB}) = D_{\text{KL}}(P_A \| Q_A) + D_{\text{KL}}(P_B \| Q_B)
$$

Without the log, you'd get multiplication instead, which is much harder to work with.

**3. The Raw Ratio Has Bad Properties**

Consider $\sum_x P(x) \cdot \frac{P(x)}{Q(x)}$ (no log). This:
- Has no upper bound (can go to infinity even for "similar" distributions)
- Doesn't equal zero when $P = Q$ (it equals 1)
- Isn't connected to coding theory or information

**4. Connection to Cross-Entropy**

KL divergence decomposes beautifully:

$$
D_{\text{KL}}(P \| Q) = H(P, Q) - H(P)
$$

where $H(P) = -\sum_x P(x) \log P(x)$ is entropy and $H(P,Q) = -\sum_x P(x) \log Q(x)$ is cross-entropy.

This is why minimizing KL divergence is equivalent to minimizing cross-entropy (since $H(P)$ is constant w.r.t. Q)—the foundation of maximum likelihood estimation!

**5. Coding Theory Interpretation**

If you're designing a compression code, the optimal code length for outcome $x$ with probability $p$ is $-\log_2(p)$ bits. KL divergence measures how many extra bits you waste by using a code optimized for Q when the true distribution is P.

### A Concrete Example

Suppose we have two coins:
- **P (fair coin)**: Heads = 0.5, Tails = 0.5
- **Q (biased coin)**: Heads = 0.9, Tails = 0.1

How different is Q from P's perspective?

$$
D_{\text{KL}}(P \| Q) = 0.5 \cdot \log\frac{0.5}{0.9} + 0.5 \cdot \log\frac{0.5}{0.1}
$$

$$
= 0.5 \cdot \log(0.556) + 0.5 \cdot \log(5)
$$

$$
= 0.5 \cdot (-0.588) + 0.5 \cdot (1.609) \approx 0.51 \text{ nats}
$$

Now flip it—how different is P from Q's perspective?

$$
D_{\text{KL}}(Q \| P) = 0.9 \cdot \log\frac{0.9}{0.5} + 0.1 \cdot \log\frac{0.1}{0.5}
$$

$$
= 0.9 \cdot \log(1.8) + 0.1 \cdot \log(0.2)
$$

$$
= 0.9 \cdot (0.588) + 0.1 \cdot (-1.609) \approx 0.37 \text{ nats}
$$

Notice: **$D_{\text{KL}}(P \| Q) \neq D_{\text{KL}}(Q \| P)$**! KL divergence is asymmetric.

### Key Properties

1. **Non-negative**: $D_{\text{KL}}(P \| Q) \geq 0$ always
2. **Zero iff identical**: $D_{\text{KL}}(P \| Q) = 0$ if and only if $P = Q$
3. **Asymmetric**: $D_{\text{KL}}(P \| Q) \neq D_{\text{KL}}(Q \| P)$ in general
4. **Not a true distance**: Doesn't satisfy triangle inequality
5. **Infinite if Q has zeros where P doesn't**: If $P(x) > 0$ but $Q(x) = 0$, then $D_{\text{KL}}(P \| Q) = \infty$

### Why KL Divergence for TRPO?

TRPO uses $D_{\text{KL}}(\pi_{\text{old}} \| \pi_{\text{new}})$—measuring divergence from the **old** policy's perspective. Why this direction?

1. **We sample from $\pi_{\text{old}}$**: We collect data using the old policy, so we care about states/actions that the old policy actually visits
2. **Importance sampling validity**: The surrogate objective uses importance sampling with $\pi_{\text{old}}$ as the proposal distribution. KL divergence in this direction controls the variance of importance sampling estimates
3. **Prevents collapse**: If the new policy assigns zero probability to actions the old policy takes, $D_{\text{KL}}(\pi_{\text{old}} \| \pi_{\text{new}}) = \infty$, preventing such catastrophic changes

### KL Divergence for Common Distributions

**Categorical (discrete actions):**
$$
D_{\text{KL}}(P \| Q) = \sum_{i=1}^{k} p_i \log \frac{p_i}{q_i}
$$

**Gaussian (continuous actions):**
For $P = \mathcal{N}(\mu_1, \sigma_1^2)$ and $Q = \mathcal{N}(\mu_2, \sigma_2^2)$:

$$
D_{\text{KL}}(P \| Q) = \log\frac{\sigma_2}{\sigma_1} + \frac{\sigma_1^2 + (\mu_1 - \mu_2)^2}{2\sigma_2^2} - \frac{1}{2}
$$

This has a nice interpretation:
- First term: penalty for different spreads
- Second term: penalty for different means (scaled by Q's variance)

**Multivariate Gaussian:**
For $P = \mathcal{N}(\mu_1, \Sigma_1)$ and $Q = \mathcal{N}(\mu_2, \Sigma_2)$:

$$
D_{\text{KL}}(P \| Q) = \frac{1}{2}\left[ \log\frac{|\Sigma_2|}{|\Sigma_1|} - d + \text{tr}(\Sigma_2^{-1}\Sigma_1) + (\mu_2-\mu_1)^T\Sigma_2^{-1}(\mu_2-\mu_1) \right]
$$

where $d$ is the dimensionality.

---

**Back to TRPO:** By constraining $D_{\text{KL}}(\pi_{\text{old}} \| \pi_{\text{new}}) \leq \delta$, TRPO ensures:

1. The new policy behaves similarly to the old one in most situations
2. The surrogate objective (our local approximation) remains accurate
3. We're unlikely to catastrophically destroy performance

## The TRPO Recipe (Without the Math)

Here's what TRPO does at each iteration:

1. **Collect experience** using your current policy
2. **Estimate advantages** (which state-action pairs are better than average)
3. **Find the best policy update direction** that would maximize improvement
4. **Scale that direction** so you stay within the trust region (KL constraint)
5. **Double-check** with a line search: if the update violates the constraint or hurts performance, shrink it until it works

The magic is in steps 4 and 5. Instead of picking an arbitrary learning rate, TRPO *computes* the right step size based on the geometry of the policy space.

## Why Not Just Use a Small Learning Rate?

You might think: "Why not just use a tiny learning rate and avoid the complexity?"

The problem is that a fixed learning rate is **state-agnostic**. Consider:

- In some states, the policy is very confident (95% probability for one action). A small parameter change barely affects behavior.
- In other states, the policy is uncertain (50-50 between two actions). The same small parameter change could completely flip the decision.

A fixed learning rate treats both situations the same. TRPO's KL constraint naturally adapts—it measures actual behavior change, not just parameter change. This is sometimes called the "natural gradient" perspective.

## The Computational Challenge

There's a catch: enforcing the KL constraint requires knowing the curvature of the policy space, captured by something called the Fisher Information Matrix (FIM).

For a neural network with millions of parameters, this matrix would have *trillions* of entries. You can't even store it, let alone invert it.

TRPO's clever trick: use the **Conjugate Gradient algorithm**. CG can solve systems like $Fx = g$ without ever forming $F$ explicitly:

- $F$ = Fisher Information Matrix (the curvature we can't store)
- $g$ = policy gradient (what direction improves the objective)
- $x$ = the solution $F^{-1}g$ = the **natural gradient** = the update direction we want

Why do we want $F^{-1}g$ instead of just $g$? Because $F^{-1}g$ is the steepest ascent direction in *policy space* (measured by KL divergence), not parameter space. It automatically accounts for how sensitive the policy is to each parameter.

CG finds $x = F^{-1}g$ iteratively, only requiring matrix-vector products $Fv$, which can be computed efficiently with automatic differentiation.

---

## The Math Behind TRPO

Now that you have the intuition, let's formalize it. We'll build up the math step by step.

### Setup and Notation

- $\pi_\theta(a|s)$: Policy (probability of action $a$ in state $s$) parameterized by $\theta$
- $J(\theta)$: Expected total discounted reward under policy $\pi_\theta$
- $V^\pi(s)$: Value function—expected return starting from state $s$ under policy $\pi$
- $Q^\pi(s,a)$: Action-value function—expected return starting from state $s$, taking action $a$, then following $\pi$
- $A^\pi(s,a) = Q^\pi(s,a) - V^\pi(s)$: Advantage function—how much better is action $a$ than average?

### The Policy Gradient (Starting Point)

The standard policy gradient tells us how to improve $J(\theta)$:

$$
\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_t \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot A^{\pi_\theta}(s_t, a_t) \right]
$$

<details>
<summary><strong>Derivation of the Policy Gradient Formula (click to expand)</strong></summary>

#### Step 1: Define the Objective

We want to maximize the expected total reward:

$$
J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}[R(\tau)] = \mathbb{E}_{\tau \sim \pi_\theta}\left[\sum_{t=0}^{T} \gamma^t r_t\right]
$$

where $\tau = (s_0, a_0, r_0, s_1, a_1, r_1, \ldots)$ is a trajectory sampled by following policy $\pi_\theta$.

#### Step 2: Write Out the Expectation

The probability of a trajectory under policy $\pi_\theta$ is:

$$
P(\tau | \theta) = p(s_0) \prod_{t=0}^{T} \pi_\theta(a_t|s_t) \cdot p(s_{t+1}|s_t, a_t)
$$

So the objective is:

$$
J(\theta) = \int P(\tau|\theta) R(\tau) \, d\tau
$$

#### Step 3: Take the Gradient

$$
\nabla_\theta J(\theta) = \int \nabla_\theta P(\tau|\theta) \cdot R(\tau) \, d\tau
$$

#### Step 4: The Log-Derivative Trick

Here's the key trick. Notice that:

$$
\nabla_\theta P(\tau|\theta) = P(\tau|\theta) \cdot \nabla_\theta \log P(\tau|\theta)
$$

This is because $\nabla_\theta \log P = \frac{\nabla_\theta P}{P}$, so $\nabla_\theta P = P \cdot \nabla_\theta \log P$.

Substituting:

$$
\nabla_\theta J(\theta) = \int P(\tau|\theta) \cdot \nabla_\theta \log P(\tau|\theta) \cdot R(\tau) \, d\tau
$$

$$
= \mathbb{E}_{\tau \sim \pi_\theta}\left[\nabla_\theta \log P(\tau|\theta) \cdot R(\tau)\right]
$$

#### Step 5: Simplify $\nabla_\theta \log P(\tau|\theta)$

Taking the log of the trajectory probability:

$$
\log P(\tau|\theta) = \log p(s_0) + \sum_{t=0}^{T} \log \pi_\theta(a_t|s_t) + \sum_{t=0}^{T} \log p(s_{t+1}|s_t, a_t)
$$

Now take the gradient with respect to $\theta$:
- $\nabla_\theta \log p(s_0) = 0$ (initial state doesn't depend on $\theta$)
- $\nabla_\theta \log p(s_{t+1}|s_t, a_t) = 0$ (environment dynamics don't depend on $\theta$)

Only the policy terms survive:

$$
\nabla_\theta \log P(\tau|\theta) = \sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t|s_t)
$$

#### Step 6: The REINFORCE Formula

Substituting back:

$$
\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\left[\sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot R(\tau)\right]
$$

This is the basic REINFORCE gradient! But we can do better.

#### Step 7: Causality — Future Actions Don't Affect Past Rewards

The reward at time $t$ can only depend on actions at times $\leq t$. So we can replace $R(\tau)$ with the "reward-to-go" from time $t$:

$$
\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\left[\sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot \left(\sum_{t'=t}^{T} \gamma^{t'-t} r_{t'}\right)\right]
$$

The term $G_t = \sum_{t'=t}^{T} \gamma^{t'-t} r_{t'}$ is the return from time $t$.

#### Step 8: Baselines Don't Change the Expectation

For any function $b(s_t)$ that only depends on the state (not the action):

$$
\mathbb{E}_{a_t \sim \pi_\theta(\cdot|s_t)}\left[\nabla_\theta \log \pi_\theta(a_t|s_t) \cdot b(s_t)\right] = 0
$$

**Proof:** 
$$
\mathbb{E}_{a}\left[\nabla_\theta \log \pi_\theta(a|s) \cdot b(s)\right] = b(s) \cdot \mathbb{E}_{a}\left[\nabla_\theta \log \pi_\theta(a|s)\right]
$$

Expanding the expectation (by definition, $\mathbb{E}_{a \sim \pi}[f(a)] = \sum_a \pi(a) \cdot f(a)$):

$$
= b(s) \cdot \sum_a \pi_\theta(a|s) \cdot \nabla_\theta \log \pi_\theta(a|s)
$$

Using the chain rule: $\nabla_\theta \log \pi = \frac{\nabla_\theta \pi}{\pi}$ (derivative of log):

$$
= b(s) \cdot \sum_a \pi_\theta(a|s) \cdot \frac{\nabla_\theta \pi_\theta(a|s)}{\pi_\theta(a|s)} = b(s) \cdot \sum_a \nabla_\theta \pi_\theta(a|s) = b(s) \cdot \nabla_\theta \underbrace{\sum_a \pi_\theta(a|s)}_{=1} = b(s) \cdot \nabla_\theta 1 = 0
$$

So we can subtract any baseline $b(s_t)$ without changing the expected gradient.

#### Step 9: Use the Value Function as Baseline

From Step 7, we have:

$$
\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\left[\sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot G_t\right]
$$

Now here's the trick: we can write this as:

$$
\nabla_\theta J(\theta) = \mathbb{E}\left[\sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot G_t\right] - \mathbb{E}\left[\sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot V^{\pi_\theta}(s_t)\right] + \mathbb{E}\left[\sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot V^{\pi_\theta}(s_t)\right]
$$

We just added and subtracted the same thing (sneaky but legal!).

From Step 8, the last term equals zero (because $V^{\pi_\theta}(s_t)$ only depends on $s_t$, not $a_t$):

$$
\mathbb{E}\left[\nabla_\theta \log \pi_\theta(a_t|s_t) \cdot V^{\pi_\theta}(s_t)\right] = 0
$$

So we can just drop it:

$$
\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\left[\sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot \left(G_t - V^{\pi_\theta}(s_t)\right)\right]
$$

**Why bother?** The gradient is the same in expectation, but $G_t - V(s_t)$ has much lower variance than $G_t$ alone. Lower variance → faster, more stable learning.

#### Step 10: Recognize the Advantage Function

The quantity $G_t - V^{\pi_\theta}(s_t)$ is an estimate of the advantage:

$$
A^{\pi_\theta}(s_t, a_t) = Q^{\pi_\theta}(s_t, a_t) - V^{\pi_\theta}(s_t)
$$

(Since $\mathbb{E}[G_t | s_t, a_t] = Q^{\pi_\theta}(s_t, a_t)$)

**Final result:**

$$
\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\left[\sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot A^{\pi_\theta}(s_t, a_t)\right]
$$

</details>

**Intuition:** This formula says: increase the probability of actions that have positive advantage (better than average), decrease for negative advantage (worse than average).

**The problem:** We follow this gradient with some step size $\alpha$, giving us $\theta_{\text{new}} = \theta_{\text{old}} + \alpha \nabla_\theta J$. But what should $\alpha$ be?

### The Surrogate Objective

Instead of directly optimizing $J(\theta)$, TRPO optimizes a **surrogate objective** that we can evaluate using data from our current policy $\pi_{\theta_{\text{old}}}$.

The key insight: we can estimate the performance of a *new* policy $\pi_\theta$ using data collected from the *old* policy $\pi_{\theta_{\text{old}}}$ via importance sampling:

$$
L_{\theta_{\text{old}}}(\theta) = \mathbb{E}_{s \sim \rho_{\theta_{\text{old}}}, a \sim \pi_{\theta_{\text{old}}}} \left[ \frac{\pi_\theta(a|s)}{\pi_{\theta_{\text{old}}}(a|s)} A^{\pi_{\theta_{\text{old}}}}(s,a) \right]
$$

**Notation clarification:**

- $L_{\theta_{\text{old}}}(\theta)$ — "L" for "local approximation" or "lower bound". The subscript $\theta_{\text{old}}$ means "using data from the old policy." The argument $\theta$ is what we're optimizing. So $L_{\theta_{\text{old}}}(\theta)$ asks: "using samples collected with $\theta_{\text{old}}$, how good would $\theta$ be?"

- $\rho_{\theta_{\text{old}}}$ — the **state visitation distribution** under the old policy. When you roll out $\pi_{\theta_{\text{old}}}$ for many trajectories, $\rho_{\theta_{\text{old}}}(s)$ is the fraction of time you spend in state $s$. Formally: $\rho_\pi(s) = \sum_{t=0}^{\infty} \gamma^t P(s_t = s | \pi)$ (discounted state visitation).

- $\frac{\pi_\theta(a|s)}{\pi_{\theta_{\text{old}}}(a|s)}$ — the **importance sampling ratio**. It corrects for the fact that we're evaluating $\pi_\theta$ but using actions sampled from $\pi_{\theta_{\text{old}}}$.

**Important property:** At $\theta = \theta_{\text{old}}$, the ratio $\frac{\pi_\theta}{\pi_{\theta_{\text{old}}}} = 1$, and:

$$
\nabla_\theta L_{\theta_{\text{old}}}(\theta) \big|_{\theta = \theta_{\text{old}}} = \nabla_\theta J(\theta) \big|_{\theta = \theta_{\text{old}}}
$$

The surrogate gradient matches the true policy gradient at the current parameters! This means locally, optimizing $L$ is equivalent to optimizing $J$.

### The Trust Region Constraint

The surrogate $L$ is only a good approximation of $J$ when $\theta$ is close to $\theta_{\text{old}}$. We enforce this with a KL divergence constraint:

$$
\bar{D}_{\text{KL}}(\theta_{\text{old}} \| \theta) = \mathbb{E}_{s \sim \rho_{\theta_{\text{old}}}} \left[ D_{\text{KL}}\left( \pi_{\theta_{\text{old}}}(\cdot|s) \| \pi_\theta(\cdot|s) \right) \right] \leq \delta
$$

where $D_{\text{KL}}(P \| Q) = \sum_x P(x) \log \frac{P(x)}{Q(x)}$ measures how different distribution $Q$ is from $P$.

### The TRPO Optimization Problem

Putting it together, TRPO solves:

$$
\max_\theta \quad L_{\theta_{\text{old}}}(\theta) = \mathbb{E}\left[ \frac{\pi_\theta(a|s)}{\pi_{\theta_{\text{old}}}(a|s)} A^{\pi_{\theta_{\text{old}}}}(s,a) \right]
$$

$$
\text{subject to} \quad \bar{D}_{\text{KL}}(\theta_{\text{old}} \| \theta) \leq \delta
$$

This is a constrained optimization problem: maximize improvement while staying in the trust region.

### Making It Tractable: Taylor Approximations

Directly solving the above is hard. TRPO approximates both the objective and constraint using Taylor expansions around $\theta_{\text{old}}$:

**Objective (first-order approximation):**

$$
L_{\theta_{\text{old}}}(\theta) \approx L_{\theta_{\text{old}}}(\theta_{\text{old}}) + g^T(\theta - \theta_{\text{old}}) = g^T \Delta\theta
$$

where $g = \nabla_\theta L_{\theta_{\text{old}}}(\theta)|_{\theta_{\text{old}}}$ is the policy gradient and $\Delta\theta = \theta - \theta_{\text{old}}$.

(Note: $L_{\theta_{\text{old}}}(\theta_{\text{old}}) = 0$ because when $\theta = \theta_{\text{old}}$, the ratio is 1 and we're measuring advantage relative to itself.)

**Constraint (second-order approximation):**

$$
\bar{D}_{\text{KL}}(\theta_{\text{old}} \| \theta) \approx \frac{1}{2} \Delta\theta^T F \Delta\theta
$$

where $F$ is the **Fisher Information Matrix**:

$$
F = \mathbb{E}_{s, a \sim \pi_{\theta_{\text{old}}}} \left[ \nabla_\theta \log \pi_{\theta_{\text{old}}}(a|s) \cdot \nabla_\theta \log \pi_{\theta_{\text{old}}}(a|s)^T \right]
$$

The FIM captures the curvature of the KL divergence—it tells us how sensitive the policy distribution is to parameter changes.

### The Simplified Problem

Now we have:

$$
\max_{\Delta\theta} \quad g^T \Delta\theta \quad \text{subject to} \quad \frac{1}{2} \Delta\theta^T F \Delta\theta \leq \delta
$$

This is a quadratic constraint with a linear objective—it has a closed-form solution!

Using Lagrangian methods, the optimal update direction is:

$$
\Delta\theta^* = \frac{1}{\lambda} F^{-1} g
$$

where the Lagrange multiplier $\lambda$ is chosen to satisfy the constraint with equality. Working through the math:

$$
\Delta\theta^* = \sqrt{\frac{2\delta}{g^T F^{-1} g}} \cdot F^{-1} g
$$

This is the **natural gradient** $F^{-1}g$ scaled to exactly hit the trust region boundary.

### The Computational Problem: Fisher Matrix is Huge

For a neural network with $n$ parameters, $F$ is an $n \times n$ matrix. If $n = 10^6$ (common for modern networks), $F$ has $10^{12}$ entries—impossible to store or invert.

**Solution: Conjugate Gradient (CG)**

CG is an iterative algorithm that solves $Fx = g$ (i.e., finds $x = F^{-1}g$) without ever forming $F$. It only needs to compute **matrix-vector products** $Fv$ for various vectors $v$.

### Fisher-Vector Product (FVP)

The FVP $Fv$ can be computed efficiently using automatic differentiation:

1. Compute the KL divergence: $D = \bar{D}_{\text{KL}}(\theta_{\text{old}} \| \theta)$
2. Compute its gradient: $k = \nabla_\theta D \big|_{\theta_{\text{old}}}$
3. Compute the directional derivative: $k^T v$ (a scalar)
4. Compute the gradient of this scalar: $\nabla_\theta (k^T v) \big|_{\theta_{\text{old}}} = Fv$

This uses two backward passes but never forms the full matrix. For numerical stability, we add a damping term: $(F + \beta I)v$ where $\beta$ is small (e.g., 0.1).

### Backtracking Line Search

The approximations we made aren't perfect. The computed $\Delta\theta^*$ might:
- Violate the actual (non-approximated) KL constraint
- Actually decrease performance (the surrogate $L$ isn't the true objective $J$)

So we do a **line search**: try the full step, and if it fails our checks, shrink it:

```
for i in 0, 1, 2, ..., max_backtracks:
    θ_new = θ_old + (0.5)^i · Δθ*
    
    if KL(θ_old, θ_new) ≤ δ  AND  L(θ_new) ≥ 0:
        accept θ_new
        break
```

### Generalized Advantage Estimation (GAE)

TRPO needs advantage estimates $A(s,a)$. The simple approach is Monte Carlo: $A_t = G_t - V(s_t)$ where $G_t$ is the actual return. But this has high variance.

**GAE** provides a bias-variance tradeoff:

$$
A_t^{\text{GAE}} = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l}
$$

where $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$ is the TD error.

- $\lambda = 0$: Pure TD (low variance, high bias)
- $\lambda = 1$: Pure Monte Carlo (high variance, low bias)
- $\lambda \in (0,1)$: Interpolates between them (typically $\lambda = 0.95$)

### The Complete TRPO Algorithm

```
Initialize policy π_θ (actor) and value function V_φ (critic)

For each iteration:
    1. Collect trajectories using π_θ_old
    
    2. Compute advantages using GAE:
       δ_t = r_t + γV(s_{t+1}) - V(s_t)
       A_t = Σ (γλ)^l δ_{t+l}
    
    3. Compute policy gradient:
       g = (1/N) Σ ∇_θ log π_θ(a|s)|_{θ_old} · A
    
    4. Use Conjugate Gradient to compute:
       s ≈ F^{-1}g  (using FVP for matrix-vector products)
    
    5. Compute step size:
       α = sqrt(2δ / (s^T F s))
    
    6. Line search to find acceptable step:
       θ_new = θ_old + β·α·s  (where β found by backtracking)
    
    7. Update value function V_φ via regression on returns
```

### Why This Works: The Theory

The original TRPO paper proves that under certain conditions, this procedure guarantees **monotonic improvement**:

$$
J(\theta_{\text{new}}) \geq J(\theta_{\text{old}}) - C \cdot \bar{D}_{\text{KL}}(\theta_{\text{old}} \| \theta_{\text{new}})
$$

By keeping $\bar{D}_{\text{KL}} \leq \delta$ small enough, the penalty term $C \cdot \delta$ is smaller than the improvement, guaranteeing $J(\theta_{\text{new}}) \geq J(\theta_{\text{old}})$.

In practice, the approximations mean we don't get a hard guarantee, but the line search catches most failures.

---

## TRPO vs PPO: Why PPO Won

TRPO works well but is complex to implement correctly. The conjugate gradient, Fisher-vector products, and line search all require careful implementation.

PPO (Proximal Policy Optimization) achieves similar stability with a simpler idea: instead of a hard KL constraint, just clip the objective function to prevent large updates. It's not as theoretically principled, but it's much easier to implement and tune, which is why PPO became the standard.

## The One-Sentence Summary

**TRPO ensures stable policy learning by constraining each update to stay within a "trust region" where our local approximations are reliable, using KL divergence to measure policy similarity and conjugate gradient to efficiently compute the constrained update.**

## When to Use TRPO

- When you need rock-solid stability and are willing to pay for implementation complexity
- When you want to deeply understand the theory (TRPO's ideas underpin much of modern RL)
- When PPO isn't working and you suspect the clipping is too aggressive or too loose

In practice, most people use PPO. But understanding TRPO gives you a much deeper appreciation for *why* PPO works and when it might fail.
