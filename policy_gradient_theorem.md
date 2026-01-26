# Policy Gradient Theorem Explained

The **Policy Gradient Theorem** is the mathematical foundation that makes policy-based RL possible. It answers: *how do we compute the gradient of expected reward with respect to policy parameters?*

## The Problem

We want to maximize the expected total reward:

$$J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}[R(\tau)]$$

where $\tau$ is a trajectory (sequence of states, actions, rewards) and $\theta$ are our policy network's parameters.

**The challenge:** The expectation is over trajectories sampled from the policy. When we change $\theta$, we change *which trajectories get sampled*. How do we differentiate through that?

---

## The Theorem

The Policy Gradient Theorem gives us a surprisingly elegant formula:

$$\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t | s_t) \cdot G_t \right]$$

Where:
- $\nabla_\theta \log \pi_\theta(a_t | s_t)$ — direction to increase probability of action $a_t$
- $G_t$ — return (total reward) from time $t$ onward

---

## Intuition: The Log-Derivative Trick

### What Is It?

It's just the **chain rule** applied to $\log f(x)$:

$$\frac{d}{dx} \log f(x) = \frac{1}{f(x)} \cdot \frac{df}{dx}$$

Rearranging:

$$\frac{df}{dx} = f(x) \cdot \frac{d}{dx} \log f(x)$$

Or in gradient notation:

$$\nabla f = f \cdot \nabla \log f$$

That's it — nothing fancy.

### Why Use It?

The reason is purely practical: **it turns something we can't estimate into something we can**.

**Before the log trick:**

$$\nabla_\theta J(\theta) = \sum_{\tau} \nabla_\theta P(\tau; \theta) \cdot R(\tau)$$

This is **not** an expectation. To compute this, you'd need to know $\nabla_\theta P(\tau; \theta)$ for every possible trajectory — impossible in practice.

**After the log trick:**

$$\nabla_\theta J(\theta) = \sum_{\tau} P(\tau; \theta) \cdot \nabla_\theta \log P(\tau; \theta) \cdot R(\tau)$$

$$= \mathbb{E}_{\tau \sim \pi_\theta} \left[ \nabla_\theta \log P(\tau; \theta) \cdot R(\tau) \right]$$

Now it's an expectation over trajectories sampled from your policy. You **can** estimate this:

1. Run your policy, collect trajectories
2. For each trajectory, compute $\nabla_\theta \log \pi_\theta(a_t|s_t)$ (easy — just backprop through your network)
3. Multiply by $R(\tau)$
4. Average

**Summary:** The log trick converts "sum weighted by $\nabla P$" (can't sample) into "expectation under $P$" (can sample). That's the entire reason — it makes the gradient computable from experience.

---

## What It Means in Plain English

The gradient formula says:

> **"Increase the probability of actions that led to high returns. Decrease the probability of actions that led to low returns."**

- $\nabla_\theta \log \pi_\theta(a_t|s_t)$ points in the direction that makes action $a_t$ more likely
- $G_t$ scales this: positive return = push toward that action, negative = push away

---

## A Simple Example

Suppose in state $s$, your policy outputs:
- $\pi(\text{left}|s) = 0.3$
- $\pi(\text{right}|s) = 0.7$

You sample `right`, and get return $G = +10$.

The gradient update will:
1. Compute $\nabla_\theta \log \pi(\text{right}|s)$ — direction to increase `right`'s probability
2. Multiply by $+10$ — strong push in that direction
3. Update: now `right` becomes even more likely (e.g., 0.8)

If instead $G = -5$, the update would make `right` *less* likely.

---

## Why This Matters

Without this theorem, we'd be stuck. We can't backprop through "sampling an action" directly. The log-derivative trick converts:

- ❌ Gradient of (sample from distribution) — **not differentiable**
- ✅ Gradient of (log probability) × reward — **differentiable!**

This is why REINFORCE works, and it's the foundation for PPO, TRPO, A2C, and essentially all modern policy gradient methods.

---

## Full Derivation (Step by Step)

### Step 1: Define Our Objective

We want to maximize expected return. A trajectory $\tau$ is a sequence:

$$\tau = (s_0, a_0, r_1, s_1, a_1, r_2, \ldots, s_{T-1}, a_{T-1}, r_T, s_T)$$

The total return of a trajectory is the sum of all rewards:

$$R(\tau) = \sum_{t=1}^{T} r_t = r_1 + r_2 + \cdots + r_T$$

(Or with discounting: $R(\tau) = \sum_{t=1}^{T} \gamma^{t-1} r_t$)

Our objective is the expected return over all possible trajectories:

$$J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}[R(\tau)]$$

### Step 2: Expand the Expectation

The expectation over trajectories can be written as a sum (or integral) weighted by trajectory probabilities:

$$J(\theta) = \sum_{\tau} P(\tau; \theta) \cdot R(\tau)$$

where $P(\tau; \theta)$ is the probability of trajectory $\tau$ occurring under policy $\pi_\theta$.

Substituting our definition of $R(\tau)$:

$$J(\theta) = \sum_{\tau} P(\tau; \theta) \cdot \left( \sum_{t=1}^{T} r_t \right)$$

### Step 3: Expand the Trajectory Probability

What is $P(\tau; \theta)$? It's the probability of seeing that exact sequence of states and actions:

$$P(\tau; \theta) = p(s_0) \cdot \pi_\theta(a_0|s_0) \cdot p(s_1|s_0,a_0) \cdot \pi_\theta(a_1|s_1) \cdot p(s_2|s_1,a_1) \cdots$$

More compactly:

$$P(\tau; \theta) = p(s_0) \prod_{t=0}^{T-1} \pi_\theta(a_t|s_t) \cdot p(s_{t+1}|s_t, a_t)$$

Where:
- $p(s_0)$ — initial state distribution (from environment, independent of $\theta$)
- $\pi_\theta(a_t|s_t)$ — policy probability of action $a_t$ in state $s_t$ (**depends on $\theta$**)
- $p(s_{t+1}|s_t, a_t)$ — environment dynamics (independent of $\theta$)

### Step 4: Take the Gradient

Now we differentiate $J(\theta)$ with respect to $\theta$:

$$\nabla_\theta J(\theta) = \nabla_\theta \sum_{\tau} P(\tau; \theta) \cdot R(\tau)$$

Move the gradient inside the sum (since summation is over trajectories, not $\theta$):

$$= \sum_{\tau} \nabla_\theta P(\tau; \theta) \cdot R(\tau)$$

**Important subtlety:** You might think "$\tau$ depends on $\theta$ because we sampled it using $\pi_\theta$!" — and you're right that *which trajectories we observe* depends on $\theta$. But that dependence is captured in $P(\tau; \theta)$, not in $R(\tau)$.

Here's the key: we're summing over **all possible trajectories** $\tau$. For each fixed $\tau$, $R(\tau)$ is just a number (sum of rewards in that trajectory). The function $R(\cdot)$ doesn't contain $\theta$ — it just adds up rewards. What $\theta$ affects is **how likely** each trajectory is, which is exactly $P(\tau; \theta)$.

Think of it this way:
- $R(\tau)$ answers: "If trajectory $\tau$ happened, what's the total reward?" (no $\theta$ here)
- $P(\tau; \theta)$ answers: "How likely is trajectory $\tau$ under policy $\pi_\theta$?" ($\theta$ is here)

So when we differentiate, $R(\tau)$ is treated as a constant with respect to $\theta$, and all the $\theta$-dependence flows through $P(\tau; \theta)$.

### Step 5: Apply the Log-Derivative Trick

Here's the key insight. For any function $f(x) > 0$:

$$\nabla \log f(x) = \frac{\nabla f(x)}{f(x)}$$

Rearranging:

$$\nabla f(x) = f(x) \cdot \nabla \log f(x)$$

Apply this to $P(\tau; \theta)$:

$$\nabla_\theta P(\tau; \theta) = P(\tau; \theta) \cdot \nabla_\theta \log P(\tau; \theta)$$

Substitute back:

$$\nabla_\theta J(\theta) = \sum_{\tau} P(\tau; \theta) \cdot \nabla_\theta \log P(\tau; \theta) \cdot R(\tau)$$

This is now an expectation again!

$$= \mathbb{E}_{\tau \sim \pi_\theta} \left[ \nabla_\theta \log P(\tau; \theta) \cdot R(\tau) \right]$$

### Step 6: Simplify $\nabla_\theta \log P(\tau; \theta)$

Take the log of the trajectory probability:

$$\log P(\tau; \theta) = \log p(s_0) + \sum_{t=0}^{T-1} \log \pi_\theta(a_t|s_t) + \sum_{t=0}^{T-1} \log p(s_{t+1}|s_t, a_t)$$

Now take the gradient with respect to $\theta$:

$$\nabla_\theta \log P(\tau; \theta) = \underbrace{\nabla_\theta \log p(s_0)}_{= 0} + \sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_t|s_t) + \underbrace{\sum_{t=0}^{T-1} \nabla_\theta \log p(s_{t+1}|s_t, a_t)}_{= 0}$$

The first and third terms are **zero** because $p(s_0)$ and $p(s_{t+1}|s_t, a_t)$ are environment dynamics — they don't depend on $\theta$!

This leaves us with:

$$\nabla_\theta \log P(\tau; \theta) = \sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_t|s_t)$$

### Step 7: Put It All Together

Substituting back into our gradient:

$$\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \left( \sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_t|s_t) \right) \cdot R(\tau) \right]$$

Expanding $R(\tau) = \sum_{t'=1}^{T} r_{t'}$:

$$\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \left( \sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_t|s_t) \right) \cdot \left( \sum_{t'=1}^{T} r_{t'} \right) \right]$$

### Step 8: Causality Refinement (Optional but Important)

The action $a_t$ at time $t$ can only affect rewards at time $t+1$ and later — it can't affect past rewards. So we can replace $R(\tau)$ with the "reward-to-go" $G_t$:

$$G_t = \sum_{k=t}^{T-1} \gamma^{k-t} r_{k+1} = r_{t+1} + \gamma r_{t+2} + \gamma^2 r_{t+3} + \cdots$$

This gives the final **REINFORCE gradient**:

$$\boxed{\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot G_t \right]}$$

### Summary of What We Did

| Step | What We Did |
|------|-------------|
| 1 | Defined objective as expected total reward |
| 2 | Wrote expectation as probability-weighted sum |
| 3 | Expanded trajectory probability into policy × dynamics |
| 4 | Took gradient, noting only $P(\tau;\theta)$ depends on $\theta$ |
| 5 | Used log-derivative trick to get expectation form |
| 6 | Showed environment terms vanish, only policy terms remain |
| 7 | Combined everything |
| 8 | Used causality to replace $R(\tau)$ with $G_t$ |
