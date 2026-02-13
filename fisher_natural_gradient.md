# Fisher Information Matrix & Natural Gradients

## The Problem with Regular Gradients

Consider optimizing a function $J(\theta)$ with gradient descent:

$$
\theta_{\text{new}} = \theta_{\text{old}} + \alpha \nabla_\theta J
$$

This seems natural, but there's a hidden problem: **the gradient depends on your choice of parameterization.**

### A Simple Example

Suppose you're fitting a Gaussian distribution to data. You could parameterize it as:
- **Option A**: $\theta = \sigma$ (standard deviation)
- **Option B**: $\phi = \sigma^2$ (variance)
- **Option C**: $\psi = \log \sigma$ (log standard deviation)

These all represent the *same* family of distributions. But the gradient $\nabla_\theta J$, $\nabla_\phi J$, and $\nabla_\psi J$ point in **different directions** in parameter space!

Why? Because the gradient is defined as "the direction of steepest ascent in parameter space." But parameter space is just an arbitrary coordinate system we invented. Stretching or squashing coordinates changes what "steepest" means.

### Why This Matters

Imagine two researchers optimizing the same model:
- Alice uses parameterization A
- Bob uses parameterization B (which is just A with different coordinates)

They should get the same results, right? But with regular gradient descent:
- They'll take different step sizes in "distribution space"
- They'll converge at different rates
- One might work great while the other fails

This is bad. The algorithm's behavior shouldn't depend on arbitrary coordinate choices.

---

## What Does "Invariant to Parameterization" Mean?

A method is **parameterization-invariant** if:

> Changing how you represent the same family of functions doesn't change what the algorithm actually does in function space.

For probability distributions: if two parameterizations describe the same set of distributions, the algorithm should move through distribution space the same way, regardless of which parameterization you use.

### The Coordinate System Analogy

Think of navigating on Earth:
- You could use (latitude, longitude)
- You could use (x, y, z) Cartesian coordinates
- You could use some weird projection

"Walk 1 unit north" should mean the same thing regardless of your coordinate system. But if you naively say "increase the first coordinate by 0.01," you'll end up in different places depending on your coordinates!

Regular gradients are like "increase each coordinate proportionally to the slope." Natural gradients are like "walk in the direction of steepest ascent on the actual surface."

---

## The Fisher Information Matrix

The Fisher Information Matrix (FIM) solves this by defining a **metric** on probability distribution space.

### Definition

For a distribution $p(x|\theta)$ parameterized by $\theta$:

$$
F_{ij} = \mathbb{E}_{x \sim p(x|\theta)} \left[ \frac{\partial \log p(x|\theta)}{\partial \theta_i} \frac{\partial \log p(x|\theta)}{\partial \theta_j} \right]
$$

Or equivalently:

$$
F = \mathbb{E}\left[ \nabla_\theta \log p(x|\theta) \cdot \nabla_\theta \log p(x|\theta)^T \right]
$$

### What Does It Measure?

The FIM measures **how sensitive the distribution is to parameter changes**.

- If changing $\theta_i$ barely changes the distribution → small $F_{ii}$
- If changing $\theta_i$ drastically changes the distribution → large $F_{ii}$
- If changing $\theta_i$ and $\theta_j$ have correlated effects → nonzero $F_{ij}$

### Connection to KL Divergence

Here's the key insight. For nearby distributions:

$$
D_{\text{KL}}(p_\theta \| p_{\theta + \Delta\theta}) \approx \frac{1}{2} \Delta\theta^T F \Delta\theta
$$

The FIM is the **Hessian of KL divergence**! It tells you how "far apart" two distributions are in an information-theoretic sense.

This is why $F$ defines a metric on distribution space—it measures distance in terms of how different the distributions actually are, not how different the parameters look.

---

## Natural Gradient: The Derivation

### What We Want

Instead of asking "what's the steepest direction in parameter space?", ask:

> "What parameter change gives the steepest ascent in distribution space, subject to moving a fixed distance in distribution space?"

Mathematically, we want to solve:

$$
\max_{\Delta\theta} \quad \nabla_\theta J \cdot \Delta\theta \quad \text{subject to} \quad \frac{1}{2}\Delta\theta^T F \Delta\theta = c
$$

This says: maximize the improvement (linear approximation), but only move a fixed "distance" $c$ in distribution space (measured by the FIM).

### The Solution

Using Lagrange multipliers, the solution is:

$$
\Delta\theta \propto F^{-1} \nabla_\theta J
$$

This is the **natural gradient**:

$$
\tilde{\nabla}_\theta J = F^{-1} \nabla_\theta J
$$

### Intuition

The natural gradient "pre-warps" the regular gradient to account for the geometry of distribution space:

- If a parameter has a big effect on the distribution (large $F_{ii}$), take a smaller step in that parameter
- If a parameter has a small effect (small $F_{ii}$), take a larger step
- Account for correlations between parameters (off-diagonal terms)

---

## Why Natural Gradients Are Parameterization-Invariant

This is the key question. Here's the proof sketch:

### Setup

Say we have two parameterizations related by a smooth invertible transformation:
$$
\phi = h(\theta)
$$

The same distribution can be written as $p(x|\theta)$ or $p(x|\phi)$ where $\phi = h(\theta)$.

### How Things Transform

**The gradient transforms with the Jacobian:**
$$
\nabla_\phi J = (J_h^{-1})^T \nabla_\theta J
$$

where $J_h = \frac{\partial h}{\partial \theta}$ is the Jacobian of the transformation.

**The Fisher matrix transforms as a metric tensor:**
$$
F_\phi = (J_h^{-1})^T F_\theta (J_h^{-1})
$$

### The Natural Gradient Is Invariant

Now compute the natural gradient in both parameterizations:

In $\theta$-space:
$$
\tilde{\nabla}_\theta J = F_\theta^{-1} \nabla_\theta J
$$

In $\phi$-space:
$$
\tilde{\nabla}_\phi J = F_\phi^{-1} \nabla_\phi J
$$

These are related by:
$$
\tilde{\nabla}_\phi J = J_h \cdot \tilde{\nabla}_\theta J
$$

This is exactly how a **tangent vector** should transform! The natural gradient represents the same "direction" on the distribution manifold, just expressed in different coordinates.

### What This Means Practically

If Alice uses $\theta$ and Bob uses $\phi = h(\theta)$:
- Their regular gradients point in "different directions" (incompatible under coordinate change)
- Their natural gradients point in the "same direction" (transform correctly as tangent vectors)
- They'll trace out the same path through distribution space
- They'll get the same sequence of distributions, just with different parameter values

---

## A Concrete Example: Variance Parameterization

Let's see this with numbers.

### Setup

Fit a zero-mean Gaussian. We'll use two parameterizations:
- $\theta = \sigma$ (standard deviation)
- $\phi = \sigma^2$ (variance), so $\phi = \theta^2$

The distribution is $p(x|\sigma) = \frac{1}{\sqrt{2\pi}\sigma} e^{-x^2/2\sigma^2}$.

### Compute the Fisher Information

**In $\theta = \sigma$ parameterization:**

$$
\log p(x|\sigma) = -\log\sigma - \frac{x^2}{2\sigma^2} + \text{const}
$$

$$
\frac{\partial}{\partial \sigma} \log p = -\frac{1}{\sigma} + \frac{x^2}{\sigma^3}
$$

$$
F_\theta = \mathbb{E}\left[\left(-\frac{1}{\sigma} + \frac{x^2}{\sigma^3}\right)^2\right] = \frac{2}{\sigma^2}
$$

**In $\phi = \sigma^2$ parameterization:**

Using the transformation rule with $J_h = \frac{d\phi}{d\theta} = 2\sigma$:

$$
F_\phi = \frac{F_\theta}{(2\sigma)^2} = \frac{2/\sigma^2}{4\sigma^2} = \frac{1}{2\sigma^4} = \frac{1}{2\phi^2}
$$

### Compare Gradients

Suppose our objective gradient is:
- $\nabla_\theta J = g_\theta$ in $\sigma$-space
- $\nabla_\phi J = g_\phi = \frac{g_\theta}{2\sigma}$ in $\sigma^2$-space (chain rule)

**Regular gradients:**
- In $\theta$-space: step $\propto g_\theta$
- In $\phi$-space: step $\propto g_\phi = \frac{g_\theta}{2\sigma}$

These give *different* changes to the distribution for the same step size!

**Natural gradients:**
- In $\theta$-space: $\tilde{\nabla}_\theta J = F_\theta^{-1} g_\theta = \frac{\sigma^2}{2} g_\theta$
- In $\phi$-space: $\tilde{\nabla}_\phi J = F_\phi^{-1} g_\phi = 2\phi^2 \cdot \frac{g_\theta}{2\sigma} = 2\sigma^4 \cdot \frac{g_\theta}{2\sigma} = \sigma^3 g_\theta$

Check the relationship: $\tilde{\nabla}_\phi J = 2\sigma \cdot \tilde{\nabla}_\theta J$? 

$$
\sigma^3 g_\theta = 2\sigma \cdot \frac{\sigma^2}{2} g_\theta = \sigma^3 g_\theta \quad \checkmark
$$

Yes! The natural gradients transform correctly. Taking a natural gradient step in either parameterization moves the distribution by the same amount.

---

## Why This Matters for TRPO

In TRPO, we're optimizing a policy $\pi_\theta(a|s)$. The policy could be parameterized many ways:
- Direct softmax weights
- Log-probabilities
- Some neural network architecture
- A completely different network with the same expressiveness

With regular policy gradients, the learning dynamics would depend on these arbitrary choices. With the natural gradient (approximated via the Fisher matrix and conjugate gradient), TRPO's updates are determined by how the *policy distribution* changes, not by the accident of how we parameterized it.

This is one reason TRPO is more stable than vanilla policy gradient—it's doing something geometrically meaningful rather than depending on coordinate artifacts.

---

## Summary

| | Regular Gradient | Natural Gradient |
|---|---|---|
| **Definition** | $\nabla_\theta J$ | $F^{-1} \nabla_\theta J$ |
| **Measures steepest ascent in** | Parameter space | Distribution space |
| **Depends on parameterization** | Yes | No |
| **Accounts for parameter sensitivity** | No | Yes |
| **Computational cost** | Cheap | Expensive (need $F^{-1}$) |

**The core insight:** The regular gradient tells you how to change *parameters* fastest. The natural gradient tells you how to change the *distribution* fastest. Since we care about the distribution (the policy), not the parameters (arbitrary numbers), the natural gradient is the "right" thing to follow.

---

## Further Reading

- Amari, S. (1998). "Natural Gradient Works Efficiently in Learning" - The original paper introducing natural gradients
- Martens, J. (2014). "New insights and perspectives on the natural gradient method" - Deep dive into the theory
- Kakade, S. (2001). "A Natural Policy Gradient" - Application to RL that led to TRPO
