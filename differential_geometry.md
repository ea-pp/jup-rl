# Differential Geometry: Curvature, Christoffel Symbols, and Background Independence

This document explains the geometric foundations underlying concepts like natural gradients and the Fisher information matrix.

---

## The Big Picture

We want to do calculus on curved spaces (manifolds). On flat space (like $\mathbb{R}^n$), this is easy—derivatives just work. On curved spaces, we need new machinery:

1. **Metric tensor** ($g$): Defines distances and angles
2. **Christoffel symbols** ($\Gamma$): Tell us how to take derivatives that respect the geometry
3. **Riemann tensor** ($R$): Measures intrinsic curvature
4. **Background independence**: The physics/geometry shouldn't depend on arbitrary coordinate choices

The punchline: **Curvature is what remains when you've removed all coordinate artifacts. It's encoded in second derivatives of the metric because first derivatives can always be "gauged away" locally.**

---

## Part 1: Why Flat-Space Calculus Breaks on Curved Spaces

### The Problem with Regular Derivatives

On flat $\mathbb{R}^n$, if you have a vector field $V^\mu(x)$ and want its derivative, you just compute:

$$
\partial_\nu V^\mu = \frac{\partial V^\mu}{\partial x^\nu}
$$

This works because the basis vectors $\hat{e}_\mu$ are constant everywhere—they don't change as you move around.

**On a curved manifold, basis vectors change from point to point.**

Imagine you're on a sphere. At the equator, "east" points one direction. Walk to the north pole—"east" now points a completely different direction! The basis vectors themselves have rotated.

If you naively compute $\partial_\nu V^\mu$, you're mixing up:
1. How the components $V^\mu$ change
2. How the basis vectors change

This gives a coordinate-dependent mess, not a geometric quantity.

### What We Need

We need a **covariant derivative** $\nabla_\nu V^\mu$ that:
1. Accounts for how basis vectors change
2. Gives the same geometric answer regardless of coordinates
3. Reduces to the ordinary derivative on flat space

---

## Part 2: The Metric Tensor

### Definition

The **metric tensor** $g_{\mu\nu}(x)$ defines the inner product of tangent vectors at each point:

$$
\langle U, V \rangle = g_{\mu\nu} U^\mu V^\nu
$$

It tells you:
- **Distances**: $ds^2 = g_{\mu\nu} dx^\mu dx^\nu$
- **Angles**: via the inner product
- **How to raise/lower indices**: $V_\mu = g_{\mu\nu} V^\nu$

### Examples

**Flat 2D space (Cartesian):**
$$
g_{\mu\nu} = \begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix}
$$

**Flat 2D space (polar coordinates):**
$$
g_{\mu\nu} = \begin{pmatrix} 1 & 0 \\ 0 & r^2 \end{pmatrix}
$$

Same flat space, different metric components! The metric depends on coordinates, but distances don't.

**Sphere of radius $R$:**
$$
g_{\mu\nu} = \begin{pmatrix} R^2 & 0 \\ 0 & R^2 \sin^2\theta \end{pmatrix}
$$

This space is intrinsically curved—no coordinate change makes $g_{\mu\nu}$ constant.

---

## Part 3: Christoffel Symbols

### What They Represent

Christoffel symbols $\Gamma^\lambda_{\mu\nu}$ encode **how the basis vectors change** as you move around:

$$
\partial_\mu \vec{e}_\nu = \Gamma^\lambda_{\mu\nu} \vec{e}_\lambda
$$

"The derivative of basis vector $\vec{e}_\nu$ in the $\mu$ direction is a linear combination of basis vectors, with coefficients $\Gamma^\lambda_{\mu\nu}$."

### Formula (Levi-Civita Connection)

For the standard "metric-compatible, torsion-free" connection:

$$
\Gamma^\lambda_{\mu\nu} = \frac{1}{2} g^{\lambda\sigma} \left( \partial_\mu g_{\nu\sigma} + \partial_\nu g_{\mu\sigma} - \partial_\sigma g_{\mu\nu} \right)
$$

Key observation: **Christoffel symbols involve first derivatives of the metric.**

### Example: Polar Coordinates on Flat Space

With $g_{rr} = 1$, $g_{\theta\theta} = r^2$, $g_{r\theta} = 0$:

$$
\Gamma^r_{\theta\theta} = -r, \quad \Gamma^\theta_{r\theta} = \Gamma^\theta_{\theta r} = \frac{1}{r}
$$

Even on flat space, Christoffel symbols can be nonzero! They just reflect that polar coordinates have non-constant basis vectors (the $\hat{\theta}$ direction changes as you move).

### The Covariant Derivative

Now we can define the proper derivative:

$$
\nabla_\nu V^\mu = \partial_\nu V^\mu + \Gamma^\mu_{\nu\lambda} V^\lambda
$$

The $\Gamma$ term corrects for the changing basis vectors.

---

## Part 4: The Riemann Curvature Tensor

### The Key Question

Christoffel symbols can be nonzero even on flat space (like polar coordinates). How do we detect *actual* curvature vs. just funny coordinates?

**The Riemann tensor answers this.**

### Intuitive Definition: Parallel Transport Around a Loop

Take a vector, parallel transport it around a small closed loop. On flat space, it comes back unchanged. On curved space, it comes back rotated!

The Riemann tensor measures this rotation:

$$
[\nabla_\mu, \nabla_\nu] V^\rho = R^\rho_{\sigma\mu\nu} V^\sigma
$$

"The commutator of covariant derivatives (going around a small loop) rotates vectors proportionally to $R$."

### Formula

$$
R^\rho_{\sigma\mu\nu} = \partial_\mu \Gamma^\rho_{\nu\sigma} - \partial_\nu \Gamma^\rho_{\mu\sigma} + \Gamma^\rho_{\mu\lambda}\Gamma^\lambda_{\nu\sigma} - \Gamma^\rho_{\nu\lambda}\Gamma^\lambda_{\mu\sigma}
$$

Key observation: **The Riemann tensor involves derivatives of Christoffel symbols, hence second derivatives of the metric.**

### The Crucial Property

**On flat space, $R^\rho_{\sigma\mu\nu} = 0$ in ALL coordinate systems.**

**On curved space, $R^\rho_{\sigma\mu\nu} \neq 0$ in ALL coordinate systems.**

You cannot make curvature appear or disappear by changing coordinates. It's intrinsic.

---

## Part 5: Why Curvature Lives in Second Derivatives

This is the heart of your question. Here's the deep reason:

### First Derivatives Can Be Gauged Away (Locally)

At any single point $p$, you can always find coordinates where:

$$
\Gamma^\lambda_{\mu\nu}(p) = 0
$$

These are called **normal coordinates** or **Riemann normal coordinates** at $p$.

Proof sketch: The $\Gamma$'s depend on first derivatives of $g$. You have enough freedom in choosing coordinates to set these first derivatives to zero at one point.

**This means:** First derivatives of the metric don't contain coordinate-independent information at a point—they can always be transformed away.

### Second Derivatives Cannot Be Removed

Even in normal coordinates where $\Gamma(p) = 0$, the **derivatives of $\Gamma$** (which give the Riemann tensor) generally don't vanish.

You've used up all your coordinate freedom making $\Gamma = 0$ at $p$. You can't also make $\partial\Gamma = 0$.

**This means:** Second derivatives of the metric contain genuine, coordinate-independent information: the curvature.

### Physical Intuition

Think of a bumpy surface:
- At any point, you can tilt your coordinates so the surface looks locally flat (first derivative = 0)
- But you can't flatten out the bumps—the second derivative (curvature) remains

The surface of a sphere looks flat if you zoom in enough (first derivative zero), but parallel transport still detects the curvature (second derivative nonzero).

---

## Part 6: Background Independence

### What It Means

**Background independence** = The fundamental laws/geometry shouldn't depend on arbitrary choices (like coordinate systems).

In general relativity: the equations of physics should have the same form in all coordinate systems (general covariance).

In our context (natural gradients): the learning update should depend on the distributions themselves, not on how we parameterized them.

### The Connection to Derivatives

| Derivative Order | Geometric Status | Can Gauge Away? |
|------------------|------------------|-----------------|
| 0th (metric values) | Coordinate-dependent | Yes (at a point) |
| 1st (Christoffel) | Coordinate-dependent | Yes (at a point) |
| 2nd (Riemann) | **Coordinate-independent** | **No** |

- **0th order** ($g_{\mu\nu}$ values): Depend on coordinates. At any point, you can choose coordinates where $g_{\mu\nu} = \delta_{\mu\nu}$ (looks Euclidean locally).

- **1st order** ($\partial g$, or $\Gamma$): Still coordinate-dependent. Can be set to zero at any single point using normal coordinates.

- **2nd order** ($\partial^2 g$, or $R$): **Coordinate-independent!** This is the curvature. It's "background independent"—it's a property of the space itself, not our description of it.

### Why This Matters

When we want coordinate/parameterization-independent quantities, we need to look at:
- Scalars constructed from the Riemann tensor (like the Ricci scalar $R = g^{\mu\nu}R_{\mu\nu}$)
- Objects that transform properly (tensors)
- Second-order information rather than first-order

The Fisher metric gives us a natural inner product on distribution space. The curvature of this space (computed from second derivatives of the Fisher metric) tells us about intrinsic properties of the distribution manifold that don't depend on parameterization.

---

## Part 7: Connection to Fisher Matrix and Natural Gradients

### The Statistical Manifold

The space of probability distributions forms a manifold. Each point is a distribution; coordinates are parameters $\theta$.

**The Fisher information matrix is the metric tensor on this manifold:**

$$
F_{\mu\nu}(\theta) = \mathbb{E}\left[\frac{\partial \log p(x|\theta)}{\partial \theta_\mu} \frac{\partial \log p(x|\theta)}{\partial \theta_\nu}\right]
$$

This is a Riemannian metric! It defines distances between nearby distributions:

$$
ds^2 = F_{\mu\nu} d\theta^\mu d\theta^\nu \approx 2 D_{\text{KL}}(p_\theta \| p_{\theta+d\theta})
$$

### Natural Gradient and Curvature

The natural gradient $F^{-1}\nabla J$:
- Uses the Fisher metric to measure distances
- Is the steepest ascent direction in distribution space
- Is coordinate (parameterization) independent

The Christoffel symbols of the Fisher metric tell you how the "natural" directions change as you move through distribution space.

The Riemann curvature of the statistical manifold measures intrinsic properties of the distribution family—this is the subject of **information geometry**.

### Why Second Derivatives Matter Here Too

- First derivatives of the Fisher metric → Christoffel symbols → can be made zero at a point
- Second derivatives of the Fisher metric → Curvature of statistical manifold → intrinsic, can't be gauged away

The curvature of the statistical manifold affects:
- How "curved" the path of natural gradient descent is
- Whether there are geodesics (shortest paths) between distributions
- The convergence properties of learning algorithms

---

## Summary

| Concept | What It Is | Derivative Order | Coordinate-Independent? |
|---------|-----------|------------------|------------------------|
| Metric $g_{\mu\nu}$ | Inner product, distances | 0 | No |
| Christoffel $\Gamma$ | How basis vectors change | 1st of $g$ | No |
| Riemann $R$ | Intrinsic curvature | 2nd of $g$ | **Yes** |

**The punchline:**
- First derivatives of the metric ($\Gamma$) can be transformed away at any point—they're coordinate artifacts
- Second derivatives ($R$) cannot—they represent genuine curvature
- Background independence means focusing on coordinate-independent quantities
- This is why curvature (the thing that's "really there") lives in second derivatives

For natural gradients and TRPO:
- The Fisher matrix is a metric on distribution space
- Natural gradients use this metric for parameterization-invariant updates
- The deeper geometry (curvature of the statistical manifold) influences learning dynamics

---

## Notation Reference

- $\partial_\mu = \frac{\partial}{\partial x^\mu}$: Ordinary partial derivative
- $\nabla_\mu$: Covariant derivative (accounts for geometry)
- $g_{\mu\nu}$: Metric tensor (defines inner product)
- $g^{\mu\nu}$: Inverse metric
- $\Gamma^\lambda_{\mu\nu}$: Christoffel symbols (connection coefficients)
- $R^\rho_{\sigma\mu\nu}$: Riemann curvature tensor
- Einstein summation: repeated indices are summed ($V^\mu U_\mu = \sum_\mu V^\mu U_\mu$)
