# Curvature of Plane Curves – Derivation

Here is a clean, step-by-step derivation of the two most common curvature formulas for plane curves.

We derive both:

1. The **parametric form** (most general):  
   κ = \frac{|x' y'' - y' x''|}{(x'^2 + y'^2)^{3/2}}

2. The **graph form** y = f(x):  
   κ = \frac{|y''|}{(1 + (y')^2)^{3/2}}

## Starting Point: Intrinsic Definition of Curvature

Curvature κ measures **how fast the tangent direction changes per unit arc length**.

Let **r**(t) = (x(t), y(t)) be a smooth parametric curve (twice differentiable, with r'(t) ≠ 0).

Let **s** be **arc length** from some starting point.

The **unit tangent vector** is:

**T**(s) = dr/ds

By definition (Frenet), curvature is:

κ(s) = || d**T**/ds ||

(This is always ≥ 0; signed curvature adds a direction later if needed.)

## Step 1: Express everything in terms of parameter t (not s)

We know the chain rule:

dr/ds = (dr/dt) / (ds/dt)  ⟹  **T**(s) = **r**'(t) / ||**r**'(t)||

Let v(t) = ||r'(t)|| = √(x'² + y'²)  (speed)

Then:

**T**(t) = **r**'(t) / v(t)

Now differentiate **T** with respect to **s**:

d**T**/ds = (d**T**/dt) ⋅ (dt/ds) = (d**T**/dt) ⋅ (1/v)

So:

κ = || d**T**/ds || = (1/v) || d**T**/dt ||

## Step 2: Compute dT/dt

**T** = **r**' /