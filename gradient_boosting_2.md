# Gradient Boosting II: The Functional Gradient View

## From Residuals to Gradients

In the previous part, we discovered that the residual $r_i = y_i - f(x_i)$ for squared error is actually the negative gradient of the loss with respect to the prediction:
$$
r_i = -\frac{\partial L(y_i, f(x_i))}{\partial f(x_i)}
$$

This observation seems almost too convenient. But it suggests a profound reinterpretation of what boosting is doing. Instead of thinking about "fitting residuals," we should think about "*gradient descent in function space*."

To make this precise, we need to develop some new conceptual machinery. We need to understand:
1. What it means to view functions as points in a space
2. How gradients work in function space
3. How to approximate functional gradients with finite learners

Let's build this understanding carefully.

## Functions as Points in a Space

When we do ordinary gradient descent, we have parameters $\theta \in \mathbb{R}^d$ and a loss function $\mathcal{L}(\theta)$. We compute:
$$
\theta_{t+1} = \theta_t - \eta \nabla_\theta \mathcal{L}(\theta_t)
$$

The gradient $\nabla_\theta \mathcal{L}$ points in the direction of steepest ascent in parameter space, and we step in the opposite direction to decrease the loss.

Now, instead of optimizing over parameters $\theta$, imagine we're optimizing directly over *functions* $f$. Our loss is:
$$
\mathcal{L}(f) = \frac{1}{n}\sum_{i=1}^n L(y_i, f(x_i))
$$

where $L(y, \hat{y})$ is a loss function like squared error, logistic loss, etc.

We can think of $f$ as living in some function space $\mathcal{F}$. Just as $\theta \in \mathbb{R}^d$ is a point in parameter space, $f \in \mathcal{F}$ is a "point" in function space. This is a big conceptual leap—we're used to thinking of functions as objects that *do* things (they map inputs to outputs), not as points in a space. But for optimization purposes, this is exactly the right perspective.

## The Functional Gradient

If we want to do gradient descent in function space, we need to define what a gradient means there. In parameter space, the gradient $\nabla_\theta \mathcal{L}$ tells us how the loss changes when we perturb each parameter. In function space, we need something analogous: how does the loss change when we perturb the *function*?

Consider a small perturbation $f + \epsilon h$ where $h$ is some other function and $\epsilon$ is small. The loss changes as:
$$
\mathcal{L}(f + \epsilon h) = \frac{1}{n}\sum_{i=1}^n L(y_i, f(x_i) + \epsilon h(x_i))
$$

Taking the derivative with respect to $\epsilon$ and evaluating at $\epsilon = 0$:
$$
\frac{d}{d\epsilon}\mathcal{L}(f + \epsilon h)\bigg|_{\epsilon=0} = \frac{1}{n}\sum_{i=1}^n \frac{\partial L(y_i, f(x_i))}{\partial f(x_i)} h(x_i)
$$

This is the *directional derivative* of $\mathcal{L}$ at $f$ in the direction $h$. It tells us how much the loss changes if we move from $f$ toward $f + h$.

Now, in finite dimensions, the gradient is the unique vector such that:
$$
\nabla_\theta \mathcal{L} \cdot v = D_v \mathcal{L}
$$

where $D_v \mathcal{L}$ is the directional derivative in direction $v$.

By analogy, we want a functional gradient $\nabla_f \mathcal{L}$ such that:
$$
\langle \nabla_f \mathcal{L}, h \rangle = \frac{1}{n}\sum_{i=1}^n \frac{\partial L(y_i, f(x_i))}{\partial f(x_i)} h(x_i)
$$

Here $\langle \cdot, \cdot \rangle$ is an inner product on function space. If we use the empirical inner product:
$$
\langle g, h \rangle = \frac{1}{n}\sum_{i=1}^n g(x_i) h(x_i)
$$

then the functional gradient is the function $g$ such that:
$$
\frac{1}{n}\sum_{i=1}^n g(x_i) h(x_i) = \frac{1}{n}\sum_{i=1}^n \frac{\partial L(y_i, f(x_i))}{\partial f(x_i)} h(x_i)
$$

for all $h$. This means:
$$
g(x_i) = \frac{\partial L(y_i, f(x_i))}{\partial f(x_i)}
$$

So the functional gradient, evaluated at training point $x_i$, is simply the partial derivative of the loss with respect to the prediction at that point.

### Making It Concrete

Let's write this explicitly:
$$
\nabla_f \mathcal{L} = \left[ \frac{\partial L(y_1, f(x_1))}{\partial f(x_1)}, \ldots, \frac{\partial L(y_n, f(x_n))}{\partial f(x_n)} \right]
$$

This looks like a vector in $\mathbb{R}^n$, and in a sense it is—but conceptually, it represents a function. We only have $n$ values because we only have $n$ training points. The function is defined by how it behaves at the training points, just like in standard supervised learning.

For squared error $L(y, \hat{y}) = \frac{1}{2}(y - \hat{y})^2$:
$$
\frac{\partial L(y_i, f(x_i))}{\partial f(x_i)} = -(y_i - f(x_i))
$$

So the negative functional gradient is:
$$
-\nabla_f \mathcal{L} = [y_1 - f(x_1), \ldots, y_n - f(x_n)]
$$

These are exactly the residuals! This confirms our earlier observation: for squared error, the residuals *are* the negative functional gradient.

## Functional Gradient Descent

Now we can write down functional gradient descent formally:
$$
f_{m+1} = f_m - \eta \nabla_f \mathcal{L}(f_m)
$$

But there's a problem: $\nabla_f \mathcal{L}(f_m)$ is a function (or rather, a vector of values at training points), but we typically want our model $f_{m+1}$ to be in some restricted class $\mathcal{H}$ (like trees, or neural networks, or whatever).

We can't usually add an arbitrary function to $f_m$ and stay in our model class. This is fundamentally different from parametric gradient descent, where $\theta - \eta \nabla_\theta \mathcal{L}$ is always a valid parameter vector.

## The Approximation Step

Here's the key insight: we *project* the negative functional gradient onto our hypothesis space $\mathcal{H}$. We find the function in $\mathcal{H}$ that best approximates the negative gradient.

Specifically, we solve:
$$
h_m = \arg\min_{h \in \mathcal{H}} \| h - (-\nabla_f \mathcal{L}(f_m)) \|^2
$$

where the norm is with respect to our empirical inner product:
$$
\|g\|^2 = \frac{1}{n}\sum_{i=1}^n g(x_i)^2
$$

Expanding this:
$$
h_m = \arg\min_{h \in \mathcal{H}} \frac{1}{n}\sum_{i=1}^n \left(h(x_i) + \frac{\partial L(y_i, f_m(x_i))}{\partial f_m(x_i)}\right)^2
$$

But notice: minimizing this is equivalent to minimizing:
$$
h_m = \arg\min_{h \in \mathcal{H}} \sum_{i=1}^n \left(h(x_i) - \left(-\frac{\partial L(y_i, f_m(x_i))}{\partial f_m(x_i)}\right)\right)^2
$$

We're fitting $h$ to the negative gradients using squared error! The negative gradients are playing the role of "pseudo-residuals" or "pseudo-responses."

Then we update:
$$
f_{m+1} = f_m + \nu h_m
$$

where $\nu > 0$ is a step size (learning rate).

## The Complete Algorithm

This gives us gradient boosting for arbitrary differentiable losses:

**Gradient Boosting Algorithm:**

1. Initialize $f_0(x) = \arg\min_c \sum_{i=1}^n L(y_i, c)$ (constant that minimizes loss)

2. For $m = 0, 1, \ldots, M-1$:

   a. Compute pseudo-residuals:
   $$
   r_i^{(m)} = -\frac{\partial L(y_i, f_m(x_i))}{\partial f_m(x_i)} \quad \text{for } i = 1, \ldots, n
   $$

   b. Fit a base learner to pseudo-residuals:
   $$
   h_m = \arg\min_{h \in \mathcal{H}} \sum_{i=1}^n (r_i^{(m)} - h(x_i))^2
   $$

   c. Compute step size (can use line search or fixed learning rate $\nu$):
   $$
   \rho_m = \arg\min_\rho \sum_{i=1}^n L(y_i, f_m(x_i) + \rho h_m(x_i))
   $$

   d. Update:
   $$
   f_{m+1}(x) = f_m(x) + \rho_m h_m(x)
   $$

3. Return $f_M(x)$

### Key Observations

Let's highlight what's important here:

1. **Universality**: This works for *any* differentiable loss function $L(y, \hat{y})$. We just need to compute $\frac{\partial L}{\partial \hat{y}}$.

2. **Base learners fit gradients**: The base learners $h_m$ are always fit to pseudo-residuals using squared error, regardless of the overall loss function.

3. **Two-stage optimization**: We first find the best direction (the base learner $h_m$), then find the best step size ($\rho_m$) in that direction. These can be optimized separately.

4. **Connection to residual fitting**: For squared error, the pseudo-residuals are exactly the residuals, so we recover our original algorithm.

## Examples: Different Loss Functions

Let's see what the pseudo-residuals look like for common losses.

### Squared Error
$$
L(y, \hat{y}) = \frac{1}{2}(y - \hat{y})^2
$$
$$
r_i = -\frac{\partial L}{\partial \hat{y}} = y_i - f(x_i)
$$

The ordinary residual.

### Absolute Error (for quantile regression at median)
$$
L(y, \hat{y}) = |y - \hat{y}|
$$
$$
r_i = -\frac{\partial L}{\partial \hat{y}} = \text{sign}(y_i - f(x_i))
$$

The sign of the residual. We fit the direction of the error, not its magnitude.

### Logistic Loss (for binary classification)
$$
L(y, \hat{y}) = \log(1 + e^{-y\hat{y}}) \quad \text{where } y \in \{-1, +1\}
$$
$$
r_i = -\frac{\partial L}{\partial \hat{y}} = \frac{y_i}{1 + e^{y_i f(x_i)}}
$$

This is bounded in $(-1, 1)$, giving robustness to outliers.

### Poisson Loss (for count data)
$$
L(y, \hat{y}) = e^{\hat{y}} - y\hat{y}
$$
$$
r_i = -\frac{\partial L}{\partial \hat{y}} = y_i - e^{f(x_i)}
$$

The difference between observed count and predicted expected count.

In each case, the pseudo-residual has a natural interpretation related to the prediction error, but adapted to the specific loss function's geometry.

## Why This Perspective Matters

The functional gradient view gives us much more than just an algorithm. It provides:

1. **Geometric intuition**: We're doing gradient descent, just in function space instead of parameter space.

2. **Principled generalization**: We're not heuristically adapting residual fitting to different losses—we're following a unified principle.

3. **Theoretical foundation**: The convergence analysis of gradient boosting connects directly to gradient descent theory.

4. **Design flexibility**: We can design custom losses for specific problems and immediately get a boosting algorithm.

5. **Connection to other methods**: We can see relationships to functional gradient descent methods in other areas (like functional gradient ascent in policy optimization).

## The Approximation Trade-off

There's a subtle but important point here. In parametric gradient descent, we take exact gradients. In functional gradient boosting, we *approximate* the gradient by projecting it onto $\mathcal{H}$.

This means:
- If $\mathcal{H}$ is rich (like deep trees), we approximate the gradient well but might overfit
- If $\mathcal{H}$ is simple (like shallow trees), we approximate the gradient poorly but gain regularization

This is actually a feature, not a bug. The constraint that $h_m \in \mathcal{H}$ acts as implicit regularization. We're doing "approximate gradient descent," which can generalize better than exact gradient descent.

The learning rate $\nu$ provides additional regularization—smaller values mean we trust each gradient approximation less, taking more conservative steps. This is the bias-variance trade-off manifesting in the boosting framework.

## The Interpolation Problem: How Gradients Propagate to Unseen Points

There's a subtlety we've been glossing over, and it's worth examining carefully because it reveals something fundamental about why the choice of hypothesis class matters so much.

We compute the functional gradient at $n$ training points: $\{x_1, \ldots, x_n\}$. This gives us $n$ scalar values:
$$
g_i = -\frac{\partial L(y_i, f_m(x_i))}{\partial f_m(x_i)} \quad \text{for } i = 1, \ldots, n
$$

Then we fit a function $h_m \in \mathcal{H}$ to these values. But here's the key question: *what is $h_m(x)$ for $x \notin \{x_1, \ldots, x_n\}$?*

We only have gradient information at training points. What happens at test points is entirely determined by how our hypothesis class $\mathcal{H}$ *interpolates* or *extrapolates* from the training data.

This might seem like a mundane point—isn't this just what supervised learning always does? But in the context of functional gradient descent, it's deeper than it appears. We're not just fitting arbitrary functions to arbitrary targets. We're approximating a *gradient direction* in function space, and how we extend that direction to unseen points fundamentally shapes the optimization trajectory.

### Inductive Bias as Gradient Propagation

Let's make this concrete with decision trees, the most common base learner for gradient boosting.

Suppose we fit a regression tree with maximum depth $d$ to the pseudo-residuals $\{(x_i, g_i)\}_{i=1}^n$. The tree partitions the input space $\mathbb{R}^p$ into regions $R_1, \ldots, R_K$ (where $K \leq 2^d$), and predicts a constant value in each region:
$$
h_m(x) = \sum_{k=1}^K c_k \mathbb{1}[x \in R_k]
$$

where $c_k$ is typically the mean of the pseudo-residuals in region $k$:
$$
c_k = \frac{1}{|R_k|} \sum_{x_i \in R_k} g_i
$$

Now, what's happening here? The tree is making a crucial inductive assumption: *all points in the same region should move in the same gradient direction*.

If training points $x_i$ and $x_j$ fall in the same leaf, their pseudo-residuals get averaged, and *any test point* $x$ in that leaf receives the same gradient correction. The tree's splitting criterion (typically minimizing squared error on $g_i$) is choosing which points to group together—it's deciding how gradient information should be shared across input space.

### Different Hypothesis Classes, Different Propagation

This perspective illuminates why the choice of $\mathcal{H}$ is so critical. Different hypothesis classes propagate gradient information in fundamentally different ways:

**Decision Trees**: Partition space into regions with constant predictions. Gradient information propagates via *similarity in feature space*—points that fall in the same region get the same update, regardless of their actual distance. A tree with axis-aligned splits imposes a strong prior: gradients should be constant within hyperrectangular regions.

**Smoothness-based methods** (like regression splines or kernel methods): Nearby points in input space receive similar gradient corrections. The inductive bias is continuity—the gradient direction should vary smoothly. A point's update is influenced by a *neighborhood* of training points, weighted by distance.

**Neural Networks**: Learn hierarchical feature representations. Gradient information propagates through learned features—points that activate similar neurons receive similar updates. The inductive bias is that gradient structure should align with compositional features.

**Linear Models**: Assume the gradient direction is a linear function of inputs:
$$
h_m(x) = w^\top x + b
$$
This is an extremely strong assumption: the correction should be the same hyperplane for all iterations.

### Why This Matters for Optimization

In standard gradient descent with parameters $\theta$, every component gets its own gradient. If we have 1000 parameters, we get 1000 gradient values, one for each parameter. The update is exact (modulo sampling noise in SGD).

In functional gradient boosting, we might have millions of possible inputs $x$, but only $n$ training points. The functional gradient is fundamentally *underspecified* away from training data. The hypothesis class $\mathcal{H}$ fills in the missing information by imposing structure.

This has profound implications:

1. **Generalization through structure**: The inductive bias of $\mathcal{H}$ determines which gradient directions are even *expressible*. A shallow tree can't represent a gradient that varies smoothly—it must discretize. This constraint acts as regularization.

2. **Sample efficiency**: If the true optimal gradient direction aligns with the inductive bias of $\mathcal{H}$, we can generalize from few examples. If you're learning a piecewise-constant function, trees are sample-efficient. If you're learning a smooth function, trees waste capacity.

3. **The depth hyperparameter**: When we increase tree depth, we're not just increasing model capacity—we're changing how finely we discretize the gradient field. Depth 1 assumes the gradient is constant everywhere. Depth 2 assumes at most 4 different gradient regions. Depth $d$ allows $2^d$ regions.

4. **Interaction between iterations**: Each $h_m$ propagates gradient information according to the same inductive bias. Over many iterations, this compounds. Trees naturally discover interactions: splits at iteration $m$ can depend on residual patterns created by previous trees.

### A Concrete Example

Consider a simple 1D problem. Our current model underpredicts in region $[0, 0.5]$ and overpredicts in $[0.5, 1]$. The pseudo-residuals reflect this:
- Points with $x < 0.5$: $g_i > 0$ (increase predictions)
- Points with $x > 0.5$: $g_i < 0$ (decrease predictions)

**A depth-1 tree** might split at $x = 0.5$, creating:
$$
h_m(x) = \begin{cases}
\bar{g}_{\text{left}} & \text{if } x < 0.5 \\
\bar{g}_{\text{right}} & \text{if } x \geq 0.5
\end{cases}
$$

Every point with $x < 0.5$ gets the same correction, the average gradient in that region. The tree has decided that all points to the left of 0.5 are similar enough to share a gradient direction.

**A linear model** would fit:
$$
h_m(x) = w \cdot x + b
$$

It assumes the gradient changes linearly across the input space. This is a very different structural assumption—it smoothly interpolates rather than discretely partitioning.

**A depth-3 tree** could have up to 8 regions, capturing finer gradient structure. Maybe there's a subregion in $[0, 0.25]$ that needs a different correction than $[0.25, 0.5]$.

In each case, the choice of hypothesis class determines:
- How we group training points
- How gradient information transfers to test points
- What patterns we can and cannot represent
- How we trade off between fitting the empirical gradient and generalizing

### The Deep Connection to Regularization

Here's the key insight: the approximation error in fitting the functional gradient isn't just about *accuracy*—it's about *generalization*.

If we could fit the functional gradient perfectly at training points and had a perfect interpolation to test points (say, knowing the true data distribution), we'd do exact gradient descent in function space. We'd minimize training loss perfectly and likely overfit catastrophically.

But we can't. Our base learners in $\mathcal{H}$ are limited. They impose structure. They force us to share gradient information across points in structured ways. This is exactly what we need for generalization.

The "approximation" in approximate functional gradient descent is not a limitation to overcome—it's a feature to exploit. The structure of $\mathcal{H}$ embeds prior knowledge about how function changes should propagate through input space.

This is why shallow trees work so well in practice. They're not good at fitting the functional gradient accurately. But they impose strong regularization through their coarse partitioning. They force the model to pool gradient information across large regions, preventing overfitting to noise in individual pseudo-residuals.

### Choosing the Right Inductive Bias

This perspective gives us a principled way to think about choosing base learners:

- **Trees**: Best when the true function has local discontinuities, interactions, and piecewise structure. The gradient field is naturally chunky.

- **Splines/Kernels**: Best when the true function is smooth. The gradient field should vary continuously.

- **Linear models**: Best when the relationship is approximately linear, or as a starting point before adding trees for nonlinearities.

- **Neural networks**: Best when there are hierarchical features. The gradient field has compositional structure.

The power of boosting is that we iteratively refine. Early iterations might learn coarse structure with shallow trees. Later iterations pick up fine details. The hypothesis class determines what patterns can be learned at each iteration, and how those patterns generalize.

## Connecting Back to Intuition

Let's not lose sight of the simple picture we started with. At each iteration:

1. We look at where our current model is making errors
2. We compute how the loss wants us to change our predictions (the gradient)
3. We fit a new weak learner to approximate those desired changes
4. We add that weak learner to our ensemble

The functional gradient framework gives us the *right* notion of "desired changes" for any loss function. It's not always the residual, but it's always the direction that decreases loss fastest.

For squared error, this collapses to the simple residual-fitting view we started with. But now we understand *why* that works, and we can generalize it to arbitrary losses.

## Looking Forward

We now have a complete picture of gradient boosting:
- The intuitive view (fitting errors iteratively)
- The geometric view (gradient descent in function space)
- The algorithmic view (pseudo-residuals and base learners)
- The theoretical view (functional optimization with approximation)

These perspectives are all the same algorithm, viewed through different lenses. Each lens highlights different aspects:
- Intuition emphasizes what the algorithm is doing
- Geometry shows why it makes sense
- Algorithm tells us how to implement it
- Theory explains when it works and how to tune it

This kind of multilayered understanding is what lets us use gradient boosting effectively in practice. We can reason about regularization (learning rate, tree depth, number of iterations) through the functional gradient lens. We can design custom losses for specific problems. We can debug models by examining pseudo-residuals.

The functional gradient view isn't just mathematical machinery—it's a lens that makes the algorithm transparent and manipulable. And that's exactly what we want from our theoretical frameworks: not just correctness, but clarity.
