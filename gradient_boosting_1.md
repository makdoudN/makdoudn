# Gradient Boosting I: Building Models Incrementally

## The Residual Learning Perspective

When we first encounter supervised learning, we're often presented with a straightforward narrative: we have data, we choose a model family, and we find the best model within that family by minimizing some loss function. This is clean, elegant, and completely correct. But it's also somewhat limited.

What if, instead of trying to find the single best model, we could *iteratively improve* our predictions? What if we could start with a simple guess and systematically correct our mistakes? This is the core insight behind boosting, and understanding it properly requires us to think carefully about what it means to "correct mistakes" in a principled way.

### Starting Simple: The Regression Problem

Let's begin with the most transparent setting: regression with squared error loss. We have training data $(x_1, y_1), \ldots, (x_n, y_n)$ where $x_i \in \mathbb{R}^p$ and $y_i \in \mathbb{R}$. Our goal is to learn a function $f: \mathbb{R}^p \to \mathbb{R}$ that predicts $y$ from $x$.

The squared error loss for a prediction $f(x_i)$ is:
$$
L(y_i, f(x_i)) = \frac{1}{2}(y_i - f(x_i))^2
$$

Our empirical risk is:
$$
\mathcal{R}(f) = \frac{1}{n}\sum_{i=1}^n \frac{1}{2}(y_i - f(x_i))^2
$$

Now, suppose we have a current model $f_m(x)$ that makes predictions. These predictions won't be perfect—there will be residuals:
$$
r_i^{(m)} = y_i - f_m(x_i)
$$

Here's the key question: *if we're allowed to add another function to our model, what function should we add?*

### The Residual Fitting Perspective

The most intuitive answer is: we should add a function $h_m(x)$ that predicts the residuals. If we can predict what our current model is missing, we can correct for it.

Consider the new model:
$$
f_{m+1}(x) = f_m(x) + h_m(x)
$$

What loss should we minimize to find $h_m$? Well, ideally we want:
$$
f_{m+1}(x_i) = f_m(x_i) + h_m(x_i) \approx y_i
$$

This means:
$$
h_m(x_i) \approx y_i - f_m(x_i) = r_i^{(m)}
$$

So we should fit $h_m$ to the residuals using the same loss:
$$
h_m = \arg\min_{h} \sum_{i=1}^n \frac{1}{2}(r_i^{(m)} - h(x_i))^2
$$

This is beautiful: it's simple, intuitive, and correct. We fit a model to the data, look at what we got wrong, fit another model to those mistakes, and add it to our ensemble.

### The Iterative Algorithm

This gives us our first boosting algorithm:

1. Initialize $f_0(x) = 0$ (or to some constant)
2. For $m = 0, 1, 2, \ldots, M-1$:
   - Compute residuals: $r_i^{(m)} = y_i - f_m(x_i)$ for $i = 1, \ldots, n$
   - Fit a model to residuals: $h_m = \arg\min_{h \in \mathcal{H}} \sum_{i=1}^n \frac{1}{2}(r_i^{(m)} - h(x_i))^2$
   - Update: $f_{m+1}(x) = f_m(x) + \nu h_m(x)$ where $\nu \in (0,1]$ is a learning rate

The learning rate $\nu$ is crucial—it controls how much we trust each new weak learner. Smaller values lead to more robust models but require more iterations.

### Why Does This Work?

To understand why this works, let's look at what happens to the loss after we add $h_m$. Our risk is:
$$
\mathcal{R}(f_{m+1}) = \frac{1}{n}\sum_{i=1}^n \frac{1}{2}(y_i - f_m(x_i) - \nu h_m(x_i))^2
$$

Expanding:
$$
= \frac{1}{n}\sum_{i=1}^n \frac{1}{2}(r_i^{(m)} - \nu h_m(x_i))^2
$$

If $h_m$ does a good job predicting the residuals, this will be smaller than:
$$
\mathcal{R}(f_m) = \frac{1}{n}\sum_{i=1}^n \frac{1}{2}(r_i^{(m)})^2
$$

Each iteration reduces the training error, at least locally. We're making progress.

### Generalizing Beyond Squared Error

This residual-fitting perspective is clean and intuitive for squared error. But what about other losses? What if we're doing classification with logistic loss? What about quantile regression with absolute error?

Here's where things get interesting. For general losses, the residual isn't always the right thing to fit. Consider the absolute error:
$$
L(y, f(x)) = |y - f(x)|
$$

The residual $r_i = y_i - f_m(x_i)$ doesn't have the same special status—there's no obvious reason why fitting it with squared error would minimize absolute error loss.

This is the tension that will motivate our next step. We have an algorithm that works perfectly for squared error, but the underlying principle—fit the residuals—seems tied to that specific loss function.

### The Hint of Something Deeper

Before we move forward, let's notice something subtle about the residual. For squared error, the residual is:
$$
r_i^{(m)} = y_i - f_m(x_i) = -\frac{\partial}{\partial f(x_i)} \frac{1}{2}(y_i - f(x_i))^2 \bigg|_{f=f_m}
$$

The residual is the *negative gradient* of the loss with respect to the prediction.

This is not a coincidence. This is the key that will unlock everything.

When we fit the residual, we're actually doing something more general: we're fitting the negative gradient of the loss. For squared error, these happen to coincide. But the gradient perspective will let us handle arbitrary differentiable losses.

This observation—that boosting with squared error is secretly following gradients—is the bridge to a much more general framework. But to cross that bridge properly, we need to think carefully about what it means to take gradients in function space. That's where we'll turn next.

### What We've Learned

Let's consolidate what we know:

1. **Residual fitting works**: For squared error, we can iteratively improve by fitting residuals
2. **The algorithm is simple**: Compute errors, fit them, add the correction
3. **Learning rates matter**: Small steps are more robust
4. **The residual has a deeper meaning**: It's the negative gradient of the loss

This residual-fitting view gives us a concrete algorithm and builds our intuition. But it's also somewhat limited—it's not clear how to generalize beyond squared error in a principled way.

The key insight we've uncovered is that the residual is actually a gradient. This hints at a deeper structure: maybe we're not just fitting residuals, but rather performing *gradient descent in function space*.

Making this precise requires us to think about functions as points in an infinite-dimensional space, about what gradients mean in that space, and about how we can approximate those gradients with our finite base learners. That's the conceptual journey we'll take in the next part.
