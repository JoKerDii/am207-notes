## Logistic Regression and Gradient Descent

Content:

1. Logistic Regression
2. Gradient Descent
3. Convex Optimization

### 1. Logistic Regression

#### Coin-Toss Revisited: Modeling a Bernoulli Variable with Covariates

We assume that the outcomes $Y^{(n)}$ were independently and identically distributed as Bernoulli's $Y^{(n)} \sim Ber(\theta)$. We will examine the **identical** part of the modeling assumptions.

Realistically, the probability of $Y^{(n)}=1$ depends on variables like force, angle, spin, etc.

Let $X^{(n)} \in \mathbb{R}^D$ be $D$ number of such measurements of the $n$-th toss. We model the probability of $Y^{(n)}=1$ as a function of these **covariates** $X^{(n)}$:
$$
Y^{(n)} \sim Ber (sigm(f(X^{(n)};w)))
$$
where $w$ are the parameters of $f$ and $sigm$ is the sigmoid function $sigm(z) = {1\over 1+e^{-z}}$.

**Note**: we need the sigmoid function to transform an arbitrary real number $f(X^{(n)};w)$ into a probability (i.e. a number in $[0,1]$).

#### The Logistic Regression Model

Given a set of $N$ observations $(x^{(1)},y^{(1)}), ..., (x^{(N)},y^{(N)})$. We assume the following model for the data.
$$
Y^{(n)} \sim Ber(sigm(f(X^{(n)};w)))
$$
This is called the **logistic regression** model.

Fitting this model on the data means **inferring** the parameters $w$ that best aligns with the observations.

Once we have inferred the parameters $w$, given a new set of covariates $x^{new}$, we can **predict** the probability of $Y^{new}=1$ by computing
$$
sigm(f(X^{(n)};w))
$$
For now, we will assume that $f$ is a linear function:
$$
f(X^{n};w) =w^TX^{(n)}
$$

#### Maximizing the Logistic Regression Log-likelihood

Given a set of $N$ observations $(x^{(1)},x^{(1)}), ... , (x^{(N)},x^{(N)})$. We want to find $w_{MLE}$ that maximizes the log (input) likelihood:
$$
\begin{aligned}
w_{MLE} &= \arg \max \space \ell(w) = \arg \min_w - \ell(w) = \arg \min_w - \log \prod^N_{n=1} p(y^{(n)}|x^{(n)})\\
&= \arg \min_w \prod^N_{n=1} - \log \Big(sigm(w^T x^{(n)}) ^{y^{(n)}} (1-sigm(w^T x^{(n)}))^{1-y^{(n)}}\Big)\\
&= \arg \min_w \prod^N_{n=1} - y^{(n)}\log sigm(w^T x^{(n)}) - (1-{y^{(n)}}) \log (1-sigm(w^T x^{(n)}))\\
\end{aligned}
$$
Optimizing the likelihood requires us to find the stationary points of the gradient of $\ell(w)$:
$$
\nabla_w \ell(w) = - \sum_{n=1}^{N} \Big(y^{(n)} - {1\over 1 + \exp(-w^Tx^{(n)})}\Big) x^{(n)}=0
$$

### 2. Gradient Descent

The gradient is orthogonal to the level curve of $f$ at $x^*$ and hence, when it is not zero, points in the direction of the greatest instantaneous increase in $f$.

#### Gradient Descent: the Algorithm

1. Start at random place: $w^{(0)} \leftarrow \text{random}$

2. until (stopping condition satisfied):

   a. compute gradient at $w^{(t)}$: gradient $(w^{(t)})=\nabla_w$ loss function $(w^{(i)})$

   b. take a step in the negative gradient direction: $w^{(t+1)} \leftarrow w^{(t)} - \eta * \text{gradient}(w^{(t)})$.

**Note**: stopping conditions - 

* Your parameters have ceased to update very much.
* The magnitude of gradient becomes close to zero.

### 3. Convex Optimization

#### Convex Sets

A **convex set** $S \subset \mathbb{R}^D$ is a set that contains the line segment between any two points in $S$. Formally if $x,y \in S$ then $S$ contains all convex combinations of $x$ and $y$.
$$
tx+(1-t) \in S, \space t \in [0,1]
$$

#### Convex Functions

A function $f$ is a convex function if domain of $f$ is a convex set, and the line segment between the points $(x,f(x))$ and $(y,f(y))$ lie above the graph of $f$. Formally, for any $x,y \in \text{dom} (f)$, we have
$$
f(tx+(1-t)y) \leq tf(x)+(1-t)f(y)
$$

#### Convex Function: First Order Condition

To check a function $f$ is convex? If $f$ is differentiable, then $f$ is convex if the graph of $f$ lies above every tangent plane.

**Theorem**: If $f$ is differentiable then $f$ is convex if and only if for every $x \in \text{dom} (f)$, we have
$$
f(y) \geq f(x) + \nabla f(x)^T(y-x), \forall y \in \text{dom}(f)
$$
if a function is differentiable, then we can use its first derivative in order to check that this function is convex. This is called a first order condition.

#### Convex Function: Second Order Condition

If $f$ is twice-differentiable then $f$ is convex if the "second derivative is positive".

**Theorem**: If $f$ is twice-differentiable then $f$ is convex if and only if the Hessian $\nabla^2 f(x)$ is positive semi-definite for every $x \in \text{dom} (f)$.

#### Properties of Convex Functions

Complex convex functions can be built from simple convex functions:

1. If $w_1, w_2 \geq 0$ and $f_1, f_2$ are convex, then $h=w_1f_1 + w_2 f_2$ is convex
2. If $f$ and $g$ are convex, and $g$ is univariate and non-decreasing then $h=g \cdot f$ is convex
3. Log-sum-exp functions are convex: $f(x)=\log \sum_{k=1}^{K}\exp(x)$

**Note**: there are many other convexity preserving operations on functions

#### Convex Optimization

A **convex optimization problem** is an optimization of the following form:
$$
\begin{aligned}
\min f(x) &\space\space\space \text{(convex objective function)}\\
\text{subject to } h_i(x) \leq 0, i=1,...,i &\space\space\space \text{(convex inequality constraints)}\\
a^T_j x - b_j=0, j=1, ..., J &\space\space\space \text{(convex equality constraints)}
\end{aligned}
$$
The set of points that satisify the constraints is called the **feasible set**.

You can prove that the a convex optimization problem optimizes a convex objective function over a convex feasible set. But why should we care about convex optimization problems?

**Theorem**: Let $f$ be a convex function defined over a convex feasible set $\Omega$. Then if $f$ has a local minimum at $x \in \Omega$ -- $f(y) \geq f(x)$ for $y$ in a small neighbourhood of $x$ -- then $f$ has a global minimum at $x$.

**Corollary**: Let $f$ be a differentiable convex function:

1. if $f$ is unconstrained, then $f$ has a **local minimum** and hence **global minimum** at $x$ if $\nabla f(x)=0$.
2. if $f$ is constrained by equalities, then $f$ has a global minimum at $x$ if $\nabla J(x,\lambda)=0$, where $J(x,\lambda)$ is the Lagrangian of the constrained optimization problem.

#### Convexity of the Logistic Regression Negative Log-Likelihoods

Connecting the theory of convex optimization to MLE inference for logistic regression. Recall that the negative log-likelihood of the logistic regression model is
$$
\begin{aligned}
-\ell(w) &= - \sum_{n=1}^{N} y^{(n)} \log sigm(w^T x^{(n)}) + (1-y^{(n)}) \log(1- sigm(w^T x^{(n)}))\\
&=\sum^N_{n=1} y^{(n)} \log (\exp(0) + \exp(w^T x^{(n)})) + (1-y^{(n)})(-w^Tx^{(n)})
\end{aligned}
$$
**Proposition**: The negative log-likelihood of logistic regression $-\ell(w)$ is convex.

**What does this mean for gradient descent**? If gradient descent finds that $w^*$ is a stationary point of $-\nabla_w \ell(x)$ then $-\ell(w)$ has a global minimum at $w^*$. Hence, $\ell(w)$ is a maximized at $w^*$.

If we run gradient descent correctly until convergence, we can be guaranteed that this algorithm has maximized the log likelihood of the observed data.

#### Does it Scale?

Gradient is a simple algorithm that can be applied to any optimization problem for which you can compute the gradient of the objective function.

This doesn't mean that maximum likelihood inference for statistical models is now an easy task (i.e. just use gradient descent).

For every likelihood optimization problem, evaluating the gradient at a set of parameters $w$ requires evaluating the likelihood of the entire dataset using $w$:
$$
\nabla_w \ell(w) = - \sum_{n=1}^{N} \Big(y^{(n)} - {1\over 1+ \exp(-w^Tx^{(n)})}\Big) x^{(n)} = 0
$$
Imagine if the size of your dataset $N$ is in the millions. Naively evaluating the gradient just once may take up to seconds or minutes, thus running gradient descent until convergence may be unachievable in practice.

**Idea:** Without using the entire data set to evaluate the gradient during each step of gradient descent, we can approximate the gradient at $w$ well enough with just a sebset of the data.
