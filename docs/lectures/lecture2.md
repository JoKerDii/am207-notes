## Maximum Likelihood Estimation

Outline:

1. A motivating example
2. A statistical model for a coin flip
3. Maximum likelihood estimation
4. Convex optimization: constrained and unconstrained
5. Propoerties of MLE
6. Uncertainty quantification

### 1. A Statistical Model for a Coin Toss

 A common way to test a coin for bias is to toss this coin $N$ number of times and count the number of heads, $H$, The fraction $H/N$ is one way to quantify the probability of the coin to land heads up on any given toss.

**Question 1**: is this estimate the bias valid? 

**Question 2**: is this the best way to estimate the bias?

#### Likelihood for a Coin Toss

We can formally model the outcome of the single toss of a coin by a Bernoulli distribution
$$
Y \sim Ber(\theta)
$$
where $\theta$ is the probability that the outcome $Y$ will be heads

**Question**: is it a valid statistical model of a coin toss in real life? What assumptions does this statistical model expose?

* The model assumes that the outcome of a coin toss depends only on the probability of heads, not on any environmental factors that intervene with the outcome (e.g. how hard I throw the coin, am I try to throw the coin in a particular way that will lands it heads up). 

After $N$ number of independent tosses of an identical coin, the probability (or likelihood) of observing $Y=H$ number of heads is 
$$
{N \choose H} \theta^H (1-\theta)^{N-H}
$$
That is, $Y$ is a random variable with a binomial distribution $Y \sim Bin(N,\theta)$.

We see that the fraction $H/N$ from our empirical experiment is an estimate of the parameter $\theta$ of the binomial distribution $Bin(N,\theta)$. Now that we have a statistical model, we can give formal justification for why our estimate is desirable (or undesirable).

### 2. Maximum Likelihood Estimate

#### Parameter Estimation: Maximum Likelihood

Let $Y_1, ..., Y_N$ be $Y_n \overset{iid}{\sim} p(Y|\theta)$, where $p(Y|\theta)$ is a distribution parameterized by $\theta$ ($\theta$ can be a scalar, a vector, a matrix, or a n-tuple of such quantities). The **joint likelihood** of $N$ observations, $y_1, ..., y_N$, is
$$
\mathcal{L}(\theta) = \prod^N_{n=1}p(y_n|\theta)
$$
Note that we use upper-case letters $Y_n$ to represent random variables and lower-case $y_n$ to represent specific observed values of those variables.

The joint likelihood quantifies how likely (or probable, if $Y$ is discrete) we are to observed the data assuming the model $\theta$. when we consider the joint likelihood as a function of $\theta$ (That is, treat the observed data as fixed), the $\mathcal{L}(\theta)$ is called the **likelihood function**.

The maximum likelihood estimate of $\theta$ is defined as
$$
\theta_{MLE} = \arg\max_{\theta} \mathcal{L}(\theta) = \arg\max_\theta \prod_{n=1}^Np(y_n|\theta)
$$

#### Maximizing Likelihood is Equivalent to Maximizing Log-Likelihood

Frequently, the likelihood function is complex and so it's often preferable to work with the log of the likelihood function. Luckily, **maximizing the likelihood is equivalent to maximizing the log likelihood** due to the fact that log function is a monotonic function.

**Theorem**: for an $f: \mathbb{R}^D \rightarrow \mathbb{R}$, we have that $x^* = \arg\max_\theta f(x)$ if and only if $x^* = \arg\max_{\theta} \log(f(x))$

### 3. Convex Optimization: Constrained and Unconstrained

#### Stationary Points

The instantaneous rate of change, at $x=x_0$, of a differentiable function $f: \mathbb{R} \rightarrow \mathbb{R}$ is given by it's first derivative at $x = x^*, {df \over dx} \Big|_{x^*}$

For a multivariate differentiable function $f:\mathbb{R}^D \rightarrow \mathbb{R}$, the gradient of $f$ at a point $x^*$ is a vector consisting of the partial derivatives of $f$ evaluated at $x^*$:
$$
\nabla_xf|_{x^*} = \Big[{\partial \over \partial x^{(1)}}\Big|_{x^*}, ..., {\partial \over \partial x^{(D)}} \Big|_{x^*}\Big]
$$
Each ${\partial \over \partial x^{(1)}}$ compute the instantaneous change of $f$ at $x=x^*$ with respect to $x^{(1)}$.

The gradient is **orthogonal** to the level curve of $f$ at $x^*$ and hence, when it is not zero, **points in the direction of the greatest instantaneous increase** in $f$.

#### Characterization of Local Optima

A local optima must be a stationary point, but a stationary point need not be a local optima (can be saddle point). 

To check that a stationary point is a local max (or local min), we must check that the function is concave (or convex) at the point.

For a twice differentiable function $f: \mathbb{R} \rightarrow \mathbb{R}, f$ is concave at $x=x^*$ if the second derivative of $f$ is negative; $f$ is convex at $x=x^*$ if the second derivative of $f$ is positive. 

For a multivariate twice differentiable function $f: \mathbb{R}^D \rightarrow \mathbb{R}, f$ is concave at $x=x^*$ if the Hessian matrix is semi-negative definite; $f$ is convex at $x=x^*$ if the Hessian is semi-positive definite.

#### Characterization of Global Optima

For an arbitrary function, we cannot generally determine if a local optimal is a global one. In certain very restricted cases, we can deduce if a local optimal is global.

**Theorem**: If a continuous function $f$ is convex (or resp. concave) on its domain then every local min (or resp. max) is a global min (or resp. max).

#### Unconstrained Optimization

Analytically solving an optimization problem without constraints on the domain of the function,
$$
x_{max} = \arg\max_x f(x)
$$
involves:

1. Find the expression for $\nabla_xf(x)$
2. Find the stationary points for $\nabla_x f(x)$. That is, solve the equation $\nabla_x f(x) = 0$ for $x$.
3. Determine local optima. That is, check the concavity of $f$ at the stationary points.
4. Determine global optima. That is, check if local optima can be characterized as global optima (e.g. check that $f$ is convex everywhere on its domain)

#### Example: (Univariate) Gaussian Distribution

##### Likelihood and Log-Likelihood

Suppose that $Y_m \overset{iid}{\sim} \mathcal{N}(\mu,\sigma^2)$, where $\sigma > 0$. Let $\theta$ denotes the set of parameters $(\mu, \sigma)$. The likelihood for $N$ observations $y_1, ..., y_N$ is
$$
\mathcal{N}(\theta) = \prod^N_{n=1}{1\over \sqrt{2\pi \sigma^2}} \exp \Big\{-{(y_n-\mu)^2\over 2\sigma^2}\Big\} = {1\over (2 \pi \sigma^2)^{N/2}} \exp \Big\{-{\sum^N_{n=1}(y_n-\mu)^2\over 2\sigma^2}\Big\}
$$
The log likelihood is 
$$
\ell(\theta) = -{N\over 2} \log2\pi - N \log \sigma - {(y_n-\mu)^2\over 2\sigma^2}
$$

##### Gradient of Log-Likelihood

The gradient of $\ell$ with respect to $\theta$ is the vector $\nabla_\theta \ell(\theta) = \Big[{\partial \ell\over \partial \mu},{\partial \ell\over \partial \sigma}\Big]$, where the partial derivatives are given by:
$$
\begin{aligned}
{\partial \ell \over \partial \mu} &= {1\over \sigma^2}\sum^N_{n=1}(y_n-\mu)\\
{\partial \ell \over \partial \sigma} &= -{N\over \sigma} + \sigma^{-3}\sum^N_{n=1}(y_n-\mu)^2
\end{aligned}
$$

##### Stationary Points of the Gradients

The stationary points of the gradients are solutions to the following system of equations:
$$
\begin{cases}
{\partial \ell \over \partial \mu} &= {1\over \sigma^2}\sum^N_{n=1}(y_n-\mu)=0\\
{\partial \ell \over \partial \sigma} &= -{N\over \sigma} + \sigma^{-3}\sum^N_{n=1}(y_n-\mu)^2=0
\end{cases}
$$
Solving this system, we get a unique solution at:
$$
\begin{cases}
\mu = {1\over N} \sum^N_{n=1} y_n\\
\sigma = \sqrt{{1\over N}\sum^N_{n=1}(y_n - \mu)^2}
\end{cases}
$$

##### Characterize Local and Global Optima

The log-likelihood in this case is concave because the Hessian is negative semi-definite for $\mu$ and $\sigma>0$. Thus, the log-likelihood is globally maximized at:
$$
\begin{cases}
\mu_{MLE} = {1\over N}\sum^N_{n=1}y_n\\
\sigma_{MLE} = \sqrt{{1\over N}\sum^N_{n=1} (y_n-\mu)^2}
\end{cases}
$$
If you write out the matrix of second order partial derivatives of the log likelihood, check that all the upper-left sub-matrices have negative determinants.

**Note**: if the objective is not concave, then there is no guarantee that the stationary points will be global maxima.

#### Constrained Optimization

Many times, we are constrained by the application o only consider certain types of values of the input $x$. Suppose that the **constraints** on $x$ are given by the equation $g(x)=0$. The set of values of $x$ that satisfy the equation are called **feasible**. 

We are only interested in optimiqzing the function in our set of feasible points.

**Theorem**: For a differentiable function $f: \mathbb{R}^D \rightarrow \mathbb{R}$, the local optima of $f$ constrained by $g(x)=0$ occur at points where the following hold for some $\lambda \in \mathbb{R}$.
$$
g(x) = 0, \space \nabla_x f(x) = \lambda \nabla_x g(x)
$$
The theorem says that the local optima of $f$ satisfying $g(x)=0$ are where the gradients of $f$ and $g$ are **parallel**.

##### Constrained Optimization via Lagrange Multipliers

Solving an optimization problem within the **feasible region** of the function, i.e.
$$
\max_x f(x), \space g(x)=0
$$
involves

1. Finding the stationary points of the augmented objective $J(x) = f(x)-\lambda g(x)$, with respect to $x$ and $\lambda$.
2. Determine global optima. Determine if any of the stationary points maximizes $f$.

The augmented objective $J$ is called the **Lagrangian** of the constrained optimization problem and $\lambda$ is called the **Lagrange multiplier**.

for a general arbitrary function $f$, it is almost impossible to verify the stationary point of the Lagrangian are indeed global optima. However, when the function is convex, and the constraints are affined (i.e. defined by linear functions), then the staionary points of Lagrangian will be global optima.

**Note**: Constrained optimization with inequality constraints can similarly be formulated in erms of finding stationary points of an augmented objective like the Lagrangian; this follows from the **Karush-Kuhn-Tucker theorem**.

#### Example: Binomial Distribution

##### Likelihood and Log-Likelihood

Suppose that $Y\sim Bin(N, \theta)$. To make the connection with constrained optimization (and to motivate the multinomial case), let's write $\theta$ as a vector $[\theta_0, \theta_1]$, where $\theta_1$ is the probability of a head and $\theta_0+\theta_1 = 1$.

The likelihood for a single observation is 
$$
\mathcal{L}(\theta) = {N!\over y!(N-1)!} \theta_1^y \theta_0^{N-y}
$$
The log likelihood is 
$$
\ell(\theta) = \log(N!)-\log(y!) -\log(N-y)! + y \log \theta_1 + (N-y)\log \theta_0
$$
We are interested in solving the following constrained optimization problem:
$$
\max \ell(\theta),\space \theta_0+\theta_1= 1
$$
Whose Lagrangian is given by
$$
{J}(\theta, \lambda) = \ell(\theta) - \lambda(\theta_0 + \theta_1 -1 )
$$

##### Gradient of Log-Likelihood

The gradient of the Lagrangian $J$ with respect to $(\theta,\lambda)$ is the vector $\nabla_{\theta, \lambda} J = \Big[{\partial \ell \over \partial \theta_0} , {\partial \ell \over \partial \theta_1}, {\partial \ell \over \partial \lambda}\Big]$, where the partial derivatives are given by
$$
\begin{aligned}
{\partial \ell \over \partial \theta_0} &= {(N-y)\over \theta_0}- \lambda\\
{\partial \ell \over \partial \theta_1} &= {y \over \theta_1} - \lambda\\
{\partial \ell \over \partial \lambda} &= \theta_0 + \theta_1 -1
\end{aligned}
$$

##### Stationary Points of the Lagrangian

The stationary points of the Lagrangian are solutions to the following system of equations:
$$
\begin{cases}
{\partial \ell \over \partial \theta_0} &= {(N-y)\over \theta_0}- \lambda=0\\
{\partial \ell \over \partial \theta_1} &= {y \over \theta_1} - \lambda=0\\
{\partial \ell \over \partial \lambda} &= \theta_0 + \theta_1 -1=0
\end{cases}
$$
Solving this system we get a unique solution at
$$
\begin{cases}
\theta_0 = {N-y \over \lambda}\\
\theta_1 = {y\over \lambda}\\
\theta_0 + \theta_1 = 1
\end{cases}
$$
In other words, $\lambda = N$ and $\theta_1 = {y\over N}$ and $\theta_0 = {N-y \over N}$.

##### Characterize Global Optima

The stationary point of the Lagrangian is not necessarily a global optima of the constraint optimization problem.

Since the log-likelihood $\ell(\theta)$ is **concave** and the constraint $\theta_0+\theta_1=1$ is **affine** (linear up to a constant), we know then that $\ell(\theta)$ is **maximized** on the line at the stationary point. Hence
$$
\begin{cases}
\theta_0^{MLE} = {N-y \over N}\\
\theta_1^{MLE} = {y \over N}
\end{cases}
$$
**Note**: The stationary point of the Lagrangian is not necessarily a global optima of the constraint optimization problem. If the objective function we are maximizing is not concave and the equality constraint is not affine, then we have no guarantee that the stationary points of the Lagrangian either locally or globally optimizes our objective.

#### What Is a Good Estimator?

If we assume of a binomial model, $Bin(N,\theta)$, for the number of heads in $N$ trials, then the fraction ${H/N}$ is the MLE of $\theta$. (Which is the bias of a coin or the probability a head for that coin.)

**Question 1**: is the MLE a good estimator of $\theta$? 

* MLE of $\theta$ is based on a finite set of observations, and we could combined the finite set of  observations to produce estimate that looks very different than the MLE. 

**Question 2**: is this the ''best" way to estimate the $\theta$? For example, is the quantity ${H+1\over N+2}$ an equally valid or better estimate of $\theta$?

The answers depend on what you mean by a good estimate of a statistical parameter i.e. our list of desiderata for our estimator.

### 4. Properties of MLE

#### Desiderata of Estimator

Let $\hat{\theta}$ be an estimator of the parameter $\theta$ of a statistical mode. We ideally want

1. **Consistency**: when the sample size $N$ increases, in the limit, $\hat{\theta}$ approaches the true value of $\theta$.

   More formally, let ${p_\theta: \theta \in \Theta}$ be a family of candidate distributions and $X^\theta$ be an infinite sample from $p_\theta$, Define $\hat{g}_N(X^\theta)$ to be an estimator for some parameter $g(\theta)$ that is based on the first $N$ samples. Then we say that the sequence of estimators $\big[\hat{g}_N(X^\theta)\big]$ is (weakly) consistent if $\lim_{N\rightarrow \infty} \hat{g}_N(X^{\theta}) = g(\theta)$ in probability for all $\theta \in \Theta$.

2. **Unbiasedness**: on average, over all possible sets of observations from the distribution, the estimator nails the true value of $\theta$.

   * The estimator is a function of the data that we happen to have observed. If we compute the same estimator on a slightly different set of data drawn from the same data generating process, we are going to get different estimations. What we want is when we average over all possible sets of observations drawn from the data generating process or distribution, our estimator is going to actually nail the true value of theta.  

   More formally, we want $\mathbb{E}_{X^\theta} \hat{\theta} (X^\theta)=\theta$.

3. **Minimum Variance**: Note that since our estimator $\hat{\theta}$ depends on the random sample $X^{\theta}$, it follows that $\hat{\theta}$ also a random variable. The distribution of $\hat{\theta}$ is called the **sampling distribution**. Given that our estimator is unbiased, we want it to have minimum variance with respect to the sampling distribution.
   * This means the least sensitive to small minor changes in the observed data. [Why this is important?]

#### Properties of MLE

1. **Consistency**: the MLE of iid observations is consistent - as the number of observations increase towards infinity, the MLE of theta is going to approach theta itself. The asymptotic sampling distribution of the MLE is a Gaussian.
2. **Unbiasedness**: the MLE can be biased if we don't have infinite number of observations.
3. **Minimum Variance**: the MLE is not the estimator with the lowest variance, with the finite number of observations.

Asymptotically, however, when we have infinite number of observations, the MLE is unbiased and has the lowest variance (for unbiased estimators). This means when the number of observations increases, the properties of MLE become better and better. 

**Assumptions**: with finite observations, MLE may not have any of those properties. We need to make some assumption for properties of MLE to hold, including

* The model is **well-specified** - the observed data is drawn from the same model class as the model being fitted.
* The estimation problem is **well-posed** - there are not two different set of parameters that generate the same data. [What if this is violated?]

### 5. Uncertainty Quantification

Since MLE depends on the random sample of observational data that we drew, this estimator is a random variable. What is the uncertainty we have about this random variable? 

#### Confidence Intervals

Since MLE depends on the sample, it is important to quantify how certain we are about the maximum likelihood estimator of model parameters. 

**Confidence Intervals** of estimates $\theta_{MLE}$ are ways of summarizing the sampling distribution by describing it's coverage. Specifically, a 95% confidence interval for $\theta$ is a **random interval** $(L_{\theta_{MLE}}, U_{\theta_{MLE}} )$, where $L$ and $U$ are bounds constructed from the estimate $\theta_{MLE}$, that contains the fixed true parameter $\theta$ with $95\%$ probability.

Let $\delta = \theta_{MLE} -\theta$ be the distribution of the error of the estimator $\theta_{MLE}$, then the following is a confidence interval for $\theta$:
$$
\Big[\hat{\theta} - \delta_{0.25}, \hat{\theta} + \delta_{-0.975}\Big]
$$
Where $\delta_{0.25}, \delta_{0.975}$ aer the $2.5\%$ and $97.5\%$ thresholds of $\delta$ respectively.

We can take advantage of the asymptotic normality of the MLE and approximate the distribution of $\delta$ as a Gaussian distribution.

**Note**: we are not going to construct confidence interval in this way (by $\delta = \theta_{MLE} - \theta$, the difference between MLE estimator and ground truth theta), because in practice we don't know anything about the sampling distribution.

#### Interpretation of Confidence Intervals

It is easy to misinterpret confidence intervals

**A simplified rule**: when in doubt, treat the confidence interval just as **an indication of the precision of the measurement**.

If different people estimated some quantified in a study with a different confidence intervals: $[17-6,17+6]$ and $[23-5,23+5]$, then there is little reason to think that the two studies are inconsistent.

If two confidence intervals are $[17-2,17+2]$ and $[23-1,23+1]$, then there is evidence that these studies differ.

#### Bootstrap Confidence Interval

In practice we don't know anything about the sampling distribution, and we may not know how to approximate the sampling distribution of $\theta_{MLE}$, but we can approximate the sampling distribution by bootstrapping, i.e. we simulate samples $X^{\theta}$ with size $N$ from $p_\theta$ by sampling observations with size $N$ from the observed data (also with size $N$).

We denote MLE obtained on a bootstrap sample by $\theta_{MLE}^{bootstrap}$. When $N$ is sufficiently large, $\theta_{MLE}^{bootstrap}$ approximates the distribution of $\theta_{MLE}$.

Thus, we can approximate the $95\%$ confidence interval of $\theta$ using $\theta_{MLE}^{bootstrap}$.

