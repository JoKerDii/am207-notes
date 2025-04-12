## Monte Carlo Integration

Content:

1. Basics of Monte Carlo Simulation
2. Variance Reduction: Control Variates
3. Variance Reduction: Stratified Sampling
4. Variance Reduction: Importance Sampling
5. Application: Monte Carlo Estimation of Arbitrary Integrals

### 1. Basics of Monte Carlo Simulation

#### Naive Monte Carlo Estimation of Integrals

Let ${I}$ denote the integral
$$
\mathbb{E}_{\theta|Y}[f(\theta)]= \int_{\Theta} f(\theta)p(\theta | Y) d\theta
$$
and let $\hat{I}$ denote the approximation
$$
{1\over S} \sum^S_{s=1}f(\theta_s), \theta_s \sim p(\theta|Y)
$$
We call $\hat{I}$ the **Monte Carlo estimate** of $I$.

In general, **Monte Carlo Integration** is the process of estimating a deterministic quantity (an integral) using a procedure involving stochasticity (sampling).

#### The Consistency and Unbiasedness of Monte Carlo Estimators

Recall that the **Strong Law of Large Numbers** says that the sample mean of $S$ iid random variables converges to the theoretical mean, with probability 1, as $S \rightarrow \infty$. This means that
$$
\lim_{S \rightarrow \infty} {1 \over S} \sum^S_{s=1} f(\theta_s), \theta_s \sim p(\theta|Y) = \mathbb{E}_{\theta | Y}[f(\theta)]
$$
with probability $1$. Hence, the Monte Carlo Estimator $\hat{I}$ is **consistent**. 

The expected value of $\hat{I}$ is
$$
\mathbb{E}_{\theta_S}[\hat{I}] = \mathbb{E}_{\theta_S} \Big[{1\over S} \sum^S_{s=1}f(\theta_s)\Big]= {1\over S} \sum^S_{s=1}\mathbb{E}_{\theta_s} [f(\theta_s)] = {1\over S} \sum^S_{s=1} I=I
$$
where $\theta_S \sim p(\theta|Y)$. Hence, the Monte Carlo Estimator $\hat{I}$ is **unbiased**.

#### The Variance and Error of Monte Carlo Estimators

The variance of $\hat{I}$ is given by
$$
Var[\hat{I}] = Var\Big[{1\over S} \sum^S_{s=1} f(\theta_s)\Big] = {1\over S^2} \sum^S_{s=1}Var[f(\theta_s)] = {Var[f(\theta)]\over S}
$$
where $\theta_s$, $\theta \sim p(\theta|Y)$. **Plainly put, the variance of the estimator is reduced when number of samples $S$ is large and the variance $\sigma_f^2 =  Var[f(\theta)]$ of $f(\theta)$ is low.**

The **Central Limit Theorem** says that the difference between sample mean and the theoretical mean of iid random variables will approach a normal distribution as $N \rightarrow \infty$. For us, this means that the error of $\hat{I}$ has a roughly normal distribution:
$$
p(\hat{I} - I) \rightarrow \mathcal{N}\Big(0, {\sigma_f^2\over S}\Big), S\rightarrow \infty
$$
Again, this says that, **to reduce the error of $\hat{I}$ we can increase the number of samples $S$ or decrease the variance $\sigma^2_f$ of $f(\theta)$.**

**Note**: we care about the variance because it relates to the errors (e.g. differences between sample mean and theoretical mean) an estimate makes. Since our Monte Carlo Estimator is both consistent and unbiased, the only way to improve our estimator is by lowering the variance. 

### 2. Variance Reduction: Control Variates

#### A Baseline for Variance of Monte Carlo Estimates

The variance of the naive Monte Carlo estimate $\hat{I}$ is:
$$
\begin{aligned}
Var[\hat{I}] &= {1\over S} Var[f(\theta)]\\
&= \mathbb{E}_{\theta|Y}\Big[(f(\theta)-\mathbb{E}_{\theta|Y}[f(\theta)])^2\Big]\\
&= \mathbb{E}_{\theta|Y}\Big[f(\theta)^2\Big] - \mathbb{E}_{\theta|Y}[f(\theta)]^2\\
&= \int_\Theta f(\theta)^2 p(\theta|Y) d\theta - I^2\\
\end{aligned}
$$
To check for variance reduction, we will compare alternatives with the above variance.

**Note**: The variance of $f(\theta)$ depends on both the variance of $\theta$ and the amount of variation in the function $f$. **Functions that change a great amount over the domain will have a high variance and functions that are flat will have a low variance.**

#### Variance of Control Variates

Based on our realization that "flat" functions have lower variance, we will try to engineer "flat" functions that allow us to compute the integral of $f$.

Fix a function $f(\theta), \theta \sim p(\theta|Y)$. Let $h(\theta)$ be a function with known mean $\mu_h= E_{\theta|Y}[h(\theta)]$ and such that $h(\theta)$ is correlated with $f(\theta)$. We call $h$ the **control variate** for $f$.

We define the **control variate Monte Carlo estimate** of $E_{\theta|Y}[f(\theta)]$ to be
$$
\widehat{I}_{control} = {1\over S} \sum^S_{s=1} f(\theta) - c(h(\theta)-\mu_h)
$$
where $c$ is our choice of a constant.

The variance of this estimator is
$$
\sigma^2_{\widehat{I}_{control}} = \sigma^2_f + c^2\sigma_h^2 - 2c \times cov[f(\theta), h(\theta)]/S
$$
Thus, we see that when $cov[f(\theta), h(\theta)] \neq 0$ there is hope for variance reduction, $\sigma^2_{\widehat{I}_{control}}/S< \sigma^2_f/S$.

#### Nitty Gritty of Control Variates

Using control variate Monte Carlo requires that we:

1. choose a control variate $h$ with known mean and who is correlated with $f$.
2. choose a constant $c$.

Typically, we want to choose an $h$ that follows to the trends of $f$ that is easy to integrate.

The value of $c$ that minimizes $\sigma^2_{\widehat{I}_{control}}$ is
$$
c^* = {cov[f(\theta), h(\theta)]\over \sigma^2_h}
$$
In case $cov[f(\theta), h(\theta)]^2$ and $\sigma_h^2$ are difficult to compute analytically, they can be empirically estimated from the samples.

When we choose the optimal $c=c^*$, the variance of $\widehat{I}_{control}$ is
$$
\sigma^2_{\widehat{I}_{control}} = \Big(1- {cov[f(\theta), h(\theta)]\over \sigma^2_h \sigma^2_f}\Big) \sigma_f^2/S=(1-\rho_{f,h}) \sigma_f^2 / S
$$
where $\rho_{f,h}$ is the correlation between the outputs of $f$ and $h$. Hence, the more $h$ is correlated with the output the greater the variance reduction.

#### Example: Control Variate

