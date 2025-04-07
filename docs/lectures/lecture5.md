## Sampling for Posterior Simulation

Content:

1. Basics of Sampling
2. Inverse CDF Sampling
3. Rejection Sampling
4. Gibbs Sampling

### 1. Basics of Sampling

#### Motivation

Nearly every computation for Bayesian models requires being able to obtain samples from the posterior.

1. Generating synthetic data from the posterior predictive requires us to sample from the model parameters from the posterior, and then sample data given each of these parameters samples
2. Empirically computing the posterior predictive mean requires samples from the posterior
3. Evaluating the fit of the Bayesian model, i.e. empirically computing the marginal data log-likelihood, requires us to average the likelihood of the data over parameter samples from the posterior.

This is why Bayesian Inference relies crucially on the tractability of the posterior. But the posterior of most Bayesian models do not have nice simple closed form.

#### What is Sampling?

Given a distribution $p(x)$ over a space $\mathbb{R}^D $, **sampling** $x \sim p(x)$ means to generate a random $x \in \mathbb{R}^D$, such that the pdf describes the asymptotic frequencies of the samples generated $p(x)$.

A **sampler** is an algorithm or procedure that produces numbers with a certain distribution $p(x)$.

In practice, "random numbers" are simulated using deterministic algorithms called **pseudo number generators**. A pseudo number generator takes an initial value, the random seed, and produces an array of random-looking numbers (from a uniform distribution).

**Note**: for a given pseudo number generator, if the random seed is fixed then the random number output will also be fixed.

#### Simulating a Uniform Random Variable: Linear Congruence

Fix an integer $c > N$ and fix integers $a,b >0$.

1. Seed: set an integer $0 \leq s_0 <c$
2. Iterate $N$ times: $s_n = (a s_{n-1}+b)_{mod \space c}$

Output is an array of random integers $[s_0, ...,s_N]$ in $[0,c]$.

For an array of random real numbers in $[0,1]$, we compute $[{s_0\over c}, ...,{s_N\over c}]$.

**The simulation of all random variables is based on the simulation of the uniform distribution over $[0,1]$.** i.e. if you can simulate sampling from a uniform distribution over $[0,1]$, you can build up a sampler to sample from any distribution.

**Note**: the apparent randomness of the output is sensitive to choices of a, b, c.

```python
#parameters of linear congruence algorithm
c = 100
a = 11
b = 5
#total number of simulations
N = 10
#random seed
s_current = 3
#array of random numbers
random_numbers = []

#run the linear congruence algorithm N times
for n in range(N):
    s_next = (a * s_current + b) % c
    random_numbers.append(s_next)
    s_current = s_next
    
#convert random integers to random real values in [0, 1]    
random_numbers = np.array(random_numbers) * 1.
random_numbers /= c
#print
random_numbers
```

### 2. Inverse CDF Sampling

#### The Cumulative Distribution Function

The cumulative distribution function (CDF) of a real-valued random variable $X$ with continuous pdf $f_X$ is defined as
$$
F_X(a) = \mathbb{P}[X\leq a] = \int^a_{-\infty} f_X(t)dt
$$

#### Inverse CDF Sampling: An Intuition

The idea behind inverse CDF sampling is that while it is sometimes difficult to generate values for $X$ with the relative frequency described by pdf, $f_X$, it can be easier to generate values for $X$ using the CDF.

Th intuition is as follows:

1. While the support of the PDF and CDF can be unbounded, the range of the CDF is bounded between 0 and 1.
2. The CDF for a continuous single-variable PDF is an invertible function (on the support of the PDF). That is, each value between 0 and 1 corresponds to a unique value of the random variable $X$.
3. Values of $X$ that lie under peaks of the PDF occupy larger portions of the interval $[0,1]$. That is, the range of the CDF $[0,1]$, can be subdivided to exactly reflect the areas of high probability mass and low probability mass under the PDF.

So if we uniformly sample values in the range of the CDF $[0,1]$, and find the corresponding $X$ values for these samples (using the inverse function of the CDF), we obtain samples of $X$.

#### Inverse CDF Sampling: Algorithm

We use a random variable $U$ with uniform distribution over $[0,1]$ to simulate a univariate random variable $X$ with PDF $f_X$, where:

* We know the analytical form of the CDF of $X, F_X$.
* We know the analytical form of the inverse of the CDF of $X, F_X^{-1}$.

To simulate $X$, we repeat for $N$ number of samples:

1. Sample $U_n \sim U(0,1)$
2. Compute $X_n = F_X^{-1}(U_n)$

**Question**: How do we simulate multivariate random variables? What if we don't have the analytical form of $F_X^{-1}$?

#### Inverse CDF Sampling: Proof of Correctness

#### Simulating an Exponential Random Variable

We can use a uniform random variable $U \sim U(0,1)$ to simulate an exponential variable $X \sim \exp(\lambda)$. Recall that the exponential CDF is
$$
F_X(x)=1-\exp\{-\lambda x\}
$$
The inverse of the CDF can be found by solving for the input $x$ of the CDF
$$
\begin{aligned}
y=&1-\exp\{-\lambda x\}\\
\exp\{-\lambda x\}=&1-y\\
-\lambda x=& \log(1-y)\\
x=&-{1\over y}\log(1-y)\\
\end{aligned}
$$

where we take log to be base $e$. Thus,
$$
F_X^{-1}(y) = -{1\over y} \log(1-y)
$$

#### What Can We Simulate?

We can simulate sampling from univariate continuous distributions with closed form inverse CDF's e.g. exponential random variables. We can't simulate normal random variables since the normal CDF does not have a closed form inverse. We can't simulate multivariate random variables.

We can simulate discrete random variables.

### 3. Rejection Sampling

#### Rejection Sampling

The idea is to by-pass the problem of sampling from a difficult distribution $f_X$, by:

1. Approximate $f_X$ (called the **target distribution**), with a PDF $g$ (called the **proposal distribution**) that is easy to sample.
2. Sample from $g$ and reject the samples that are unlikely to be from $f_X$.

#### Rejection Sampling: Algorithm

We can use rejection sampling o simulate multivariate random variables and random variables for which we don't have a closed form for $F_X^{-1}$. We choose a proposal distribution $g$ such that (1) the support of $g$ covers the support of $f$ and (2) there is a constant $M>0$ with ${{f_X(y)\over g(y)}\leq M}$ for all $y$.

i.e. We need (1) $g$ to be non-zero where ever $f$ is and $g$ must decay slower than $f$, (2) $M*g$ is an upper bound of $f$.

To simulate $X$, we repeat until $N$ samples are accepted:

1. sample $Y_k \sim g(Y)$
2. Sample a random height $U_k \sim U(0,1)$
3. if $U_k < {f_X(Y_k)\over M_g(Y_k)}$ then accept $Y_k$ as a sample, else reject

**Question**: How long does it take to accumulate $N$ samples? What is the effect of the choice of the proposal distribution $g$ have on the sampling process? Now that we can sample from any distribution, this less is over right?

#### Rejection Sampling: Efficiency

Given a proposal $Y=y$, the probability of accepting it is
$$
\mathbb{P}\Big[U \leq {f(Y)\over Mg(Y)}\Big| Y=y\Big] = {f(y)\over M g(y)}
$$
So the overall probability $p$ of accepting any given proposal can be computed by integrating out $y$:
$$
\begin{aligned}
p &= \int_{\mathbb{R}} {f(y)\over M g(y)} g(y) dy\\
&={1\over M}\int_{\mathbb{R}} f(y)dy\\
&={1\over M}
\end{aligned}
$$
We see that the expected number of times it takes to darw and accept a sample $X=x$ is precisely $M$. This means that roughly $(M-1)/M$ samples drawn from $g$ will be rejected. In order to accumulate $N$ numbers of samples, we need $M*N$ number of iterations of rejection sampling. When $M$ is large, i.e. we need to scale $g$ a lot in order to be larger than $f$, the rejection sampler is very inefficient.

#### Simulating a Normal Random Variable

If we can simulate normal variable $X \sim \mathcal{N}(0,1)$, then we can simulate any normal variable $Y \sim \mathcal{N}(\mu, \sigma^{2})$ by setting $Y = \sigma * X + \mu$ (called **re-parameterization**).

Since the standard normal distribution is symmetric about $X=0$, we only need to simulate samples from the non-negative side of $\mathcal{N}(0,1)$ and indepedently sample a sign (+ or -) for each sample using a Bernoulli distribution with $\theta=0.5$.

Simulating the positive half of the normal distribution means we need to scale the normal PDF by a factor of 2, so that is integrates to 1 over the non-negative real numbers.

A natual candidate for a PDF to cover the non-negative half of the standard normal PDF is the exponential PDF.

#### What Can We Simulate?

We can simulate any continuous or discrete random variables as long as we can find a suitable proposal distribution.

But as the dimensions of teh random variable increases or for inappropriate choices of the proposal distribution, the efficiency of this sampler may be very low. This is due to the rejection rate of rejection sampling is going to be intolerably high.

#### Limitations of Rejection Sampling in High Dimensions

Since the acceptance rate for rejection sampling is $1/M$, where $M$ is a constant that bounds ${f_X(y)\over g(y)}$ for all $y$, we'd want to make $M$ as close to $1$ as possible, i.e. we want $g(y)$ to be approximately equal to $f_X(y)$. In general this is very difficult to achieve, especially in high dimensions.

**Example**: let's say the target distribution is a $D$- dimensional Gaussian $\mathcal{N}(0, \sigma_f^2\bold{I}_{D\times D})$, where $\bold{I}_{D\times D}$ is a $D\times D$ identify matrix. And proposal distribution is a $D$-dimensional Gaussian $\mathcal{N}(0, \sigma_g^2\bold{I}_{D\times D})$, where $\sigma_g > \sigma_f$. We can compute the optimum value of $M$ to be $\Big({\sigma_g\over \sigma_f}\Big)^D$. But this is a value that scales with $D$. For example, if $D=1000$ and ${\sigma_g\over \sigma_f}=1.01$, then the probability of accepting a sample will be $1/M = 0.000047$ - rarely accepting the sample.

**Conclusion**: rejection sampling is intuitive and easy to implement, can be highly inefficient, when the $g$ we pick is very unlike the $f$ target distribution or when the dimensionality of both distributions is extremely high.

### 4. Gibbs Sampling

#### Semi-Conjugate Priors

Let $Y \sim \mathcal{N}(\mu, \sigma^{2})$, with boths parameters unknown. We place a normal prior on $\mu$, $\mu \sim \mathcal{N}(m, s^{2})$, and an gamma prior on $\sigma^2$, $\sigma^2 \sim {IG}(\alpha, \beta)$.

The posterior $p(\mu, \sigma^2| Y)$ is then:

$$
\begin{aligned}
p(\mu, \sigma^2 | Y)  = \frac{\overbrace{\frac{1}{\sqrt{2\pi \sigma^2}} \mathrm{exp} \left\{-\frac{(Y - \mu)^2}{2\sigma^2}\right\}}^{\text{likelihood}} \overbrace{\frac{1}{\sqrt{2\pi s^2}} \mathrm{exp} \left\{-\frac{(m - \mu)^2}{2s^2}\right\}}^{\text{prior on } \mu}\overbrace{\frac{\beta^\alpha}{\Gamma(\alpha)} \left( \sigma^2\right)^{-\alpha -1}\mathrm{exp} \left\{-\frac{\beta}{\sigma^2}\right\}}^{\text{prior on }  \sigma^2}}{p(Y)}
\end{aligned}
$$
Note that:

1. If we condition on $\sigma^2$ (i.e. hold it constant) then $p(\mu | Y,\sigma^2)$ is a normal PDF, $\mathcal{N}(\mu; {s^2 y + \sigma^2m\over s^2 + \sigma^2}, s^2 \sigma^2)$.
2. If we condition on $\mu$ (i.e. hold it constant) then $p(\sigma^2 | Y, \mu)$ is an inverse gamma PDF, $IG(\sigma^2; \alpha + 0.5,{(y-\mu)^2\over 2}+\beta)$.

**The conditional of the posterior is easy to sample from while the joint posterior is not.** In this case, we call the priors **semi-conjugate** for the likelihood.

This observation can be turned into an intuition for building a sampler. If you can't directly jointly sample both $\mu, \sigma$, maybe we can sample them one at a time, using the closed form conditional posterior distributions.

#### Gibbs Sampling: An Intuition

If we start at a point $(x^{(0)},y^{(0)})$ sampled from the joint distribution $p(X,Y)$, we can get to the next point $(x^{(1)},y^{()}) \sim p(X,Y)$ through a "stepping-stone" $(x^{(1)},y^{(0)})$, where we updated the first coordinate by $x^{(1)} \sim p(X|Y=y^{(0)})$. From there, we update the second coordinate $y^{(1)}\sim p(Y|X=x^{(1)})$. We can repeat this piece-wise process of updating $\mu, \sigma$ iteratively and obtain as many samples of $\mu, \sigma$ as we want. 

The initial samples may be unlikely under $p(X,Y)$, but this process will eventually lead us to a high density area in $p(X,Y)$ and we will mostly sample there. This process starts from arbitrary choices of $\mu, \sigma$ , it's unlikely that our choice is going to be very likely under the posterior. However, since we update $\mu, \sigma$ each time iteratively by sampling from posterior conditionals, we will quickly make our way into a region that is very likely under the posterior distribution.

#### Gibbs Sampling: Algorithm

To simulate $N$ samples of a $D$- dimentional multivariate random variable $X$ with PDF $f_X$, we

1. initialization: choose any $x^{(0)} = [ x_1^{(0)} ... x_D^{(0)}]$

2. iterate $N$ times: sample $x^{(n+1)} = [x_1^{(n+1)} ... x_D^{(n+1)}]$ by

   a. Initilization: sample $X_1^{(n+1)}$ from the conditional distribution
   $$
   f_X(X_1 | X_2 = x_2^{(n)}, ..., X_D = x_D^{(n)})
   $$
   b. iterate from $d=2$ through $d=D$: sample $x_d^{(n+1)}$ from the conditional distribution
   $$
   f_X(X_d | X_1 = x_1^{(n+1)}, ..., X_{d-1} = x_{d-1}^{(n+1)},X_{d+1} = x_{d+1}^{(n)}...,X_D = x_D^{(n)})
   $$

**Claim**: When $N$ is large enough, the latter portion of the samples we obtain will be from the distribution $X$.

**Question**: Why is this algorithm a valid sampler? That is, how do we prove that the samples we obtain are actually distributed as $f_X$? What is the effect of the initialization $x^{(0)}$?

### 5. Summarizing Sampling

#### Samplers for Simulating Random Variables

1. Linear Congruence Pseudo Random Number Generator
2. Inverse CDF Sampling
3. Rejection Sampling
4. Gibbs Samplings

#### How to Evaluate a Sampler

1. Correctness: every sampler must come with a proof of correctness - the numbers produced by the sampler have the distribution $p(X)$. Many intuitively sensible ways of sampling can fail to be correct.
2. Efficiency: every sampler must be analyzed for it's efficiency - i.e. how many times the procedure must run before it accepts a sample. You should also be aware of how the efficiency is affected by the dimension of $X$.
3. Sufficiency of repetition: some samplers like Gibbs come with the guarantee that if you repeat the procedure enough times (asymptotically), you will eventually be sampling from $p(X)$. (how many times is enough?)

#### Why are We Sampling Again?

The primary objects in a Bayesian model are the **posterior distribution** over parameters and the **posterior predictive distribution** over observations.

Evaluating the model, forming scientific hypotheses about the data and making predictions all require sampling from one of the two distributions.

Our goal is studying sampling is to develop a set of procedures that allows us to sample from arbitrary distributions i.e. posterior. So that eventually, efficient samplers will make Bayesian inference more 'generic', less 'artisanal'.

#### What If We Want Posterior Point Estimates?

Posterior samples allow us to approximately compute posterior point estimates, for example, we can approximate the posterior mean as
$$
\mathbb{E}_{\theta|Y}[\theta] = \int_{\Theta}\theta p(\theta|Y)d\theta \approx {1\over S} \sum^S_{s=1}\theta_s, \theta_s \sim p(\theta|Y)
$$
In fact, for any function $f$ of $\theta|Y$, we can estimate the expected value of $f$ by first sampling $S$ samples from the posterior $p(\theta|Y)$ and then compute the average value of $f$ on these samples:
$$
\mathbb{E}_{\theta|Y} [f(\theta)] = \int_{\Theta} f(\theta) p(\theta|Y) d\theta \approx {1\over S}\sum^{S}_{s=1} f(\theta_s), \theta_s \sim p(\theta | Y)
$$
**Question**: as in MLE, whenever there is a statistical estimate, we can always ask, is this estimate consistent, unbiased, or minimal variance. Even we could obtain samples from posterior, is this the way we should use the samples to estimate quantities associated with the posterior? 
