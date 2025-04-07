## Bayesian vs Frequentist Inference

Content:

1. Review of Bayesian Modeling
2. Examples of Conjugate and Non-Conjugate Models
3. Connections to Frequentist Inference

### 1. Review of Bayesian Modeling

#### Bayesian Modeling

We can interprete Bayesian Inference as a transformation of beliefs. That is you start with a prior belief about the model should be, then you observe data encoded by the likelihood model. The posterior through combining the prior and the likelihood represents a transformed belief. It's what the prior belief has become after being updated by seeing actual data. 

For example, if in your prior belief, you expected the model parameter to have a low value, but in actual reality, a higher value fits the data better, then the posterior becomes a compromise between your prior belief of the low value in the higher values suggested by the data. The uncertainty in the posterior belief is going to be a function of the uncertainty of the prior belief as well as the value of the data that was observed [A relationship to be explored later].

#### Model Evaluation

The product of Bayesian modeling is a posterior distribution. Rather than one single best model, we get a distribution over multiple plausible models. How do we evaluate a distribution over models against our observed data?

While intuitive, the posterior predictive check is not actually a practical procedure for most applications. This is because in most cases the actual observed data cannot be easily visualized. Secondly, it would be hard to rigorously compare the fit of two different models based on simple visual diagnostics.

1.  (**Posterior Predictive Check**) We can also compare the synthetic data generated from our posterior predictive:

   A. Sample from the posterior $\theta_s \sim p(\theta |Y)$.

   B. Plug the posterior samples into the likelihood, and sample synthetic data from the likelihood $Y_s \sim p(Y|\theta_s)$.

2. (**Log-likelihood**) We can compute the marginal log-likelihood of the data under our posterior. That is, give a set of test data $\{y_1^*, ..., y_M^*\}$, compute
   $$
   \log \prod_{m=1}^{M} p(y^*_m|\text{Data}) = \sum^M_{m=1} \log p(y^*_m|\text{Data}) = \sum^M_{m=1}\log\int_{\Theta} p(y^*_m|\theta)p(\theta|\text{Data})d\theta
   $$
   This is called a **log-likelihood** of the data under the Bayesian model.

   **Explanation**: Just as we can evalute the fit of a single best model against the data by computing the log likelihood. If we have a set of plausible models instead of one, we can compute the **average** likelihood of the data under each model. In the case of posterior distribution where we have infinite number of plausible models, this averaging means that we take the **integral** of the likelihood over all plausible models under the posterior. Taking the log of this average, this is called the log likelihood of the data under bayesian model. Essentially, this is measuring how well on average do the models in the posterior fit our observed data.

#### Components of Bayesian Inference

We see that in order to evaluate Bayesian models we need to be able to perform two tasks:

1. Integration over the posterior (required by log-likelihood)
2. Sampling from the posterior (required by the posterior predictive check)

Both requirements becomes eariser if know the closed form expression for the posterior, i.e. if the posterior is recognizable and tractable distribution. e.g. if the prior is conjugate to the likelihood.

### 2. Examples of Conjugate and Non-Conjugate Models

#### Bayesian Model for (Univariate) Gaussian Likelihood with Known Variance

The Bayesian Model:

Let $Y\sim \mathcal{N}(\mu,\sigma^2)$, with $\sigma^2$ known. We place a normal prior on $\mu$, $\mu \sim \mathcal{N}(m,s^2)$.

Question: Is the choice of prior apropriate?

**Inference:** The posterior $p(\mu|Y)$ is then:


$$
\begin{aligned}
p(\mu | Y) = \frac{p(Y| \mu)p(\mu)}{p(Y)} = \frac{\overbrace{\frac{1}{\sqrt{2\pi \sigma^2}} \mathrm{exp} \left\{-\frac{(Y - \mu)^2}{2\sigma^2}\right\}}^{\text{likelihood}} \overbrace{\frac{1}{\sqrt{2\pi s^2}} \mathrm{exp} \left\{-\frac{(m - \mu)^2}{2s^2}\right\}}^{\text{prior}}}{p(Y)}
\end{aligned}
$$
We can simplify the posterior as:
$$
\begin{aligned}
p(\mu | Y) &= const *\frac{\mathrm{exp} \left\{ -\frac{s^2(Y - \mu)^2 + \sigma^2(m - \mu)^2}{2s^2\sigma^2}\right\}}{p(Y)} \\
&= const *\mathrm{exp} \left\{ \frac{s^2Y^2 + \sigma^2m^2}{\sigma^2 s^2}\right\}\mathrm{exp} \left\{ -\frac{(s^2 + \sigma^2)\mu^2 - 2(s^2Y + \sigma^2m)\mu}{2s^2\sigma^2}\right\}\\
&= const* \mathrm{exp} \left\{ -\frac{\left(\mu - \frac{s^2Y + \sigma^2m}{s^2 + \sigma^2} \right)^2}{2s^2\sigma^2}\right\}\quad \text{(Completing the square)}
\end{aligned}
$$
Thus, we see that the posterior is a normal distribution, $\mathcal{N}\left(\frac{s^2Y + \sigma^2m}{s^2 + \sigma^2}, s^2\sigma^2\right)$.

#### Bayesian Model for (Univariate) Gaussian Likelihood with Known Mean

The Bayesian Model:

Let $Y\sim \mathcal{N}(\mu, \sigma^2)$, with $\mu$ unknown. We place an inverse-gamma prior on $\sigma^2$, $\sigma^2 \sim IG(\alpha, \beta)$.

Question: is the choice of prior appropriate?

**Inference**: The posterior $p(\sigma^2|Y)$ is then:
$$
\begin{aligned}
p(\sigma^2 | Y) = \frac{p(Y| \sigma^2)p(\sigma^2)}{p(Y)} = \frac{\overbrace{\frac{1}{\sqrt{2\pi \sigma^2}} \mathrm{exp} \left\{-\frac{(Y - \mu)^2}{2\sigma^2}\right\}}^{\text{likelihood}} \overbrace{\frac{\beta^\alpha}{\Gamma(\alpha)} \left( \sigma^2\right)^{-\alpha -1}\mathrm{exp} \left\{-\frac{\beta}{\sigma^2}\right\}}^{\text{prior}}}{p(Y)}
\end{aligned}
$$
We can simplify the posterior as:
$$
\begin{aligned}
p(\sigma^2 | Y) &= const * \left( \sigma^2\right)^{-(\alpha + 0.5) -1}\mathrm{exp} \left\{-\frac{\frac{(Y-\mu)^2}{2} + \beta}{\sigma^2}\right\}
\end{aligned}
$$
Thus, we see that the posterior is an inverse gamma distribution, $IG\left(\alpha + 0.5, \frac{(Y-\mu)^2}{2} + \beta\right)$.

#### Bayesian Model for (Univariate) Gaussian Likelihood with Unknown Mean and Variance

Let $Y \sim \mathcal{N}(\mu, \sigma^2)$, with both parameters unknown. We place a normal prior on $\mu$, $\mu \sim \mathcal{N}(m, s^2)$, and an inverse-gamma prior on $\sigma^2, \sigma^2 \sim IG(\alpha, \beta)$.

The posterior $p(\sigma^2|Y)$ is then:


$$
\begin{aligned}
p(\mu, \sigma^2 | Y)  = \frac{\overbrace{\frac{1}{\sqrt{2\pi \sigma^2}} \mathrm{exp} \left\{-\frac{(Y - \mu)^2}{2\sigma^2}\right\}}^{\text{likelihood}} \overbrace{\frac{1}{\sqrt{2\pi s^2}} \mathrm{exp} \left\{-\frac{(m - \mu)^2}{2s^2}\right\}}^{\text{prior on } \mu}\overbrace{\frac{\beta^\alpha}{\Gamma(\alpha)} \left( \sigma^2\right)^{-\alpha -1}\mathrm{exp} \left\{-\frac{\beta}{\sigma^2}\right\}}^{\text{prior on }  \sigma^2}}{p(Y)}
\end{aligned}
$$


**Question**: Can the posterior be simplified so that we recognize the form of the distribution? 

No. These two priors we've chosen together are not conjugate to the normal likelihood. This means even if you always wanted to choose a mathematically convenient prior rather than one that's meaningful in the context of the real life application. It isn't always so easy to do so. 

#### Non-Conjugate Models

We know that conjugate priors yield closed-form expressions for the posterior. In all our examples, these posteriors have been both easy to sample from and easy to integrate over. That is, we can evaluate our Bayesian models: can compute the log-likelihood of the data and perform posterior predictive checks. This is **why we care about having a closed-form expression for the posterior**.

So why would we ever consider non-conjugate priors?

Suppose that $Y\sim \mathcal{N}(\mu, 2)$ represent the height of a person randomly selected from a population. Would the conjugate prior $\mu\sim \mathcal{N}(5.7, 1)$ be appropriate for this application? No because negative numbers are not meaningful, so perhaps using a prior that's supported only on positive numbers will be more appropriate. (e.g. Gamma prior for $\mu$).

If we wanted to use the prior $\mu \sim Ga(5.7, 1)$, would we be able to derive a closed form expression for the posterior? No because a gamma prior is not conjugate to the normal likelihood, so not able to derive a closed form expression for the posterior distribution. 

The **problem with Bayesian models** is that, if we want to choose flexible forms for the likelihood that can model complex trends in the data, enrich forms for the prior so that it can encapsulate appropriate beliefs about the application, more often we are not able to capture the posterior with a closed form expression, we would not be recognize the posterior easily as an existing distribution. This means that **performing inference for and evaluating complex Bayesian models becomes the primary challenge for the paradigm**.

### 3. Connections to Frequentist Inference

1. If Bayesian models are so frequently intractable, is it worthwhile to engage with them?

   Maximum likelihood models seem easy and straightforward, why don't we always just do that? 

   There is choice and tension between Bayesian and non-Bayesian frameworks.

2. Do Bayesian models learn something fundamentally different about the data than maximum likelihood models? 

   If so, is it a good thing or a bad thing. 

3. Since we know the maximum likelihood models have nice theoretical guarantees as the number of observation goes to infinity, do Bayesian models also have similar correctness guarantees?

#### Point Estimates from the Posterior

In order to make the analogy between a Bayesian model and a Maximum Likelihood Model, we can distill posterior distribution over plausible model into a single 'best' model. This means to derive a point estimate for the parameters $\theta$ in the likelihood from the Bayesian model. There are two common ways to do it:

1. **the posterior mean**: the average of all the models in the posterior.
   $$
   \theta_{\text{post mean}} = \mathbb{E}_{\theta \sim p(\theta|Y)}[\theta | Y] = \int \theta p(\theta|Y)d\theta
   $$

2. **the posterior mode or maximum a posterior (MAP)** estimate: the model that scores the highest under the posterior distribution.
   $$
   \theta_{MAP} = \arg \max_{\theta} p(\theta | Y)
   $$

**Question**: is it better to summarize the entire posterior using a point estimate? i.e. why should we keep the posterior distribution around?

The goal of Bayesian modeling is not to produce a single best model, rather it is to infer a distribution over plausible models. And the spread of this distribution represents the **uncertainty** over which is the best model. [Will explore the value of uncertainty in modeling in hw]

Another limitation of using a point estimate rather than the entire posterior distribution, is that distributions cannot be well summarized by a single point, and these point estimates can be extremely misleading. 

For example, a posterior mode can be an atypical point of the distribution, where there is not much mass around that posterior mode value. It's also problematic to use posterior mean to represent the posterior, e.g. if the posterior distribution is multimodal, then the mean will be located in the region that the posterior considers highly unlikely. This mean the average of all of models in the posterior does not necessary be the most likely model to fit the data.

#### Comparison of Posterior Point Estimates and MLE

**Beta-Binomial Model for Coin Flips**

* Likelihood: $\text{Bin}(N,\theta)$
* Prior: $\text{Beta}(\alpha, \beta)$
* MLE: $Y/N$
* MAP: ${Y+\alpha-1\over N+\alpha+\beta-2}$
* Posterior Mean: ${Y+\alpha \over N+\alpha+\beta}$

**Question**: What is the effect of the prior on the posterior point estimates? Imagine if $Y=10, N=11, \alpha=100, \beta=300$. What if $Y=1000,N=11000,\alpha=1,\beta=3$?

We can intuitively explore the effects of the hyperparameters of the prior - alpha and beta. When $Y$ and $N$ are both very small, then the effect of your choice of alpha and beta will be very dominant.

#### The Coin Toss Example: Revisited Yet Again

One of the problems of MLEs is that they tend to overfit the outliers and the noise in the data when there are very few observations. Recall that one way to prevent the MLE from overfitting is to add regularization terms. Regularization is way to constrain MLE away from undesirable values.

In the case of estimating a probability $\theta$, we want to constrain the maximum likelihood away from values like 0 or 1, which are very extreme. To do so, we propose a regularization scheme that involves anchoring the maximum likelihood estimate to some pre-determined fraction $\alpha / \beta$ , e.g. $1/2$. Expressing prior belief that all things being equal, a coin is probably fair until proven otherwise. 
$$
\theta_{MLE \space Reg} = {Y+\alpha \over N+\beta}
$$
This is very similar to the MAP and posterior mean estimates. In fact, **one effect of adding a prior is that it regularizes our inference about $\theta$.** This means adding a prior prevents you from concluding a very extreme things about data based on very few observations. We can in fact show that every regularization scheme you put on the MLE can be expressed as the effect of a prior for a Bayesian model for the same data. In other words, for every regularization scheme, there exists an equivalent prior that not only achieves the same effect, but also yields a posterior point estimate that's exactly equal to the regularized MLE.

But this does not answer the following questions yet: How should we choose beta and alpha? what if we chose incorrectly? does wrong choice of alpha and beta ruin the nice properties of MLE?

**Question**: what happens to the MAP and posterior mean estimates as $N$ (and hence $Y$) becomes very large?
$$
\lim_{N\rightarrow \infty} {Y+\alpha-1\over N+\alpha+\beta-2}= {Y\over N}
$$
As the number of obserations goes to infinity, regularized MLE as well as posterior point estimates, all approach the MLE. Since we know asymptotically MLE approaches the ground truth value of theta, this mean that regularized MLE as well as posterior point estimates will also recover the ground truth theta. In short, if you choose inappropriate alphas and betas, you just need to wait until you observe enough data. Eventually, the effects of your choice of alpha and beta will be completely overwhelmed and washed out by the data itself. 

#### Law of Large Numbers for Bayesian Inference

In Bayesian inference, we are more interested in the properties of the asymptotic distribution of the posterior than the asymptotic behavior.

**Theorem: (Berstein-won Mises)**: Under some conditions, as $N \rightarrow \infty$, the posterior distribution converges to a Gaussian distribution centred at the MLE with covariance matrix given by a function of the Fisher information matrxi at the true population parameter value.

**Implications**:

* The posterior point estimates (both posterior mean and posterior mode) approach the MLE, with large sample sizes.

  This is because Gaussian distribution is symmetric and unimodal, its posterior mean will be equal to its posterior mode. Since posterior distribution is centered at MLE, both point estimates will be equal to the MLE.

  This means the estimation of beta binomial model can be extended to many other types of model as well. If you are worried about choosing the wrong prior for bayesian model, you just need to collect more data to wash out the effects of mistake. 

* It may be valid to approximate the posterior with a Gaussian, with large sample sizes. 

  This means it may be valid to approximate the posterior with just a Gaussian.

#### Computational Comparisons

1. **Computation of the MLE is an optimization problem.** There are many established methods for performing optimization, even when the objective fucntion is not convex - i.e. many local optima.

   More importantly, there are algorithms to perform general automatic optimization (e.g. gradient descent) on a large class of functions - i.e. MLE can be performed in a scalable general framework that can accommodate a wide range of data sets, models, and applications. 

2. **Computation of the posterior is a process of choosing the right priors and the posterior distribution is of the same type as the prior.** The derivation is simple so long as we use conjugate priors. But many intuitively appropariate priors (like the inverse gamma and normal priors for a univariate gaussian) are not conjugate. In those cases, it becomes intractable to 

   * compute posterior mean
   * simulate samples from the posterior (and hence simulate samples from the posterior predictive)

   thus inference of the posterior is the main bottleneck.
