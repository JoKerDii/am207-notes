## Bayesian Modeling

Content:

1. Review of Method of Maximum Likelihood
2. Models for Real Data
3. The Beta-Binomial Model
4. Bayesian Modeling

### 1. Review of Method of Maximum Likelihood

#### Review

1. **Model**: assume observations from a iid outcome with a certain distribution where there is a set of parameters. The likelihood function is the product of individual probability distribution functions
2. **Inference**: we estimate the parameter using MLE by finding the estimator which maximizes the likelihood.
3. **Inference Method**: 
   * Unconstrained Optimization
   * Constrained Optimization

#### Evaluating MLE

If we have the true parameters $\theta$, we can compute the **Mean Squared Error**:
$$
MLE_\theta = \mathbb{E}_{Y^{\theta}}[(\theta_{MLE}(Y^{\theta}) - \theta)^2]
$$
If we don't have the true parameter $\theta$, we can use $\theta_{MLE}$ **predict** or **simulate** data and compare it with the observed data, i.e. sample
$$
Y^{\theta_{MLE}} \sim p(Y | \theta_{MLE})
$$
compare $Y^{\theta_{MLE}}$ and $Y^{\theta}$. 

If the distribution of both sets of data are similar, then we can conclude that our maximum likelihood estimator is doing a good job of explaining data. Whereas if these two distributions are very disparate, then we can conclude the maximum likelihood estimator is not satisfactory in modeling the set of observed data. 

#### Properties of Maximum Likelihood Estimator

MLE is only one of the ways of estimating the unknown parameters of a statistical model. 

**Why Choose MLE**? Asymptotically, given an infinite number of samples, MLE has the following good properties:

1. Consistent
2. Unbiased
3. Minmum Variance

**Why Not Choose MLE**? When the sample size is small, the MLE can be 

1. Overfitted: The MLE can be sensitive to outliers in the data
2. Biased: The average $\theta_{MLE}$, taken over many different data samples, is not $\theta$.
3. Imprecise: The MLE can have high variance.

**What Other Estimators are There**?

1. Method of Moments
2. Minimum Variance Unbiased Estimator
3. Regularized MLE

**Note**: the computation of #1 and #2 can be much more complex than MLE.

#### Limitation of MLE: Overfitting Under Scarcity of Data

When the number of observed data points is small, then any outlier is likely to have a disproportional effect. MLE can overfit to the observation since it's being unduly influenced by the outliers in the data.

#### Regularization

We can prevent MLE from overfitting to the observations when training data is scarce by constraining it from unreasonable values.

If we want the MLE of the parameter $\theta$ of a Bernoulli distribution to avoid extreme values (1 and 0), we need to 'anchor' our estimation of $\theta$ to some 'reasonable' value:
$$
\theta_{MLR, Reg} = {\text{\# positive outcome} + \alpha\over \text{\# total trials} + \beta}
$$
The fraction $\alpha / \beta$ expresses your notion of what is a reasonable looking probability. $\alpha, \beta$ are anchoring our estimation of theta to a reasonable value that we believe should be the case all else being equal.

**Question**: will regularization ruin the properties of MLE? How do you choose the hyperparameters $\alpha, \beta$ in a principled manner?

### 2. Models for Real Data

#### Video Ranking

We can model the outcome, $Y$, of a user rating for a specific YouTube video as a Bernoulli distribution
$$
Y \sim Ber(\theta)
$$
where $\theta$ is the probability that a user will like the video.

Given $Y_1, ..., Y_N$ with $Y_N \overset{iid}{\sim} Ber(\theta)$, denote the total number of likes by $L$. Then $L$ can be modeled with a binomial distribution.
$$
L \sim Bin(N, \theta)
$$
**Model critique**: 

* What are the assumptions made in this model? 
* Are they realistic? is binomial model appropriate for this task?

#### Kidney Cancer Rates

Given a dataset with $N$ number of US counties and the incidents of Kidney cancer in each county, we can model the observed incidents of cancer $C_n$, of the n-th county with a Poisson distribution
$$
C_n \sim Poi(T_n\theta_n)
$$
where $T_n$ is the total population of the county and $\theta_n$ is the true cancer rate of the county.

**Model Critique**: 

* What are the assumptions made in this model? In order that the number of incidents of cancer is poisson distributed, we'd have to assume that there's no correlation between the incidence of cancer within that county, meaning that it can't be more likely to find two people with cancer if we already found one person with cancer. 
* Are the assumptions realistic? If you think biologically rather than statistically probably not. Because often times cancer is an inheritable trait and run in families. So if we find one family member with cancer, we are more likely to find more incidents of cancer within that family.

The MLE of the rate of Poisson distribution for each county is $C_n / T_n$. From cancer studies people have seen that of all the counties with the highest cancer rates, rural area is predominant rather than urban area. This implies that people living in rural counties are probably more likely to get kidney cancer than people living in urban areas. 

Recall that the MLE of Poisson theta is equal to the raw cancer rate $C_n/ T_n$, which means that MLE of this model will suffer the same problem as using the raw cancer rate to reason about the disparity between urban and rural population. So a maximum likelihood model is not the most appropriate for this particular dataset.

#### Birth Weights

There is a task of modeling the birth weights of infants born in the clinic. Naively, given observed birth weights $Y_1, ..., Y_N$, we can model each outcome $Y_n$ with a normal distribution,
$$
Y_n \sim \mathcal{N}(\mu, \sigma^2)
$$
where $\mu$ is the average birth weight for this population and $\sigma^2$ is the population variance.

**Model Critique**: what are the assumptions made in this model? Are they realistic?

* The newborns are not a homogenous group, meaning that within the group of newborns, there is probably going to be subgroups, where the infants within each subgroup are going to be more similar than they are to infants from the other subgroup. This means that it is unlikely that we can use one single distribution for the weight to model the entire population, if there are indeed very distinct subgroups of this population.
* There is a probability distribution that can describe such phenomenon. [later in the course]

**Takeaway**: every time you write down a model for a data set and choose a method for inference, you are making a bunch of assumptions. It's very important of the model critique and evaluation process to identify these assumptions explicitly, and evaluate whether they're appropriate for your particular application.

### 3. The Beta-Binomial Model

#### Incorporate Prior Beliefs

Recall the hyperparameter $\alpha, \beta$ for MLE regularization, what is the effect of them on our estimate? What values should we choose for $\alpha, \beta$?

The choice of the regularization term $\alpha, \beta$ depends on our **prior beliefs** about the coin. The way we choose to incorporate these beliefs does not indicate any uncertainty of the beliefs. i.e. when I anchor the estimation of theta to 1/2 through regularization, I have no way of expressing how confidence I am in my belief that this coin is going to be fair.

We want to incorporate both our belief and the uncertainty of the belief. As we know, **when we want to model uncertainty in a principle fashion, we use a distribution**. Therefore, alternatively, we can incorporate our prior belief about $\theta$ as a distribution , this is called the **prior distribution**. Since $\theta$ is a number between 0 and 1, a beta distribution is an appropriate choice. 

#### The Beta-Binomial Model

A model that involves both a **likelihood for the data** and **prior on the parameters in the likelihood** is called a **Bayesian model**.

Bayesian Model for Coin Flip: 
$$
\begin{cases}
Y &\sim Bin(N, \beta)& (\text{Likelihood})\\
\theta &\sim Beta(\alpha, \beta) & (\text{Prior})
\end{cases}
$$
where $\alpha, \beta$ are called **hyperparameters** of the model.

Now, computing the MLE no longer makes sense (since the MLE only considers the likelihood). Luckily, Bayes' Rule allows us to derive a distribution that considers both the prior and the likelihood:
$$
p(\theta|Y) = {p(Y|\theta) p(\theta)\over p(Y)} = {p(Y,\theta)\over \int p(Y,\theta) d\theta}
$$
where 

* $p(Y|\theta)$ is likelihood
* $p(\theta)$ is prior
* $p(Y)$ is marginal data likelihood
* The distribution $p(\theta |Y)$ is called the posterior. 

Posterior is proportional, meaning equal up to a constant, to the likelihood times prior, because the bottom term of Bayes Rule is the likelihood of the data. Since data is observed and fixed, this term is a constant.

#### Posterior for the Beta-Binomial Model

In our case, the posterior is given by
$$
p(\theta|Y) = {p(Y|\theta) p(\theta)\over p(Y)} ={ {N\choose Y} \theta^Y(1-\theta) ^{N-Y} {1\over \Beta(\alpha,\beta)} \theta^{\alpha-1} (1-\theta)^{\beta-1}\over p(Y)} 
$$
We can rewrite the posterior as
$$
p(\theta|Y)= const \times \theta^{(Y+\alpha)-1} (1-\theta)^{(N-Y+\beta)-1}
$$
where $const = {{N\choose Y}\over \Beta(\alpha,\beta) p(Y)}$ must be the normalizing constant for $p(\theta|Y)$. This means this constant has to be the normalizing constant that turns expression of $\theta^{(Y+\alpha)-1} (1-\theta)^{(N-Y+\beta)-1}$ into a valid PDF.

Takeaway: when we use a binomial likelihood and a beta prior, our posterior is a beta distribution $Beta(Y+\alpha, N-Y+\beta)$, with parameters including both the observed data and our prior beliefs. 