

## Regression Modeling Example

### 1. Model Fitting (Deterministic)

Question: What do we mean by choosing best line, $\hat{y} = w_1x+w_0$?

The **model fitting** process:

1. Choose an overall error metric. This metric is called the **loss function** or **training objective**.
   $$
   \mathcal{L}(w_1, w_0) = {1\over N} \sum^N_{n=1} |y_n - (w_1x_n+w_0)|^2, \space (\text{Mean Squared Error Loss})
   $$

2. Set up the problem of finding coefficients or **parameters**, $w_0, w_1$, such that the loss function is **minimized**:
   $$
   \arg \min_{w_0, w_1} \mathcal{L}(w_1,w_0) = \arg \min_{w_1,w_0} {1 \over N} \sum^N_{n=1} |y_n - (w_1x_n+w_0)|^2
   $$
   This show why MSE is picked as loss function. If we take the derivative of it, with respect to either one of the parameters $w_0, w_1$, it would be using the chain rule and the sum rule.

   However, is it right to choose mathematically convenient loss function?

3. Choose a method of minimizing the loss function 

#### What is a statistical model?

**Belief**: the theoretical relationship between $x$ and $y$ is given by $f(x)$. But in real life, due to unpredictable circumstances observed $y$ differ from theoretical $f(x)$ by some random amount $\epsilon$ , called **noise**:
$$
y=f(x)+\epsilon, \epsilon \sim p(\epsilon)
$$
Because we have explicitly modeled the noise in the observed data, we have formulated a **statistical model**. 

Different from previous model, a statistical model is one that explicitly accounts for uncertainty or randomness. It means that it also explains why the observed $y$ would deviate from theoretical $y$.

#### How to Quantify Fitness?

Given our statistical model, a natural way for quantifying how well $f(x) = w_1 x + w_0$ fits the data is by checking how likely our choice of $w_0$ and $w_1$ makes the observed data, i.e. compute
$$
\mathcal{L}(w_1, w_0) = \prod_{n=1}^N p(y_n|x_n, w_1, w_0)
$$
The function $\mathcal{L}(w_1, w_0)$ is called the **likelihood function** - it's a function of parameters we choose for our model, since the observed data is fixed. It's calculated by taking a product over all individual observations, as we are interested in the likelihood of the entire observed data under our model .

The rational is we should favor models of the world that renders our actual observed data the most likely - This principle for model selection is called **Maximum Likelihood Principle**.

The way to use Maximum Likelihood Principle to compare models is that, let's say we have two linear models, we calculate likelihood function for each of both by using their parameters. The model with the larger likelihood is where the data's order of magnitudes are more likely than the other model. We should choose the model that renders our data more likely.

### 2. Model Fitting (Statistical)

Question: What do we mean by choosing best line, $\hat{y} = w_1x+w_0$?

The **model fitting** process:

1. Choose a method of estimation for statistical models. For example, set up the problem of finding coefficients or parameters $w_0, w_1$, such that the likelihood of the data is maximized:
   $$
   \arg \max_{w_0,w_1} = \arg\max_{w_0,w_1} \prod^N_{n=1} p(y_n | x_n, w_1, w_0)
   $$
   Note that there are models whose likelihood functions are too complicated to compute analytically though 0 of derivative or to even compute analytically the derivative in the first place. There are actually many ways to maximize likelihood.

2. Choose a method of computing the estimate. For example, choose a way to maximized the likelihood.

#### Maximum Likelihood and Minimum Mean Square Error

Given our statistical model
$$
y=f(x)+\epsilon, \space \epsilon \overset{iid}{\sim} \mathcal{N}(0,1)
$$
Here we assume that observed data differ from prediction by some random noise. The noise is normally distributed and noise of each observation is independent of noise in another observation - two components of **IID assumption**.

Maximizing the likelihood is equivalent to minimizing the MSE:
$$
\arg \max_{w_0,w_1}\prod^N_{n=1} p(y_n | x_n, w_1, w_0) \equiv \arg\min_{w_0,w_1}{1\over N} \sum^N_{i=1}|y_i - (w_1x_1 + w_0)|^2
$$
Note that for this particular choice of noise, maximizing the likelihood of data is equivalent to minimizing the MSE of model on the observed data. This fact is extremely useful in that 1) derivative of MSE is easily taken, 2) it provides reason of choosing MSE as loss function or overall error metric.

Therefore, the reasons why not choosing MAE but MSE:

* Doing so implies that we are assuming a statistical model for the data where the observation noise is IID distributed like Gaussian, and we are fitting the model by maximizing the likelihood.
* Choosing MSE as overall error metric is entirely reasonable if you believe in your statistical model and you believe your maximum likelihood principle.

Hint: note that
$$
\prod_{n=1}^N p(y_n|x_n,w_1,w_0) = {1\over \sqrt{2\pi1}^N} \exp\Big\{-{\sum^N_{i-1}(y_n-(w_1x_n+w_0))^2\over 2 \times 1}\Big\}
$$
and that
$$
\log p(y|x,w_1,w_0) = N\log({1\over \sqrt{2 \pi 1}}) - {\sum^N_{i=1}(y_n-(w_1x_n+w_0))^2\over 2 \times 1}
$$


### 3. Model Evaluation

After fitting the model (finding coefficients that maximizes the likelihood or that minimizes the loss function), we need to **check the error or residuals of the model**. 

Working with statistical models gives us an advantage in model evaluation, because in a statistical model, modeling assumptions are made explicit, meaning we make it clear about what we assumed about the data were true. This means that if model fails, we  can know exactly one of those assumptions has failed, and systematically check for that failure, and change assumption. 

### 4. Model Interpretation

In addition to evaluating our model on training and testing data, we must also examine the coefficients themselves.

Because even if the model is a perfect fit for the data, it does not mean that this model is capturing anything meaningful about this dataset. The data itself may contain errors and outliers.

 We can come to the conclusion that a model is not reliable or acceptable by simply interpreting model parameters, rather than observing MSE. 

We can disqualify a model not based on numerical metrics but based on our interpretation and our **prior beliefs** that the slope and the intercept of this model cannot be possibly negative or positive.

However, we never incorporated these beliefs explicitly as assumptions into our statistical model.

* Unfair to train a model based on a partial set of criteria and then to disqualify that model based on the complete set
* Inefficient because if we really want the coefficient or intercept to be non-negative, why don't we explicitly incorporate that into our model in the first place.
* Other than incorporate prior beliefs about coefficient and intercept, we also want to incorporate some uncertainty about those beliefs. (The natural way of describing uncertainty is through probability distributions)

### 5. Bayesian Model

In addition to a statistical model that explains trends $f(x)$ and observation noise $\epsilon$, we also want to incorporate our **prior beliefs** about the model. Finally, we want to obtain a measure of **uncertainty** for our parameter estimates as well as our predictions.

Our Bayesian model for linear regression
$$
\begin{aligned}
y&= w_1x+w_0+\epsilon\\
\epsilon & \overset{iid}{\sim} \mathcal{N}(0,1)\\
w_1 & \sim p(w_1)\\
w_0 & \sim p(w_0)
\end{aligned}
$$
Where the prior $p(w_1)$ may express that we want $w_1$ to be non-negative and not too large.

### 6. Bayesian Model Inference

How to learn the parameters in a Bayesian model? 

Baye's Rule gives us a way to obtain a distribution over $w_1, w_0$ given the data $(x_1, y_1), ..., (x_N, y_N)$:
$$
p(w_1,w_0 | x_1, ...,x_N, y_1, ..., y_N) \propto \Big(\prod^N_{n=1}p(y_n|x_n,w_1,w_0)\Big) \cdot p(w_1)p(w_2)
$$
The distribution $p(w_1,w_0 | x_1, ...,x_N, y_1, ..., y_N)$ is called the **posterior** and gives the likelihood of a pair of parameters $w_1, w_0$ given the observed data. Posterior is proportional to the product of the likelihood, as well as the prior.

* $\Big(\prod^N_{n=1}p(y_n|x_n,w_1,w_0)\Big)$ represents how well params fit the data.
* $p(w_1)p(w_2)$ represents how well the params fit priors.

The posterior is a (multivariate) normal distribution, $\mathcal{N}(\mu,\Sigma)$ and we can derive closed form solutions for $\mu$ and $\Sigma$.

### 7. Bayesian versus Frequentist Uncertainty

The main advantage of the Bayesian approach is that rather than obtaining a single "best" estimate of the model parameters, the posterior gives us a distribution over a set of plausible model parameters (with some models being more likely than others).

The spread of this distribution over plausible models naturally gives us a way to quantify our **uncertainty** over which is the "best" model. When the spread is wide (when many very different models are equally very likely), our uncertainty is high. When the spread is narrow (when all likely models look very similar), our uncertainty is low.

We can also obtain a sense of uncertainty over models using the non-Bayesian probabilistic model. Typically, we randomly sample sets of training data point from the training data, on each set, we compute the MLE of the model parameters. This process is called **bootstrapping**.

### 8. Why it's hard

1. Finding the optimal parameters is often very hard, especially if $f(x)$ is not linear, but rather, a complex function represented by a neural network.
2. If we choose more "interesting" or "expensive" priors, or if we choose more complex $f(x)$, then it is often the case that the posterior cannot be computed in closed form.

Both model fitting and inference requires sophisticated algorithms derived from deep theoretical understanding of the models.
