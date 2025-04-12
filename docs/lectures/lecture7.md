## Markov Chain Monte Carlo

Content:

1. Gibbs Sampler for Discrete Distribution
2. Definition of Properties of Markov Chains
3. Markov Chain Monte Carlo

### 1. Gibbs Sampler for Discrete Distribution

#### Gibbs Sampler for a Bivariate Discrete Distribution

Suppose we have two independent random variables $X \sim Ber(0.2)$ and $Y \sim Ber(0.6)$. Their joint distribution is a categorical distribution:
$$
p(X,Y) = [0.12 \space\space 0.48 \space\space 0.08 \space\space 0.32]
$$
over the set of possible outcomes
$$
(X=1,Y=1),(X=0,Y=1),(X=1,Y=0),(X=0,Y=0)
$$
A Gibbs sampler for $p(X,Y)$ will start with a sample $(X=X_n,Y=Y_n)$ and then generate a sample $(X=X_{n+1},Y=Y_{n+1})$ by

1. Sampling $X_{n+1}$ from $p(X|Y=Y_n)$, which equals to $p(X)=Ber(0.2)$
2. Sampling $Y_{n+1}$ from $p(Y|X=X_{n+1})$, which equals to $p(Y)=Ber(0.6)$

We want to compute what is the distribution of the generated sample $(X=X_{n+1}, Y=Y_{n+1})$ given $(X=X_n,Y=Y_n)$, i.e. we want $p(X=X_{n+1}, Y=Y_{n+1}|X=X_n, Y=Y_n)$, but there are $4^2$ of these probabilities - how do we succinctly represent them?

#### Gibbs Sampler as Transition Matrix and State Diagram

We can represent the $p(X=X_{n+1}, Y=Y_{n+1}|X=X_n, Y=Y_n)$ as a $4\times 4$ matrix, $T$, where the $i,j$-th entry is the probability of starting with sample $i$ and generating sample $j$.

Alternatively, we can visualize how the Gibbs sampler moves around in the sample space $(X,Y)$ with a diagram.

Both transition matrix and state diagram describe show the Gibbs Sampler moves in the sample space as it's producing one sample after another.

#### Proof of Correctness - Limiting Distribution

We see that computing the probability of the next sample given the current sample $(X=1,Y=1)$
$$
p(X_{n+1},Y_{n+1}|X_{n}=1,Y_{n}=1)
$$
is equivalent to multiplying the vector $[ 1 \space\space 0 \space\space 0 \space\space 0]$ with the matrix $T$.

When we do, we get the distribution $[0.12 \space\space 0.48 \space\space 0.08 \space\space 0.32]$ over the next sample. But this distribution looks just like the joint distribution $p(X,Y)$.

This means that if we start at $(X=1,Y=1)$, the next sample the Gibbs sampler returns will be from the joint distribution.

In fact, you can start with any point in the samples space $(X,Y)$ or any distribution over the sample space, the next sample the Gibbs Sampler returns will be from the joint distribution. i.e. any vector times $T$ will return $[0.12 \space\space 0.48 \space\space 0.08 \space\space 0.32]$.

This proves the correctness of the Gibbs Sampler for $p(X,Y)$.

### 2. Definition of Properties of Markov Chains

#### Markov Chains in Discrete and Continuous Spaces

A **discrete-time stochastic process** is set of random variables $\{X_0, X_1, ...\}$ where each random variable takes value in $S$. The set $S$ is called the **state space** and can be continuous or finite.

A stochastic process satisfies the **Markov property** if $X_n$ depends only on $X_{n-1}$ (i.e. $X_n$ is independent of $X_1, ..., X_{n-2}$). A stochastic process that satisfies the Markov property is called a **Markov chain**.

We will assume that $p(X_n|X_{n-1})$ is the same for all $n$.

#### Transition Matrices and Kernels

The Markov property ensures that we can describe the dynamics of the entire chain by describing how the chain **transitions** from state $i$ to state $j$. 

If the state space is finite, then we can represent the transition from $X_{n-1}$ to $X_n$ as a **transition matrix** $T$, where $T_{ij}$ is the probability of the chain transitioning from state $i$ to $j$:
$$
T_{ij} = \mathbb{P}[X_n=j|X_{n-1}=i]
$$
The transition matrix can be represented visually as a **finite state diagram**.

If the state space is continuous, then we can represent the transition from $X_{n-1}$ to $X_n$ as **transition kernel pdf** $T(x,x')$, representing the likelihood of transitioning from state $X_{n-1}=x$ to state $X_n=x'$. The probability of transitioning into a region $A \subset S$ from state $x$ is given by
$$
\mathbb{P}[X_n\in A|X_{n-1}=x] = \int_A T(x,y)dy
$$
such that $\int_S T(x,y)dy=1$.

Transition matrix is useful because, once you know how a Markov chain moves once, you can describe its movement for any number of times, i.e., if you have the transition matrix, you will be able to deduce what is the probability of starting at a place $i$ and going to any state after $n$ transations.

#### Chapman-Kolmogorov Equations: Dynamics as Matrix Multiplication

#### Chapman-Kolmogorov Equations: Continuous State Space

#### Properties of Markov Chains: Irreducibility

A Markov chain is called **irreducible** if every state can be reached from every other state in finite time.

#### Properties of Markov Chains: Aperiodicity

A state $s \in S$ has period $t$ if one can only return to $s$ in multiples of $t$ steps. 

A Markov chain is called **aperiodic** if the period of each state is $1$.

#### Properties of Markov Chains: Stationary Distributions

The Markov Chains that are aperiodic and irreducible will have asymptotic behaviors that are very predictable. 

A distribution $\pi$ over the finite state space $S$ is a stationary distribution of the Markov chain with transition matrix $T$ if 
$$
\pi = \pi T
$$
i.e. performing the transition matrix does not change the distribution.

The equivalent condition for continuous state space $S$ is
$$
\pi(x) = \int_S T(y,x) \pi(y) dy
$$

#### Properties of Markov Chains: Limiting Distributions

We are often interested in what happens to a distribution after many transitions
$$
\pi^{(n)} = \pi^{(0)}T^{(n)}, \text{ or } \pi^{(n)}(x) = \int_S T^{(n)}(y,x)\pi^{(0)}(y) dy
$$
If $\pi^{\infty} = \lim_{n\rightarrow \infty} \pi^{(n)}$ exists (with some caveats in the continuous state case), it's called the limiting distribution of the Markov Chain. 

**Note**: A stationary distribution is an equilibrium of the Markov chain, whereas the limiting distribution can be thought of as an equilibrium that's achieved by starting at some specific distribution, meaning that a stationary distribution for Markov chain may exist, but there may be no way of reaching it starting off with an initial distribution. 

#### Fundamental Theorm of Markov Chains

**Fundatmental Theorem of Markov Chains**: if a Markov chain is irreducible and aperiodic, then it has a unique stationary distribution $\pi$ and $\pi^{\infty} = \lim_{n\rightarrow \infty} \pi^{(n)}=\pi$.

In practice, the theorem means you can start with any initial distribution over the state space $S$, asumptotically, you will always obtain the distribution $\pi$.

While we can't prove the theorem, we can indicate why both conditions are necessary.

#### Properties of Markov Chains: Reversibility

A Markov chain is called **reversible** with respect to a distribution $\pi$ over a finite state space $S$ if the following holds:
$$
\pi T = T \pi^T
$$
The above translates to $\pi_iT_{i,j} = \pi_j T_{j,i}$.

For a continuous state space, the condition is
$$
\pi(x)T(x,y) = T(y,x)\pi(y)
$$
The condition for reversibility is often called the **detailed balance** condition.

#### Reversibility and Stationary Distributions

Using reversibility, we have another way to characterize a stationary distribution.

**Theorem**: If a Markov chain, with transition matrix or kernel pdf $T$, is reversible with respect to $\pi$. Then $\pi$ is a stationary distribution of the chain.

**Proof**: we will give the proof for the case of a continuous state space $S$. Suppose that $\pi(x)T(x,y)=T(y,x)\pi(y)$, then
$$
\int_S \pi(x)T(x,y)dx = \int_S \pi(y)T(y,x)dx=\pi(y)\int_S T(y,x) dx = \pi(y) \cdot 1 = \pi(x)
$$
**Note**: Conditional revesibility is easier and more efficienct way to check whether $\pi$ is a stationary distribution for $T$, becasue integrals are not easy to compute.

### 3. Markov Chain Monte Carlo

#### Markov chain Monte Carlo Samplers

Every sampler for a distribution $p(\theta)$ over the domain $\Theta$ defines a stochastic process $\{X_0, X_1, ...,\}$, where the state space is $\Theta$.

If the sampler defines a Markov Chain whose unique stationary and limiting distribution is $p$, it's called a **Markov Chain Monte Carlo (MCMC)** sampler.

I.e., for every MCMC sampler, we have that

1. **Stationary**: $pT=p$
2. **Limiting**: $\lim_{n\rightarrow \infty} \pi^{(n)}=p$, for any $\pi^{(0)}$

where $T$ is the transition matrix or kernel pdf defined by the sampler.

#### What Do We Need to Prove to Get $pT=p$ And $\lim_{n\rightarrow \infty}\pi^{(n)}=p$

1. Prove that the sampler is **irreducible** and **aperiodic**. Then, there is a unique stationary distribution $\pi$ such that
   $$
   \pi T =\pi
   $$

2. Prove that the sampler is **reversibe** or **detailed balanced** with respect to $p$. Then,
   $$
   \pi = p
   $$

#### Gibbs as MCMC

We have seen example where teh Gibbs sampler for a discrete target distribution defines a MCMC sampler.

About Gibbs samplers for a continuous target distribution $p$? The samples $X_n$ obtained by the sampler defines a Markov chain: the distribution over the next samler depends only on the current sample.

To be a MCMC sampler, we need to prove that $p$ is the stationary and limiting distribution of the sampler.