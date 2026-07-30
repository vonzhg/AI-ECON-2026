please review these codes and see whether they implement the algorithm for the model as detailed in the document below properly. if not how to improve. Thank!
\section{Extension: Heterogeneous Agents Model}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
This section extends the Deep Ramsey Algorithm to a model with heterogeneous agents facing idiosyncratic employment risk. The key methodological innovations from the representative agent case---two-level fixed-point iteration, adaptive sampling, and boundary marking---carry over, with modifications to handle the higher-dimensional state space and occasionally binding borrowing constraints.
As in the representative agent model, we employ an \textbf{actor-critic architecture}: the policy network (actor) $\pi_\theta$ learns the optimal government policy, while the value network (critic) $V_\phi$ approximates the continuation value function. This architecture enables efficient gradient-based optimization of the Bellman equation.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\subsection{Economic Environment}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\subsubsection{State Variables}
The economy is characterized by the state vector:
\begin{equation}
    s = (K, a^e, a^u, c^e, c^u)
\end{equation}
where:
\begin{itemize}
    \item $K \in \mathbb{R}_+$: Aggregate capital stock (beginning of period).
    \item $a^e, a^u \in \mathbb{R}$: Individual asset holdings for employed and unemployed agents, representing claims on future consumption.
    \item $c^e, c^u \in \mathbb{R}_{++}$: Consumption levels, serving as co-state variables encoding past policy commitments (analogous to $\mu$ in the representative agent model, where $c = 1/\mu$).
\end{itemize}
\subsubsection{Transition Probabilities}
Employment status follows a two-state Markov process. Let $\pi^e$ and $\pi^u = 1 - \pi^e$ denote the stationary proportions of employed and unemployed agents, respectively. The transition matrix is:
\begin{equation}
    \Pi = \begin{pmatrix} \pi^{ee} & \pi^{eu} \\ \pi^{ue} & \pi^{uu} \end{pmatrix}
\end{equation}
where $\pi^{ij} = \Pr(\text{status } j \text{ next period} \mid \text{status } i \text{ this period})$.
The stationary distribution satisfies $\pi^e = \pi^{ue} / (\pi^{eu} + \pi^{ue})$ and $\pi^u = \pi^{eu} / (\pi^{eu} + \pi^{ue})$.
\subsubsection{Functional Forms}
\paragraph{Utility Function.}
The utility function is assumed to be separable between consumption and labor:
\begin{equation}
    u(c, n) = \frac{c^{1-\sigma}}{1-\sigma} - \frac{n^{1+\gamma}}{1+\gamma}
\end{equation}
where $\sigma$ is the coefficient of relative risk aversion (inverse of the intertemporal elasticity of substitution) and $\gamma$ is the inverse of the Frisch elasticity of labor supply.
\paragraph{Production Function.}
The production function is Cobb-Douglas:
\begin{equation}
    F(K, N) = K^{\alpha} N^{1-\alpha}
\end{equation}
where $\alpha$ is the capital share of output.
\paragraph{Parameter Values.}
The baseline parameters are set as follows:
\begin{center}
\begin{tabular}{lcl}
\hline
\textbf{Parameter} & \textbf{Value} & \textbf{Description} \\
\hline
$\alpha$ & $1/3$ & Capital share \\
$\sigma$ & $2$ & Coefficient of relative risk aversion \\
$\gamma$ & $2$ & Inverse Frisch elasticity \\
$\beta$ & $0.8$ & Time discount factor \\
$\delta$ & $0.1$ & Capital depreciation rate \\
$\pi^{ee}$ & $0.5$ & Probability of remaining employed \\
$\pi^{uu}$ & $0.5$ & Probability of remaining unemployed \\
\hline
\end{tabular}
\end{center}
\noindent The symmetric transition probabilities $\pi^{ee} = \pi^{uu} = 0.5$ imply $\pi^{eu} = \pi^{ue} = 0.5$ and a stationary distribution of $\pi^e = \pi^u = 0.5$.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\subsection{Ramsey Planner's Problem}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
The planner maximizes social welfare subject to implementability constraints ensuring the allocation constitutes a competitive equilibrium.
\subsubsection{Bellman Equation}
For $t > 0$, the value function satisfies:
\begin{equation}\label{eq:bellman_ha}
    V(s) = \max_{\{n^e, c'^e, c'^u\}} \left\{ \pi^e u(c^e, n^e) + \pi^u u(c^u, 0) + \beta V(s') \right\}
\end{equation}
where current-period welfare depends on consumption levels $(c^e, c^u)$ from the state, and the control variables $(n^e, c'^e, c'^u)$ determine the transition to the next state $s'$.
\subsubsection{Actor-Critic Architecture}
As in the representative agent model, we parameterize the solution using two neural networks:
\begin{enumerate}
    \item \textbf{Actor (Policy Network)} $\pi_\theta: s \mapsto (n^e, c'^e, c'^u)$\\
    Maps the current state to optimal control variables. The actor is trained to maximize the Bellman objective.
    \item \textbf{Critic (Value Network)} $V_\phi: s \mapsto \mathbb{R}$\\
    Approximates the continuation value $V(s')$. The critic is trained via temporal difference learning on simulated trajectories.
\end{enumerate}
This separation enables stable training: the critic provides a differentiable approximation to the continuation value, allowing gradient-based optimization of the actor without requiring full trajectory rollouts during each gradient step.
\subsubsection{Control Variables and Explicit Reductions}
We select $y = (n^e, c'^e, c'^u)$ as the policy outputs. This choice enables \textbf{explicit (non-iterative) transition dynamics} by using the following relationships to determine all other quantities:
\paragraph{Resource Constraint.}
Given $(n^e, c^e, c^u, K)$, next-period capital is determined:
\begin{equation}\label{eq:resource_ha}
    K' = F(K, n^e \pi^e) + (1-\delta)K - \pi^e c^e - \pi^u c^u
\end{equation}
\paragraph{Equilibrium Prices.}
From household optimality conditions:
\begin{align}
    \hat{w} &= (n^e)^\gamma (c^e)^\sigma \quad &\text{(After-tax wage)} \label{eq:wage_ha}\\
    Q &= \beta (c^e)^\sigma \left[ (c'^e)^{-\sigma} \pi^{ee} + (c'^u)^{-\sigma} \pi^{ue} \right] \quad &\text{(Bond price)} \label{eq:bondprice_ha}
\end{align}
\paragraph{Budget Constraints.}
Given prices, next-period assets are determined:
\begin{align}
    a'^e &= \frac{1}{Q} \left[ \frac{a^e \pi^e \pi^{ee} + a^u \pi^u \pi^{eu}}{\pi^e} + \hat{w} n^e - c^e \right] \label{eq:asset_e}\\
    a'^u &= \frac{1}{Q} \left[ \frac{a^e \pi^e \pi^{ue} + a^u \pi^u \pi^{uu}}{\pi^u} - c^u \right] \label{eq:asset_u}
\end{align}
These reductions eliminate $(K', \hat{w}, Q, a'^e, a'^u)$ as independent choice variables, leaving only $(n^e, c'^e, c'^u)$ as policy outputs.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\subsection{Implementability Constraints}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
The implementability constraints ensure the allocation can be supported as a competitive equilibrium. In this model, they take the form of occasionally binding borrowing constraints.
\paragraph{Borrowing Constraints.}
Agents face borrowing constraints:
\begin{equation}
    a'^e \geq 0, \quad a'^u \geq 0
\end{equation}
\paragraph{Euler Conditions.}
The Euler equations may hold with inequality when borrowing constraints bind. Define the Euler discrepancy:
\begin{equation}\label{eq:euler_discrepancy}
    \phi^i = Q (c^i)^{-\sigma} - \beta \left[ (c'^e)^{-\sigma} \pi^{ie} + (c'^u)^{-\sigma} \pi^{iu} \right], \quad i \in \{e, u\}
\end{equation}
\paragraph{Complementarity Conditions.}
The optimal allocation must satisfy:
\begin{equation}\label{eq:complementarity}
    a'^i \geq 0, \quad \phi^i \geq 0, \quad \phi^i \cdot a'^i = 0, \quad i \in \{e, u\}
\end{equation}
\paragraph{Fischer-Burmeister Formulation.}
To handle the complementarity conditions in gradient-based optimization, we use the smoothed \textbf{Fischer-Burmeister (FB) function}:
\begin{equation}\label{eq:fb_function}
    \Phi_\epsilon(a, b) = a + b - \sqrt{a^2 + b^2 + \epsilon^2}
\end{equation}
where $\epsilon > 0$ is a smoothing parameter. The condition $\Phi_\epsilon(a, b) = 0$ is equivalent to $(a \geq 0, b \geq 0, ab = 0)$ as $\epsilon \to 0$.
The FB penalty is incorporated into training:
\begin{equation}\label{eq:fb_penalty}
    \mathcal{L}_{\text{FB}} = \lambda_{\text{FB}}^{(k)} \sum_{i \in \{e, u\}} \Phi_\epsilon(\phi^i, a'^i)^2
\end{equation}
\paragraph{Augmented Lagrangian Schedule.}
The penalty weight $\lambda_{\text{FB}}^{(k)}$ increases during training:
\begin{equation}
    \lambda_{\text{FB}}^{(k+1)} = \begin{cases}
        \rho \cdot \lambda_{\text{FB}}^{(k)} & \text{if } \|\Phi_\epsilon\|_{\text{avg}} > \tau_{\text{FB}} \\
        \lambda_{\text{FB}}^{(k)} & \text{otherwise}
    \end{cases}
\end{equation}
where $\rho > 1$ is the growth factor (e.g., $\rho = 1.5$). This augmented Lagrangian approach starts with moderate penalty to allow exploration, then progressively enforces stricter feasibility.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\subsection{Unified Optimization Problem}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
We now present the complete optimization problem combining all elements. The key constraint is that states must lie in the \textbf{admissible set} $\Omega$, which is unknown a priori and must be learned jointly with the policy.
\paragraph{Actor-Critic Networks.}
\begin{enumerate}
    \item \textbf{Actor (Policy Network)} $\pi_\theta: s \mapsto (n^e, c'^e, c'^u)$
    A feedforward network with ReLU activations. Outputs are transformed to ensure positivity and appropriate ranges:
    \begin{align}
        n^e &= \sigma(\ell_n) \cdot (n_{\max} - n_{\min}) + n_{\min} \\
        c'^i &= \exp(\ell_{c^i}) \cdot c_{\text{scale}}, \quad i \in \{e, u\}
    \end{align}
    where $\sigma$ is the sigmoid function and $\ell$ denotes raw logits. The exponential transformation ensures $c'^i > 0$.
    \item \textbf{Critic (Value Network)} $V_\phi: s \mapsto \mathbb{R}$
    A separate feedforward network approximating the planner's continuation value. The critic provides the value targets for the actor's gradient computation.
\end{enumerate}
\paragraph{Complete Forward Pass and Actor-Critic Training.}
\begin{center}
\fbox{\parbox{0.95\textwidth}{
\textbf{Unified Ramsey Optimization (Heterogeneous Agents)}
\vspace{0.5em}
\textit{Input (State):} $s = (K, a^e, a^u, c^e, c^u) \in \Omega$ \quad (admissible) \\
\textit{Actor Output (Policy):} $(n^e, c'^e, c'^u) = \pi_\theta(s)$ \\
\textit{Critic Output (Value):} $V_\phi(s')$ \quad (continuation value estimate)
\vspace{0.5em}
\textbf{1. Explicit Transition (No Iteration Required):}
\begin{align}
    K' &= K^\alpha (n^e \pi^e)^{1-\alpha} + (1-\delta)K - c^e \pi^e - c^u \pi^u \tag{Capital}\\[4pt]
    \hat{w} &= (n^e)^\gamma (c^e)^\sigma \tag{Wage}\\[4pt]
    Q &= \beta (c^e)^\sigma \left[ (c'^e)^{-\sigma} \pi^{ee} + (c'^u)^{-\sigma} \pi^{ue} \right] \tag{Bond Price}\\[4pt]
    a'^e &= \frac{1}{Q} \left[ \frac{a^e \pi^e \pi^{ee} + a^u \pi^u \pi^{eu}}{\pi^e} + \hat{w} n^e - c^e \right] \tag{Asset: Employed}\\[4pt]
    a'^u &= \frac{1}{Q} \left[ \frac{a^e \pi^e \pi^{ue} + a^u \pi^u \pi^{uu}}{\pi^u} - c^u \right] \tag{Asset: Unemployed}
\end{align}
\vspace{0.3em}
\textbf{2. Euler Discrepancies:}
\begin{align}
    \phi^e &= Q (c^e)^{-\sigma} - \beta \left[ (c'^e)^{-\sigma} \pi^{ee} + (c'^u)^{-\sigma} \pi^{ue} \right] \\
    \phi^u &= Q (c^u)^{-\sigma} - \beta \left[ (c'^e)^{-\sigma} \pi^{eu} + (c'^u)^{-\sigma} \pi^{uu} \right]
\end{align}
\vspace{0.3em}
\textbf{3. Fischer-Burmeister Residuals:}
\begin{align}
    \Phi^e &= \phi^e + a'^e - \sqrt{(\phi^e)^2 + (a'^e)^2 + \epsilon^2} \\
    \Phi^u &= \phi^u + a'^u - \sqrt{(\phi^u)^2 + (a'^u)^2 + \epsilon^2}
\end{align}
\vspace{0.3em}
\textbf{4. Current Period Welfare:}
\begin{equation}
    U(s, n^e) = \pi^e \left[ \frac{(c^e)^{1-\sigma}}{1-\sigma} - \frac{(n^e)^{1+\gamma}}{1+\gamma} \right] + \pi^u \frac{(c^u)^{1-\sigma}}{1-\sigma}
\end{equation}
\vspace{0.3em}
\textbf{5. Actor Loss (Policy Gradient with $T$-Step Rollout):}
\begin{equation}
    \mathcal{L}_{\text{actor}} = -\underbrace{\left[ \sum_{t=0}^{T} \beta^t U_t + \beta^{T+1} V_\phi(s_{T+1}) \right]}_{\text{$T$-step Bellman Value (maximize)}} + \underbrace{\lambda_{\text{FB}}^{(k)} \sum_{t=0}^{T} \left[ (\Phi^e_t)^2 + (\Phi^u_t)^2 \right]}_{\text{FB Penalty (complementarity)}}
\end{equation}
\vspace{0.3em}
\textbf{6. Critic Loss (Temporal Difference with $T$-Step Rollout):}
\begin{equation}
    \mathcal{L}_{\text{critic}} = \left( V_\phi(s) - V_{\text{target}}(s) \right)^2, \quad V_{\text{target}} = \sum_{t=0}^{T} \beta^t U_t + \beta^{T+1} V_\phi(s_{T+1})
\end{equation}
\vspace{0.3em}
\textbf{7. Admissibility Requirement:}
\begin{equation}
    s_t \in \Omega \quad \text{for all } t = 0, 1, \ldots, T+1
\end{equation}
The admissible set $\Omega$ is unknown and learned via the $\alpha$-shape procedure in Section~\ref{sec:alpha_shape_training}.
}}
\end{center}
\vspace{0.5em}
\textit{Key Properties:}
\begin{itemize}
    \item All quantities in the forward pass are computed via explicit formulas---no iterative solvers are required.
    \item The actor-critic separation enables stable training: the actor optimizes policy via gradient ascent on the $T$-step Bellman objective, while the critic provides differentiable value estimates.
    \item The Fischer-Burmeister penalty handles complementarity conditions (occasionally binding borrowing constraints) in a differentiable manner.
    \item Efficient batched computation and end-to-end gradient flow through the entire $T$-step rollout.
\end{itemize}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\subsection{Admissibility and Boundary Learning in High Dimensions}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
The admissible set $\Omega$ is the set of states from which a competitive equilibrium can be sustained. As in the representative agent model, $\Omega$ is unknown a priori and depends on the policy---creating a two-level fixed-point problem.
\paragraph{Simplifying Assumption: Exogenous Consumption Bounds.}
To make the boundary learning tractable, we assume exogenous bounds on consumption:
\begin{equation}
    c^e, c^u \in [c_{\min}, c_{\max}]
\end{equation}
These bounds are set based on economic reasoning (e.g., subsistence consumption below, resource constraints above) and remain fixed throughout training.
With this assumption, the \textbf{endogenous admissible set} reduces to three dimensions:
\begin{equation}
    \Omega_{K,a} = \{(K, a^e, a^u) : \text{feasible equilibrium exists for some } (c^e, c^u) \in [c_{\min}, c_{\max}]^2\}
\end{equation}
\paragraph{Admissibility Score Components.}
The score $\mathcal{A}(s)$ has three components, using the power barrier function from Section 3.2:
\begin{enumerate}
    \item \textbf{Capital Feasibility ($A_K$):}
    \begin{equation}
        A_K = \mathcal{S}(K'; [K_{\min}, K_{\max}], \delta_K)
    \end{equation}
    Ensures next-period capital lies within economically meaningful bounds.
    \item \textbf{Asset Feasibility ($A_a$):}
    \begin{equation}
        A_a = \min_{i \in \{e, u\}} \mathcal{S}(a'^i; [0, K'], \delta_a)
    \end{equation}
    Checks that derived assets satisfy borrowing constraints and do not exceed aggregate capital.
    \item \textbf{Bond Price Feasibility ($A_Q$):}
    \begin{equation}
        A_Q = \mathcal{S}(Q; [0, \beta], \delta_Q)
    \end{equation}
    Ensures the equilibrium bond price lies in the economically valid range $[0, \beta]$. A price $Q > \beta$ would imply negative real interest rates below the discount rate; $Q < 0$ is economically meaningless.
\end{enumerate}
\textbf{Global Score:}
\begin{equation}
    \mathcal{A}(s) = w_K A_K + w_a A_a + w_Q A_Q
\end{equation}
These component scores provide continuous values in $[0, 1]$ based on proximity to constraint boundaries, enabling smooth gradient signals during training.
\paragraph{The Boundary Learning Problem.}
Given a set of sampled states $\{s_i\}_{i=1}^N$ with computed admissibility scores $\{\mathcal{A}(s_i)\}$, we partition into admissible and inadmissible sets based on thresholds. The goal is to learn the boundary of the admissible region $\Omega_{K,a}$ in the three-dimensional space $(K, a^e, a^u)$.
\textit{Challenge: High-Dimensional Boundary Representation.}
In the representative agent model, we estimated boundaries via binning in 2D: $B_{\min}(\mu, g), B_{\max}(\mu, g)$. This approach does not scale well to higher dimensions due to the curse of dimensionality---with 20 bins per dimension, 3D would require $20^3 = 8000$ bins, many of which would be sparsely populated.
\textit{Solution: $\alpha$-Convex Hull.}
We adopt the \textbf{$\alpha$-shape} (also called $\alpha$-convex hull) to represent the admissible set boundary. This geometric approach directly represents the boundary as a simplicial complex, handles non-convex regions (unlike the standard convex hull), and provides efficient membership testing. See Appendix~\ref{app:alpha_shapes} for detailed definitions and properties.
\paragraph{Binary Scoring for Boundary Membership.}
Unlike the component scores $A_K, A_a, A_Q$ which are continuous, the $\alpha$-shape provides a \textbf{binary classification}:
\begin{equation}
    \mathbf{1}_{\Omega}(K', a'^e, a'^u) = \begin{cases}
        1 & \text{if } (K', a'^e, a'^u) \in \mathcal{A}_\alpha \\
        0 & \text{otherwise}
    \end{cases}
\end{equation}
where $\mathcal{A}_\alpha$ is the $\alpha$-shape constructed from admissible points.
Computing a continuous score based on distance to the $\alpha$-shape boundary would be computationally expensive, requiring nearest-boundary-point queries for each sample. The binary indicator is both efficient and sufficient for the sampling procedure: points inside $\mathcal{A}_\alpha$ are candidates for training; points outside are excluded or assigned to $\mathcal{S}_{\text{fail}}$.
\paragraph{Implementation: $\alpha$-Shape Construction and Membership Testing.}
Given sample points with admissibility scores from the previous iteration, we construct and query the $\alpha$-shape using the \texttt{scipy.spatial} module:
\begin{enumerate}
    \item \textbf{Extract admissible points:} Filter points where $\mathcal{A}(s) > \tau_{\text{high}}$:
    \begin{equation}
        \mathcal{P}_{\text{adm}} = \{(K'_i, a'^e_i, a'^u_i) : \mathcal{A}(s_i) > \tau_{\text{high}}\}
    \end{equation}
    \item \textbf{Compute Delaunay triangulation:} Use \texttt{scipy.spatial.Delaunay(points)} to compute the Delaunay triangulation of $\mathcal{P}_{\text{adm}}$. This partitions the convex hull into tetrahedra.
    \item \textbf{Filter to $\alpha$-complex:} Retain only simplices with circumradius $\leq 1/\alpha$. The \texttt{alphashape} package provides \texttt{alphashape.alphashape(points, alpha)} for direct construction.
    \item \textbf{Batch membership testing:} For a batch of query points $\{x_j\}_{j=1}^M$, use \texttt{Delaunay.find\_simplex(query\_points)} which returns the index of the containing simplex (or $-1$ if outside). This operates in $O(M \log N)$ time via spatial indexing:
    \begin{verbatim}
    from scipy.spatial import Delaunay
    tri = Delaunay(admissible_points)
    inside = tri.find_simplex(query_points) >= 0  # Boolean array
    \end{verbatim}
\end{enumerate}
For tighter $\alpha$-shapes (excluding large circumradius tetrahedra), additional filtering is required after \texttt{find\_simplex} to check if the containing simplex belongs to $\mathcal{C}_\alpha$.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\subsection{Actor-Critic Training with $\alpha$-Shape Boundary Learning}\label{sec:alpha_shape_training}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
The $\alpha$-shape boundary learning integrates into the nested fixed-point iteration, with the actor-critic architecture enabling efficient optimization at each level. This subsection describes the complete training procedure.
\paragraph{Level 1 (Inner Fixed Point): Domain Discovery.}
For fixed actor $\pi_\theta$ and critic $V_\phi$, repeat for $K_{\text{inner}}$ iterations to stabilize the boundary estimate:
\begin{enumerate}
    \item Sample candidate states from current domain estimate
    \item Compute forward pass using actor: $s \stackrel{\pi_\theta}{\mapsto} (n^e, c'^e, c'^u) \mapsto (K', a'^e, a'^u, Q, \ldots)$
    \item Compute component scores $\mathcal{A}(s)$ based on transition feasibility
    \item Identify admissible points: $\mathcal{P}_{\text{adm}}^{(n)} = \{(K', a'^e, a'^u) : \mathcal{A}(s) > \tau_{\text{high}}\}$
    \item Update $\alpha$-shape: $\mathcal{A}_\alpha^{(n+1)} \gets \text{AlphaShape}(\mathcal{P}_{\text{adm}}^{(n)}, \alpha)$
\end{enumerate}
\paragraph{Level 2 (Outer Fixed Point): Actor-Critic Training.}
For $k = 1, \ldots, K$ total iterations:
\begin{enumerate}
    \item \textbf{Actor Training:} Update $\pi_\theta$ on states sampled from current $\mathcal{A}_\alpha$ via policy gradient on $T$-step rollouts:
    \begin{equation}
        \theta \leftarrow \theta + \eta_\theta \nabla_\theta \left[ \sum_{t=0}^{T} \beta^t U_t + \beta^{T+1} V_\phi(s_{T+1}) - \lambda_{\text{FB}}^{(k)} \sum_{t=0}^{T} \left( (\Phi^e_t)^2 + (\Phi^u_t)^2 \right) \right]
    \end{equation}
    \item \textbf{Critic Training:} Update $V_\phi$ via temporal difference learning:
    \begin{equation}
        \phi \leftarrow \phi - \eta_\phi \nabla_\phi \left( V_\phi(s) - V_{\text{target}}(s) \right)^2
    \end{equation}
    where $V_{\text{target}} = \sum_{t=0}^{T} \beta^t U_t + \beta^{T+1} V_\phi(s_{T+1})$.
    \item \textbf{Domain Refinement:} Every $K_{\text{outer}}$ iterations, run the inner loop ($K_{\text{inner}}$ iterations) to update $\mathcal{A}_\alpha$ based on the improved policy.
\end{enumerate}
As the actor improves, it may discover feasible equilibria in previously unexplored regions, causing the $\alpha$-shape to expand---the same discovery mechanism as in the representative agent model.
\paragraph{Modified Sampling Procedure.}
During adaptive sampling, candidates are filtered using both the continuous component scores and the binary $\alpha$-shape membership:
\begin{equation}
    \mathcal{S}_{\text{safe}} = \{s : \mathcal{A}(s) > \tau_{\text{high}} \text{ and } \textsc{IsAdmissible}(K'(s), a'^e(s), a'^u(s)) = 1\}
\end{equation}
\paragraph{Convergence Interpretation.}
The algorithm converges when both fixed-points stabilize:
\begin{itemize}
    \item \textbf{Inner:} The $\alpha$-shape $\mathcal{A}_\alpha$ is consistent with the admissibility scores under the current policy.
    \item \textbf{Outer:} The actor $\pi_\theta$ is optimal over $\mathcal{A}_\alpha$, and $\mathcal{A}_\alpha$ is the true feasible set under $\pi_\theta$.
\end{itemize}
At convergence, the algorithm has simultaneously discovered the endogenous feasible set and the optimal policy over that set.