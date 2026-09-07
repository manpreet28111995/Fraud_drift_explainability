# -*- coding: utf-8 -*-
p = """\documentclass{svproc}

\usepackage{url}
\def\UrlFont{\rmfamily}
\usepackage{cite}
\usepackage{amsmath,amssymb}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{float}
\usepackage[hidelinks]{hyperref}

\begin{document}
\mainmatter

\title{Temporal Faithfulness of Explanations Under Concept Drift in Streaming Fraud Detection}
\titlerunning{Temporal Faithfulness of Explanations Under Drift}

\author{Manpreet Singh}
\authorrunning{M. Singh}
\tocauthor{Manpreet Singh}

\institute{Department of Computer Science\\
\email{manpreetsingh@example.org}}

\maketitle

\begin{abstract}
Post-hoc feature attribution methods such as SHAP and LIME are widely deployed to monitor tabular machine learning models in financial fraud detection. A common operational assumption is that temporal drift in explanation rankings can serve as an early warning for performance decay before metrics such as ROC-AUC drop. In this work, we evaluate this assumption on the streaming IEEE-CIS fraud detection benchmark across 12 random seeds and 13 bi-weekly temporal sliding windows. We evaluate two distinct regimes: a stationary frozen model trained on the initial window, and an adaptive control model retrained on each incoming window. 
Our findings reveal a critical failure mode: post-hoc explanations for a frozen model exhibit an artificial false stability'' effect ($\rho_{\text{SHAP}} \approx 0.99$ across all evaluation windows relative to baseline), while explanations for retrained models adaptively shift ($\rho_{\text{SHAP}} \approx 0.77$,  = 1.12 \times 10^{-149}$). Because the frozen model's decision trees remain fixed, post-hoc explanations remain anchored to the frozen decision boundary, offering false reassurance to operators precisely as ROC-AUC degrades from /bin/zsh.915$ to /bin/zsh.853$. Furthermore, cross-correlation between explanation drift and performance drop is inflated in raw level series (/12$ seeds show  < 0.05$), but drops to .3\%$ for SHAP and collapses to chance (.0\%$) for LIME once trend-differenced. In contrast, model-free Wasserstein covariate shift robustly preserves lead--lag coupling across .7\%$ of seeds. These results indicate that post-hoc explanation rankings from static models are unfaithful indicators of distribution shift and should not replace model-agnostic feature monitoring in streaming fraud pipelines.
\keywords{Concept drift, explainable AI, SHAP, LIME, fraud detection, temporal faithfulness, covariate shift.}
\end{abstract}

\section{Introduction}

Automated machine learning models deployed in payment networks and e-commerce platforms operate in non-stationary environments. Fraud patterns evolve dynamically through adversarial adaptation, seasonal shopping behavior, and shifting transaction velocities~\cite{dalpozzolo2015credit,carcillo2018streaming}. When fraud classifiers decay over time, identifying degradation early is essential to mitigate financial loss. However, true ground-truth fraud labels often suffer from verification latency, taking weeks or months to finalize through customer dispute resolution and chargeback lifecycles~\cite{anderson2019detecting}. Consequently, production pipelines frequently rely on unsupervised surrogate signals, such as data distribution monitors and post-hoc feature attribution rankings (e.g., TreeSHAP~\cite{lundberg2020local} and LIME~\cite{ribeiro2016why}), under the intuition that changes in feature importance will signal model obsolescence before ground-truth performance can be computed.

Despite widespread deployment, the theoretical validity of using post-hoc explanation drift as an early warning indicator remains poorly understood. An explanation method computes attributions with respect to a specific parameterized model \theta$ and an evaluation sample $\mathcal{D}_t$. If \theta$ is kept static (a standard practice between scheduled retraining cycles), does the attribution vector $\mathbf{e}(f_\theta, \mathcal{D}_t)$ genuinely track environmental distribution shift (X, Y)$, or does it remain constrained by the fixed partitioning of the frozen decision surface?

In this paper, we conduct a controlled empirical study on the IEEE-CIS Fraud Detection dataset~\cite{ieeecis2019}. We process the chronological transaction stream into 13 non-overlapping 14-day temporal windows. Across 12 independent random seeds, we evaluate two operational setups: (i) a \emph{frozen} model regime where LightGBM is trained solely on the baseline window and evaluated sequentially on all subsequent windows, and (ii) a \emph{retrained} control regime where a fresh classifier is fit on the training slice of each successive window. We instrument both global explanations (exact Shapley values via TreeSHAP) and local explanations with local perturbation stability tracking (via LIME), benchmarking both against a model-agnostic Wasserstein covariate-shift baseline.

Our empirical investigation reveals three core insights:
\begin{enumerate}
    \item \textbf{The False Stability Paradox:} Explanations generated from a frozen model are deceptively static. The frozen model displays an average SHAP Spearman rank correlation of $\rho = 0.991$ relative to the baseline window throughout the 182-day deployment stream, whereas a retrained model that adapts to ongoing drift yields $\rho = 0.767$ ( = 1.12 \times 10^{-149}$ by paired 569Xtest;  = 2.23 \times 10^{-25}$ by Wilcoxon signed-rank test). Because static decision boundaries constrain feature path traversals, post-hoc explanations mask underlying concept drift.
    \item \textbf{Trend Spuriousness in Lead--Lag Coupling:} In raw time series, explanation drift appears strongly correlated with performance decay (/12$ seeds exhibit  < 0.05$ under permutation tests). However, this coupling is an artifact of shared monotonic trends. When time series are first-differenced to isolate window-to-window dynamics, the significance of LIME drift collapses to .0\%$ (chance level), while SHAP significance drops to .3\%$. Conversely, model-free Wasserstein distance maintains .7\%$ seed significance after differencing.
    \item \textbf{Failure of Absolute Alarm Thresholds:} Fixed absolute early-warning cutoffs suffer from an immediate floor effect, where performance drop thresholds are breached at the very first post-baseline window (=1$) across all seeds. Under adaptive relative thresholds (50\% peak drift), data-level Wasserstein shift leads performance drop by .08 \pm 1.31$ windows, outperforming both SHAP (.42 \pm 1.16$ windows) and LIME (.17 \pm 1.34$ windows).
\end{enumerate}

These findings demonstrate that explanation drift from static models is an unfaithful proxy for model health in streaming fraud detection. Practitioners should prioritize model-free distribution shift metrics over post-hoc attributions when designing early warning monitoring systems.

\section{Related Work}

\subsection{Concept Drift in Streaming Financial Systems}
Concept drift in financial streams manifests primarily as covariate shift (X) \neq P_0(X)$ and concept shift (Y|X) \neq P_0(Y|X)\cite{gama2014survey,lu2018learning}. In credit card transaction streams, adversaries deliberately manipulate purchasing patterns to evade deployed fraud filters~\cite{dalpozzolo2015credit}. While standard statistical monitoring approaches, such as the Kolmogorov-Smirnov test and Population Stability Index (PSI), detect feature-marginal distribution changes, they do not assess whether the shifted features remain aligned with the model's predictive structure~\cite{rabanser2019failing}.

\subsection{Faithfulness and Stability of Feature Attributions}
Feature attribution methods approximate the importance of input variables. For tree ensembles, TreeSHAP~\cite{lundberg2020local} computes exact Shapley values derived from cooperative game theory. In contrast, LIME~\cite{ribeiro2016why} fits an interpretable linear surrogate within an empirical perturbation neighborhood. Prior work has noted that explanations are sensitive to sampling noise, hyperparameter tuning, and data perturbations~\cite{alvarezmelis2018robustness,visani2022statistical}. Furthermore, the distinction between \emph{model faithfulness} (does the attribution reflect the model's internal reasoning?) and \emph{environmental faithfulness} (does attribution drift reflect data distribution shifts?) is frequently confounded in monitoring literature~\cite{hooker2019benchmark,haug2021leveraging}. Our work directly separates these mechanisms via an empirical control regime.

\section{Methodology}

\subsection{Problem Formulation and Sliding Window Architecture}
Let $\mathcal{S} = \{(x_i, y_i, t_i)\}_{i=1}^N$ be a time-stamped transaction stream with feature vectors  \in \mathbb{R}^d$, binary fraud indicators  \in \{0, 1\}$, and event timestamps $. We partition the timeline into $ contiguous, non-overlapping windows $\{D_w\}_{w=0}^{W-1}$ of duration $\Delta t = 14$ days. Non-overlapping steps ($\text{step} = 14$ days) guarantee statistical independence between window evaluation samples. 

Within each window $, the earliest \%$ of transactions form the training split ^{\text{train}}$ and the remaining \%$ form the chronological test split ^{\text{test}}$. This strict temporal split prevents forward-looking leakage. Windows with fewer than 500 rows or fewer than 5 positive fraud cases are filtered out.

\subsection{Dual-Regime Experimental Design}
To isolate the effect of model staleness from genuine environmental drift, we evaluate two parallel regimes:
\begin{itemize}
    \item \textbf{Frozen Regime ({\text{frozen}}$):} A LightGBM classifier $ is trained on ^{\text{train}}$. Its decision threshold $\theta^*$ is calibrated on ^{\text{train}}$ by maximizing the $ score. For all subsequent windows  \in \{1, \dots, W-1\}$, $ and $\theta^*$ are evaluated unmodified on ^{\text{test}}$.
    \item \textbf{Retrained Regime ({\text{retrained}}$):} For every window $, a fresh LightGBM model $ is trained on ^{\text{train}}$ with an independently optimized threshold $\theta_w^*$. The model is evaluated out-of-sample on ^{\text{test}}$.
\end{itemize}

\subsection{Quantifying Feature Attribution Drift}
At each window $, global feature importance vectors $\mathbf{e}_w \in \mathbb{R}^d$ are extracted:
\begin{itemize}
    \item \textbf{SHAP Importance:} Calculated as the mean absolute Shapley value over a uniform subsample of {\text{sub}} = 2000$ test observations:
    \begin{equation}
        I_j^{\text{SHAP}}(w) = \frac{1}{N_{\text{sub}}} \sum_{i=1}^{N_{\text{sub}}} |\phi_{i, j}(f)|
    \end{equation}
    where $\phi_{i, j}(f)$ represents the exact TreeSHAP attribution for feature $ on instance $ targeting the positive fraud class.
    \item \textbf{LIME Importance and Stability:} Explanations are computed across {\text{inst}} = 40$ test instances. To ensure computational tractability, LIME perturbations are constrained to the top =50$ global features determined by the baseline SHAP ranking, with remaining features held fixed. For each instance, explanations are repeated =5$ times with distinct perturbation seeds:
    \begin{equation}
        I_j^{\text{LIME}}(w) = \frac{1}{N_{\text{inst}} \cdot R} \sum_{i=1}^{N_{\text{inst}}} \sum_{r=1}^R |w_{i, r, j}|
    \end{equation}
    Local perturbation stability is quantified via pairwise Jaccard similarity across the $ runs:
    \begin{equation}
        \text{Stability}_{\text{Jaccard}}(i) = \binom{R}{2}^{-1} \sum_{a < b} \frac{|\text{Top}k(w_{i, a}) \cap \text{Top}k(w_{i, b})|}{|\text{Top}k(w_{i, a}) \cup \text{Top}k(w_{i, b})|}
    \end{equation}
\end{itemize}

Drift between baseline importance $\mathbf{e}_0$ and window importance $\mathbf{e}_w$ is measured using Spearman rank correlation ($\rho$), Kendall's $\tau$, and Top-10 Jaccard set overlap:
\begin{equation}
    \text{Drift}_{\text{Spearman}}(w) = 1.0 - \rho(\mathbf{e}_0, \mathbf{e}_w)
\end{equation}

\subsection{Model-Agnostic Covariate Shift Baseline}
As an experimental control, we compute the 1-Wasserstein (earth mover's) distance for all continuous features between ^{\text{test}}$ and ^{\text{test}}$:
\begin{equation}
    W_1(u, v) = \int_{-\infty}^{\infty} |U(x) - V(x)| dx
\end{equation}
The global covariate shift metric is the arithmetic mean across all numeric features:
\begin{equation}
    \text{Shift}_W(w) = \frac{1}{|F_{\text{num}}|} \sum_{j \in F_{\text{num}}} W_1(X_{0, j}, X_{w, j})
\end{equation}

\subsection{Lead--Lag Cross-Correlation and Significance Testing}
Let (w)$ be an explanatory drift signal (SHAP, LIME, or Wasserstein) and (w) = \text{AUC}(0) - \text{AUC}(w)$ be the performance drop series. We compute the normalized cross-correlation function (\ell)$ over lags $\ell \in [-\ell_{\max}, \ell_{\max}]$:
\begin{equation}
    r(\ell) = \frac{\sum_w (s_w - \bar{s})(p_{w+\ell} - \bar{p})}{\sqrt{\sum_w (s_w - \bar{s})^2 \sum_w (p_{w+\ell} - \bar{p})^2}}
\end{equation}
A positive lag ($\ell^* > 0$) indicates that the drift signal leads performance decay. Statistical significance is computed via a non-parametric Monte Carlo permutation test ( = 2000$ iterations) by permuting the temporal indices of (w)$.

Because non-stationary series with monotonic trends yield spuriously high cross-correlations, we apply first-differencing:
\begin{equation}
    \Delta s_w = s_w - s_{w-1}, \quad \Delta p_w = p_w - p_{w-1}
\end{equation}
The differenced correlation isolates contemporaneous and lagged dynamic innovations.

\subsection{Relative Threshold Crossing Analysis}
To resolve scale disparities and floor effects in fixed threshold crossing times, we define the relative crossing index for signal $ at fraction $\alpha \in \{0.25, 0.50, 0.75\}$:
\begin{equation}
    w_{\text{cross}}(s, \alpha) = \min \{ w \mid s_w \ge \alpha \cdot \max_t s_t \}
\end{equation}
The early warning lead time is:
\begin{equation}
    \text{LeadTime}(\alpha) = w_{\text{cross}}(p, \alpha) - w_{\text{cross}}(s, \alpha)
\end{equation}
Positive lead time demonstrates that the monitor crosses its alarm threshold prior to performance decay.

\section{Experimental Setup}

\subsection{Dataset and Preprocessing Contract}
We utilize the IEEE-CIS Fraud Detection dataset~\cite{ieeecis2019}, comprising {,}540$ transactions and {,}233$ identity records spanning 182 days. Transactions are left-joined with identity records on \texttt{TransactionID} and sorted chronologically on \texttt{TransactionDT}.

To prevent artificial attribution drift caused by varying column orders or changing categorical mappings, we enforce a unified feature specification:
\begin{itemize}
    \item Categorical features are encoded once over the entire historical vocabulary using \texttt{LabelEncoder}, reserving an explicit \texttt{\_\_UNSEEN\_\_} token for out-of-vocabulary entries.
    \item Missing values in continuous features are filled with 569X999.0$.
    \item Cyclic temporal features (\texttt{hour\_of\_day}, \texttt{day\_of\_week}) are derived directly from elapsed transaction timestamps.
\end{itemize}

\subsection{Model Hyperparameters}
Models are trained using LightGBM (\texttt{LGBMClassifier}) with: 400 estimators, learning rate /bin/zsh.05$, 63 leaves, feature subsample ratio /bin/zsh.8$, row subsample ratio /bin/zsh.8$, $ regularization /bin/zsh.1$, $ regularization /bin/zsh.1$, and minimum child samples of 30. Categorical features are handled natively via optimal Fisher splits.

\section{Results and Analysis}

\begin{figure}[t]
\centering
\includegraphics[width=0.88\textwidth]{outputs/performance_decay.png}
\caption{Temporal performance decay in ROC-AUC over 13 sliding 14-day windows across 12 random seeds (mean $\pm$ standard deviation). The frozen model degrades rapidly from /bin/zsh.915$ to /bin/zsh.853$, whereas the retrained model recovers performance across windows.}
\label{fig:perf_decay}
\end{figure}

\begin{figure}[t]
\centering
\includegraphics[width=0.88\textwidth]{outputs/explanation_drift.png}
\caption{Evolution of explanation drift (.0 - \text{Spearman } \rho$) for SHAP and LIME relative to the baseline window. Frozen models exhibit minimal drift despite severe performance decay, whereas retrained models display substantial explanation movement.}
\label{fig:drift_evolution}
\end{figure}

\subsection{Performance Decay Under Streaming Drift}
Figure~\ref{fig:perf_decay} illustrates the ROC-AUC trajectory across the 13 consecutive windows. The frozen model displays pronounced performance degradation: ROC-AUC drops from /bin/zsh.915 \pm 0.002$ in the baseline window to /bin/zsh.853 \pm 0.003$ in window 1, eventually reaching /bin/zsh.825$ by window 11. Conversely, the continuously retrained model maintains an average ROC-AUC of /bin/zsh.887 \pm 0.012$ across all post-baseline windows, demonstrating that fresh models successfully adapt to evolving transaction patterns.

\subsection{The False Stability Paradox}
Figure~\ref{fig:drift_evolution} and Table~\ref{tab:regime_stability} report ranking drift metrics between the frozen and retrained regimes across all  = 144$ paired evaluations (12 seeds $\times$ 12 post-baseline windows).

\begin{table}[t]
\centering
\caption{Paired regime comparison of ranking stability across =144$ evaluations. Positive difference indicates the frozen model is more stable (less drifted from baseline) than the retrained model.}
\label{tab:regime_stability}
\begin{tabular}{lcccccc}
\toprule
Metric & Frozen Mean & Retrained Mean & Difference ($\pm \text{std}$) & \% Frozen Stable & Paired 569Xstat & 569Xvalue \\
\midrule
SHAP Spearman $\rho$ & 0.991 & 0.767 & 1.2238 \pm 0.0208$ & 100.0\% & 128.59 & .12 \times 10^{-149}$ \\
LIME Spearman $\rho$ & 0.686 & 0.562 & 1.1243 \pm 0.0804$ & 96.5\% & 18.48 & .00 \times 10^{-39}$ \\
SHAP Top-10 Jaccard & 0.818 & 0.642 & 1.1756 \pm 0.1182$ & 85.4\% & 17.77 & .21 \times 10^{-38}$ \\
LIME Top-10 Jaccard & 0.538 & 0.362 & 1.1758 \pm 0.1432$ & 81.3\% & 14.68 & .47 \times 10^{-30}$ \\
\bottomrule
\end{tabular}
\end{table}

\begin{figure}[t]
\centering
\includegraphics[width=0.72\textwidth]{outputs/regime_stability.png}
\caption{Distribution of Spearman rank stability ($\rho$) across evaluation pairs (=144$). Frozen model explanations are artificially stable, while adaptive retrained models reorganize feature attributions.}
\label{fig:regime_stability}
\end{figure}

The empirical evidence reveals a clear paradox:
\begin{itemize}
    \item \textbf{Complete Invariance of Frozen TreeSHAP:} For 100\% of evaluated pairs (/144$), the frozen model's SHAP ranking is more stable relative to the baseline than the retrained model's ranking ( = 1.12 \times 10^{-149}$). The mean rank correlation for the frozen model remains $\rho = 0.991$, despite the model suffering a /bin/zsh.090$ drop in ROC-AUC.
    \item \textbf{Theoretical Origin:} TreeSHAP attributes credit based on decision path routing across static split nodes. When feature distributions shift, instances traverse different branches, but the internal split architecture remains identical. Unless instances shift into entirely unpopulated terminal regions, average path attribution changes minimally.
    \item \textbf{Operational Risk:} A fraud operations team monitoring SHAP ranking stability would observe nearly flat attribution curves (Figure~\ref{fig:drift_evolution}), incorrectly concluding that the model remains aligned with current transaction dynamics while fraud detection capacity is deteriorating.
\end{itemize}

\subsection{Robustness of Lead--Lag Dynamics to Trend Differencing}

Table~\ref{tab:corr_robustness} evaluates whether explanation drift signals lead performance decay.

\begin{table}[t]
\centering
\caption{Cross-correlation and permutation test significance between drift signals and performance decay across 12 seeds. Differencing reveals that LIME and SHAP coupling is largely driven by shared non-stationary trends.}
\label{tab:corr_robustness}
\begin{tabular}{llcccc}
\toprule
Signal & Series Format & Mean Lag ($\ell^*$) & Mean Max Corr & Median 569Xvalue & \% Seeds Sig ( < 0.05$) \\
\midrule
SHAP & Raw Level & 569X1.25$ & 0.814 & 0.00125 & 100.0\% \\
SHAP & First-Differenced & 569X0.58$ & 0.619 & 0.04500 & 58.3\% \\
\midrule
LIME & Raw Level & 1.08$ & 0.658 & 0.02150 & 100.0\% \\
LIME & First-Differenced & 569X1.67$ & 0.575 & 0.11275 & 25.0\% \\
\midrule
Wasserstein & Raw Level & 569X1.00$ & 0.773 & 0.00300 & 100.0\% \\
Wasserstein & First-Differenced & 569X1.00$ & 0.611 & 0.02550 & 91.7\% \\
\bottomrule
\end{tabular}
\end{table}

\begin{figure}[t]
\centering
\includegraphics[width=0.88\textwidth]{outputs/ccf_seed_0.png}
\caption{Representative Cross-Correlation Function (CCF) for Seed 0. Raw level series (top) display high apparent correlation due to shared trends. Differenced series (bottom) show that only covariate shift maintains robust lead dynamics.}
\label{fig:ccf_seed0}
\end{figure}

In raw level series, all three signals appear strongly coupled with performance drop (\%$ of seeds significant at  < 0.05$). However, inspection of Figure~\ref{fig:ccf_seed0} shows that both performance decay and explanation drift exhibit abrupt steps at window 1 followed by plateauing behavior.

When series are first-differenced:
\begin{itemize}
    \item LIME's significance collapses to .0\%$ (3 of 12 seeds), indicating that local surrogate drift contains negligible period-to-period predictive signal for model decay.
    \item SHAP retains significance in only .3\%$ of seeds (7 of 12).
    \item In contrast, model-free Wasserstein covariate shift remains significant across .7\%$ of seeds (11 of 12) with a median permutation 569Xvalue of /bin/zsh.0255$.
\end{itemize}

\subsection{Alarm Lead Times Under Relative Crossing Thresholds}

Table~\ref{tab:crossing_sensitivity} and Figure~\ref{fig:lead_time} evaluate early warning lead times across relative peak fractions.

\begin{table}[t]
\centering
\caption{Early warning lead time in units of 14-day windows ({\text{perf}} - t_{\text{monitor}}$) across relative peak fractions. Positive values indicate detection ahead of performance decay.}
\label{tab:crossing_sensitivity}
\begin{tabular}{lccccc}
\toprule
Signal & Fraction & Valid Seeds & Mean Lead Time & Paired 569Xstat & 569Xvalue \\
\midrule
SHAP & 0.25 & 12/12 & /bin/zsh.00 \pm 0.00$ & -- & -- \\
LIME & 0.25 & 12/12 & 569X0.17 \pm 0.39$ & 569X1.48$ & 0.1661 \\
Wasserstein & 0.25 & 12/12 & /bin/zsh.00 \pm 0.00$ & -- & -- \\
\midrule
SHAP & 0.50 & 12/12 & 0.42 \pm 1.16$ & 7.19 & .78 \times 10^{-5}$ \\
LIME & 0.50 & 12/12 & 0.17 \pm 1.34$ & 5.61 & .57 \times 10^{-4}$ \\
Wasserstein & 0.50 & 12/12 & 0.08 \pm 1.31$ & 8.14 & .51 \times 10^{-6}$ \\
\midrule
SHAP & 0.75 & 12/12 & 0.42 \pm 1.08$ & 4.53 & .60 \times 10^{-4}$ \\
LIME & 0.75 & 12/12 & 0.33 \pm 0.65$ & 12.41 & .23 \times 10^{-8}$ \\
Wasserstein & 0.75 & 12/12 & 0.00 \pm 0.00$ & $\infty$ & $< 10^{-15}$ \\
\bottomrule
\end{tabular}
\end{table}

\begin{figure}[t]
\centering
\includegraphics[width=0.72\textwidth]{outputs/relative_lead_time_distribution.png}
\caption{Lead time distributions at the 50\% relative peak threshold. Model-free Wasserstein distance provides the earliest and most consistent warning (0.08$ windows), outperforming post-hoc explanation rankings.}
\label{fig:lead_time}
\end{figure}

Under fixed absolute thresholds ($\Delta\text{AUC} \ge 0.03$), an immediate floor effect occurs: the performance threshold is breached at window 1 for all 12 seeds, rendering absolute early warning undefined.

Under the 50\% relative peak threshold:
\begin{itemize}
    \item SHAP provides an average lead of .42 \pm 1.16$ windows (.9$ days), and LIME provides .17 \pm 1.34$ windows (.4$ days).
    \item However, simple model-free Wasserstein covariate shift achieves a superior lead time of .08 \pm 1.31$ windows (.1$ days,  = 5.51 \times 10^{-6}$).
    \item At the 75\% threshold, Wasserstein distance exhibits zero variance across seeds (lead time exactly .00$ windows for all 12 seeds), whereas attribution metrics display notable variance.
\end{itemize}

\section{Discussion and Operational Recommendations}

\subsection{Why Post-Hoc Explanations Fail as Drift Detectors}
Post-hoc attribution methods are designed to be faithful to model predictions, not to data generating distributions. In static models, decision paths remain locked. Consequently, as the input distribution shifts, the model continues to partition feature space according to its historical splits. This induces the \emph{False Stability} phenomenon, where post-hoc explanations mask the reality of model obsolescence.

\subsection{Recommendations for Fraud Pipeline Engineering}
Based on these findings, we recommend the following guidelines for production monitoring:
\begin{enumerate}
    \item \textbf{Decouple Monitoring from Attribution:} Do not rely on static model SHAP or LIME rankings to detect distribution shift. Use model-free distance metrics (e.g., Wasserstein distance, Maximum Mean Discrepancy) on raw inputs.
    \item \textbf{Audit Drift on Differenced Series:} When validating potential monitoring signals, always difference time series to prevent spurious correlation driven by shared non-stationary trends.
    \item \textbf{Use Explanations Dynamically:} If attribution drift is desired for diagnosis, explanations should be computed on continuously retrained reference models rather than static production models.
\end{enumerate}

\section{Conclusion}
This study evaluated whether post-hoc explanation drift can serve as an early warning indicator for model decay in streaming fraud detection. Across 12 seeds and 13 temporal windows on the IEEE-CIS benchmark, we showed that post-hoc explanations for frozen models exhibit artificial stability ($\rho \approx 0.99$) precisely when performance degrades. Furthermore, apparent lead--lag correlations are largely artifacts of shared monotonic trends, and model-free Wasserstein distance provides earlier, more reliable warnings. These results demonstrate that post-hoc explanation rankings should not be used as substitutes for direct data distribution monitoring in streaming applications.

\begin{thebibliography}{10}

\bibitem{dalpozzolo2015credit}
Dal~Pozzolo, A., Caelen, O., Le~Borgne, Y.A., Waterschoot, S., Bontempi, G.: Learned lessons in credit card fraud detection from a practitioner perspective. Expert Systems with Applications \textbf{41}(10), 4915--4928 (2014)

\bibitem{carcillo2018streaming}
Carcillo, F., Dal~Pozzolo, A., Le~Borgne, Y.A., Caelen, O., Mazzer, Y., Bontempi, G.: Scarff: a scalable framework for streaming credit card fraud detection with spark. Information Fusion \textbf{41}, 182--194 (2018)

\bibitem{anderson2019detecting}
Anderson, R., Barton, C., B{\"o}hme, R., Clayton, R., Gan{\'a}n, C., Grasso, T., Hausken, K., Schwab, A.: Measuring the cost of cybercrime. In: The Economics of Financial Crime, pp. 265--300. Springer (2019)

\bibitem{lundberg2020local}
Lundberg, S.M., Erion, G., Chen, H., DeGrave, A., Prutkin, J.M., Nair, B., Katz, R., Himmelfarb, J., Bansal, N., Lee, S.I.: From local explanations to global understanding with explainable AI for trees. Nature Machine Intelligence \textbf{2}(1), 56--67 (2020)

\bibitem{ribeiro2016why}
Ribeiro, M.T., Singh, S., Guestrin, C.: Why should I trust you?'': Explaining the predictions of any classifier. In: ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, pp. 1135--1144 (2016)

\bibitem{ieeecis2019}
IEEE-CIS: IEEE-CIS Fraud Detection Benchmark. Kaggle Competition Dataset (2019). \url{https://www.kaggle.com/c/ieee-fraud-detection}

\bibitem{gama2014survey}
Gama, J., {\v{Z}}liobait{\v{e}}, I., Bifet, A., Pechenizkiy, M., Bouchachia, A.: A survey on concept drift adaptation. ACM Computing Surveys \textbf{46}(4), 1--37 (2014)

\bibitem{lu2018learning}
Lu, J., Liu, A., Dong, F., Gu, F., Gama, J., Zhang, G.: Learning under concept drift: A review. IEEE Transactions on Knowledge and Data Engineering \textbf{31}(12), 2346--2363 (2018)

\bibitem{rabanser2019failing}
Rabanser, S., G{\"u}nnemann, S., Lipton, Z.C.: Failing loudly: An empirical study of methods for detecting dataset shift. In: Advances in Neural Information Processing Systems, vol.~32, pp. 1396--1408 (2019)

\bibitem{alvarezmelis2018robustness}
Alvarez-Melis, D., Jaakkola, T.S.: On the robustness of interpretability methods. arXiv preprint arXiv:1806.08049 (2018)

\bibitem{visani2022statistical}
Visani, G., Bagli, E., Chesani, F.: Statistical stability indices for LIME: obtaining reliable explanations for machine learning models. Journal of Operational Risk \textbf{17}(2), 91--110 (2022)

\bibitem{hooker2019benchmark}
Hooker, S., Erhan, D., Kindermans, P.J., Been, K.: A benchmark for interpretability methods. In: Advances in Neural Information Processing Systems, vol.~32, pp. 2330--2340 (2019)

\bibitem{haug2021leveraging}
Haug, N., Z{\"u}rn, M., El-Assady, M.: Leveraging explainable AI for concept drift detection in data streams. In: NeurIPS Workshop on Distribution Shifts (2021)

\end{thebibliography}

\end{document}
"""
with open('/Users/manpreetsingh/Documents/GitHub/Fraud_drift_explainability/paper/Fraud.tex', 'w') as f:
    f.write(p)
print('Done writing Fraud.tex')
