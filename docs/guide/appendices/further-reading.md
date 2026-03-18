# Appendix E: Further Reading

> *Papers, textbooks, and resources for readers who want to go deeper on the methods behind Tallgrass.*

---

## Item Response Theory

**Baker, F. B., & Kim, S.-H. (2004).** *Item Response Theory: Parameter Estimation Techniques* (2nd ed.). Marcel Dekker.
— The standard textbook. Covers 1PL, 2PL, and 3PL models in detail with worked examples. Chapters 5-7 are most relevant to Tallgrass's use of the 2PL model.

**Clinton, J., Jackman, S., & Rivers, D. (2004).** "The Statistical Analysis of Roll Call Data." *American Political Science Review*, 98(2), 355-370.
— The foundational paper applying Bayesian IRT to legislative voting. Introduces the ideal point model that Tallgrass implements. Essential reading for understanding why IRT works for roll call data.

**Martin, A. D., & Quinn, K. M. (2002).** "Dynamic Ideal Point Estimation via Markov Chain Monte Carlo for the U.S. Supreme Court, 1953-1999." *Political Analysis*, 10(2), 134-153.
— The Martin-Quinn model that Tallgrass adapts for dynamic IRT (Phase 27, Volume 8, Chapter 2). Introduces the random walk evolution equation for ideology.

## W-NOMINATE and Related Methods

**Poole, K. T., & Rosenthal, H. (1997).** *Congress: A Political-Economic History of Roll Call Voting*. Oxford University Press.
— The book-length treatment of NOMINATE. Covers the spatial voting model, estimation procedure, and applications to Congress from 1789 to the present.

**Poole, K. T. (2005).** *Spatial Models of Parliamentary Voting*. Cambridge University Press.
— A more technical treatment including W-NOMINATE, DW-NOMINATE, and Optimal Classification. Chapter 3 covers the relationship between the Gaussian kernel utility model and the logistic IRT model.

**Lewis, J. B., & Poole, K. T. (2004).** "Measuring Bias and Uncertainty in Ideal Point Estimates via the Parametric Bootstrap." *Political Analysis*, 12(2), 105-127.
— Addresses uncertainty quantification in NOMINATE, a topic Volume 5 discusses in the context of comparing NOMINATE's point estimates to Bayesian credible intervals.

## Bayesian Methods and MCMC

**McElreath, R. (2020).** *Statistical Rethinking: A Bayesian Course with Examples in R and Stan* (2nd ed.). CRC Press.
— The most accessible introduction to Bayesian modeling. Written for applied researchers, not mathematicians. The hierarchical models chapter directly parallels Volume 4, Chapter 5.

**Gelman, A., Carlin, J. B., Stern, H. S., Dunson, D. B., Vehtari, A., & Rubin, D. B. (2013).** *Bayesian Data Analysis* (3rd ed.). CRC Press.
— The comprehensive reference. Chapter 5 (hierarchical models), Chapter 6 (model checking), and Chapter 11 (MCMC) are most relevant to Tallgrass.

**Betancourt, M. (2017).** "A Conceptual Introduction to Hamiltonian Monte Carlo." arXiv:1701.02434.
— An accessible explanation of HMC and NUTS (the sampler Tallgrass uses via nutpie). Explains why NUTS is superior to Gibbs sampling for high-dimensional models like IRT.

## External Validation Datasets

**Shor, B., & McCarty, N. (2011).** "The Ideological Mapping of American Legislatures." *American Political Science Review*, 105(3), 530-551.
— The paper behind the Shor-McCarty scores used for external validation in Volume 5, Chapter 3. Explains the bridge methodology that links state legislators to a common national scale via NPAT survey responses.

**Bonica, A. (2014).** "Mapping the Ideological Marketplace." *American Journal of Political Science*, 58(2), 367-386.
— The DIME project paper. Explains CFscores — campaign finance-based ideology estimates derived from donor networks. Used for secondary validation in Phase 18.

## Clustering and Network Analysis

**Fortunato, S. (2010).** "Community Detection in Graphs." *Physics Reports*, 486(3-5), 75-174.
— Comprehensive review of community detection methods. Covers modularity, the Leiden algorithm, and the relationship between network communities and legislative coalitions.

**Traag, V. A., Waltman, L., & van Eck, N. J. (2019).** "From Louvain to Leiden: Guaranteeing Well-Connected Communities." *Scientific Reports*, 9, 5233.
— The paper introducing the Leiden algorithm used in Phase 11. Explains why Leiden improves on Louvain for detecting meaningful communities.

**Waugh, A. S., Pei, L., Fowler, J. H., Mucha, P. J., & Porter, M. A. (2011).** "Party Polarization in Congress: A Network Science Approach." arXiv:0907.3509.
— Applies network methods to congressional roll call data. The co-voting network approach in Volume 6 draws on this framework.

## Time Series and Changepoint Detection

**Killick, R., Fearnhead, P., & Eckley, I. A. (2012).** "Optimal Detection of Changepoints with a Linear Computational Cost." *Journal of the American Statistical Association*, 107(500), 1590-1598.
— The PELT algorithm paper used in Phase 19 (Volume 8, Chapter 1). Explains the penalty-based approach and the pruning optimization.

**Bai, J., & Perron, P. (2003).** "Computation and Analysis of Multiple Structural Change Models." *Journal of Applied Econometrics*, 18(1), 1-22.
— The Bai-Perron structural break test used for confidence intervals on changepoints in Phase 19.

**Desposato, S. W. (2005).** "Correcting for Small Group Inflation of Roll-Call Cohesion Scores." *British Journal of Political Science*, 35(4), 731-744.
— The correction for small-caucus bias in the Rice Index, implemented in Phase 19's time series analysis.

## Text Analysis and NLP

**Grootendorst, M. (2022).** "BERTopic: Neural Topic Modeling with a Class-Based TF-IDF Procedure." arXiv:2203.05794.
— The BERTopic algorithm used in Phase 20 (Volume 7, Chapter 3). Explains the embedding → UMAP → HDBSCAN → c-TF-IDF pipeline.

**Kornilova, A., & Eidelman, V. (2019).** "BillSum: A Corpus for Automatic Summarization of US Legislation." *Proceedings of the 2nd Workshop on New Frontiers in Summarization*.
— Relevant to bill text analysis. Discusses challenges specific to legislative text (legal language, cross-references, procedural boilerplate).

## Political Science of Kansas

**Cigler, A. J., & Loomis, B. A. (2020).** "Kansas: The Consequences of the Brownback Tax Experiment." In *Interest Group Politics* (10th ed.).
— Political science analysis of the Brownback-era tax cuts and their aftermath, providing context for the temporal patterns documented in Volume 8, Chapter 5.

**Kansas Legislative Research Department.** Annual *Kansas Legislature Summary* reports.
— The official summary of each legislative session. Useful for identifying the specific bills and events that correspond to changepoints detected by Tallgrass.

## Software and Tools

**Salvatier, J., Wiecki, T. V., & Fonnesbeck, C. (2016).** "Probabilistic Programming in Python Using PyMC3." *PeerJ Computer Science*, 2, e55.
— The PyMC framework paper. Tallgrass uses PyMC 5.x with the nutpie sampler for all Bayesian models.

**Lundberg, S. M., & Lee, S.-I. (2017).** "A Unified Approach to Interpreting Model Predictions." *Advances in Neural Information Processing Systems*, 30.
— The SHAP paper. Explains the Shapley value framework used in Phase 15 for feature importance analysis.

**Chen, T., & Guestrin, C. (2016).** "XGBoost: A Scalable Tree Boosting System." *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*.
— The XGBoost paper. Phase 15's vote prediction and Phase 26's prediction transfer both use XGBoost.

---

## Online Resources

- **Tallgrass source code:** [github.com/codechizel/tallgrass](https://github.com/codechizel/tallgrass) (MIT license)
- **Kansas Legislature website:** [kslegislature.gov](http://www.kslegislature.gov) (the data source)
- **Voteview (DW-NOMINATE scores):** [voteview.com](https://voteview.com) (Congressional ideology data for comparison)
- **OpenStates:** [openstates.org](https://openstates.org) (multi-state legislative data; provides OCD IDs used for cross-session matching)
- **Harvard Dataverse (Shor-McCarty):** [dataverse.harvard.edu](https://dataverse.harvard.edu) (external validation data)

---

*Back to: [Appendices](./)*
