# -*- coding: utf-8 -*-
# Paper content. Citation placeholders: {{key}} or {{key1,key2}} -> [n], [m]
#
# Eight-page version. Every quantitative claim is taken from the evaluation
# reports in Artifact/reports/ and the figures built by
# Artifact/docs/paper_figures.py. Nothing is estimated by hand.

REFS = {
 "hadden": 'J. Hadden, A. Tiwari, R. Roy, and D. Ruta, "Computer assisted customer churn management: state-of-the-art and future trends," Comput. Oper. Res., vol. 34, no. 10, pp. 2902–2917, 2007.',
 "verbeke": 'W. Verbeke, K. Dejaeger, D. Martens, J. Hur, and B. Baesens, "New insights into churn prediction in the telecommunication sector: a profit driven data mining approach," Eur. J. Oper. Res., vol. 218, no. 1, pp. 211–229, 2012.',
 "lemmens": 'A. Lemmens and S. Gupta, "Managing churn to maximize profits," Mark. Sci., vol. 39, no. 5, pp. 956–973, 2020.',
 "vafeiadis": 'T. Vafeiadis, K. I. Diamantaras, G. Sarigiannidis, and K. C. Chatzisavvas, "A comparison of machine learning techniques for customer churn prediction," Simul. Model. Pract. Theory, vol. 55, pp. 1–9, 2015.',
 "ahmad": 'A. K. Ahmad, A. Jafar, and K. Aljoumaa, "Customer churn prediction in telecom using machine learning in big data platform," J. Big Data, vol. 6, no. 1, art. 28, 2019.',
 "decaigny": 'A. De Caigny, K. Coussement, and K. W. De Bock, "A new hybrid classification algorithm for customer churn prediction based on logistic regression and decision trees," Eur. J. Oper. Res., vol. 269, no. 2, pp. 760–772, 2018.',
 "alboukaey": 'N. Alboukaey, A. Joukhadar, and N. Ghneim, "Dynamic behavior based churn prediction in mobile telecom," Expert Syst. Appl., vol. 162, art. 113779, 2020.',
 "imani": 'M. Imani, M. Joudaki, A. Beikmohammadi, and H. R. Arabnia, "Customer churn prediction: a systematic review of recent advances, trends, and challenges in machine learning and deep learning," Mach. Learn. Knowl. Extr., vol. 7, no. 3, art. 105, 2025.',
 "burez": 'J. Burez and D. Van den Poel, "Handling class imbalance in customer churn prediction," Expert Syst. Appl., vol. 36, no. 3, pp. 4626–4636, 2009.',
 "chawla": 'N. V. Chawla, K. W. Bowyer, L. O. Hall, and W. P. Kegelmeyer, "SMOTE: synthetic minority over-sampling technique," J. Artif. Intell. Res., vol. 16, pp. 321–357, 2002.',
 "breiman": 'L. Breiman, "Random forests," Mach. Learn., vol. 45, no. 1, pp. 5–32, 2001.',
 "chen": 'T. Chen and C. Guestrin, "XGBoost: a scalable tree boosting system," in Proc. 22nd ACM SIGKDD Int. Conf. Knowledge Discovery and Data Mining, San Francisco, CA, USA, 2016, pp. 785–794.',
 "lundberg": 'S. M. Lundberg and S.-I. Lee, "A unified approach to interpreting model predictions," in Advances in Neural Information Processing Systems 30, Long Beach, CA, USA, 2017, pp. 4765–4774.',
 "lundberg2020": 'S. M. Lundberg, G. Erion, H. Chen, A. DeGrave, J. M. Prutkin, B. Nair, R. Katz, J. Himmelfarb, N. Bansal, and S.-I. Lee, "From local explanations to global understanding with explainable AI for trees," Nat. Mach. Intell., vol. 2, no. 1, pp. 56–67, 2020.',
 "ribeiro": 'M. T. Ribeiro, S. Singh, and C. Guestrin, "Why should I trust you? Explaining the predictions of any classifier," in Proc. 22nd ACM SIGKDD Int. Conf. Knowledge Discovery and Data Mining, San Francisco, CA, USA, 2016, pp. 1135–1144.',
 "adadi": 'A. Adadi and M. Berrada, "Peeking inside the black-box: a survey on explainable artificial intelligence (XAI)," IEEE Access, vol. 6, pp. 52138–52160, 2018.',
 "guidotti": 'R. Guidotti, A. Monreale, S. Ruggieri, F. Turini, F. Giannotti, and D. Pedreschi, "A survey of methods for explaining black box models," ACM Comput. Surv., vol. 51, no. 5, art. 93, 2018.',
 "rudin": 'C. Rudin, "Stop explaining black box machine learning models for high stakes decisions and use interpretable models instead," Nat. Mach. Intell., vol. 1, no. 5, pp. 206–215, 2019.',
 "elattar": 'A. El Attar and M. El-Hajj, "Explainable AI-driven customer churn prediction: a multi-model ensemble approach with SHAP-based feature analysis," Front. Artif. Intell., art. 1748799, 2026.',
 "alvarez": 'D. Alvarez-Melis and T. S. Jaakkola, "On the robustness of interpretability methods," in Proc. ICML Workshop on Human Interpretability in Machine Learning, Stockholm, Sweden, 2018.',
 "slack": 'D. Slack, S. Hilgard, E. Jia, S. Singh, and H. Lakkaraju, "Fooling LIME and SHAP: adversarial attacks on post hoc explanation methods," in Proc. AAAI/ACM Conf. Artificial Intelligence, Ethics, and Society, New York, NY, USA, 2020, pp. 180–186.',
 "verbraken": 'T. Verbraken, W. Verbeke, and B. Baesens, "A novel profit maximizing metric for measuring classification performance of customer churn prediction models," IEEE Trans. Knowl. Data Eng., vol. 25, no. 5, pp. 961–973, 2013.',
 "hoppner": 'S. Höppner, E. Stripling, B. Baesens, S. vanden Broucke, and T. Verdonck, "Profit driven decision trees for churn prediction," Eur. J. Oper. Res., vol. 284, no. 3, pp. 920–933, 2020.',
 "devriendt": 'F. Devriendt, D. Moldovan, and W. Verbeke, "A literature survey and experimental evaluation of the state-of-the-art in uplift modeling: a stepping stone toward the development of prescriptive analytics," Big Data, vol. 6, no. 1, pp. 13–41, 2018.',
 "zadrozny": 'B. Zadrozny and C. Elkan, "Transforming classifier scores into accurate multiclass probability estimates," in Proc. 8th ACM SIGKDD Int. Conf. Knowledge Discovery and Data Mining, Edmonton, AB, Canada, 2002, pp. 694–699.',
 "niculescu": 'A. Niculescu-Mizil and R. Caruana, "Predicting good probabilities with supervised learning," in Proc. 22nd Int. Conf. Machine Learning, Bonn, Germany, 2005, pp. 625–632.',
 "akiba": 'T. Akiba, S. Sano, T. Yanase, T. Ohta, and M. Koyama, "Optuna: a next-generation hyperparameter optimization framework," in Proc. 25th ACM SIGKDD Int. Conf. Knowledge Discovery and Data Mining, Anchorage, AK, USA, 2019, pp. 2623–2631.',
 "chicco": 'D. Chicco and G. Jurman, "The advantages of the Matthews correlation coefficient (MCC) over F1 score and accuracy in binary classification evaluation," BMC Genomics, vol. 21, art. 6, 2020.',
 "parasuraman": 'R. Parasuraman and V. Riley, "Humans and automation: use, misuse, disuse, abuse," Human Factors, vol. 39, no. 2, pp. 230–253, 1997.',
 "barocas": 'S. Barocas and A. D. Selbst, "Big data\'s disparate impact," California Law Review, vol. 104, no. 3, pp. 671–732, 2016.',
 "mehrabi": 'N. Mehrabi, F. Morstatter, N. Saxena, K. Lerman, and A. Galstyan, "A survey on bias and fairness in machine learning," ACM Comput. Surv., vol. 54, no. 6, art. 115, 2021.',
 "mitchell": 'M. Mitchell, S. Wu, A. Zaldivar, P. Barnes, L. Vasserman, B. Hutchinson, E. Spitzer, I. D. Raji, and T. Gebru, "Model cards for model reporting," in Proc. Conf. Fairness, Accountability, and Transparency, Atlanta, GA, USA, 2019, pp. 220–229.',
 "goodman": 'B. Goodman and S. Flaxman, "European Union regulations on algorithmic decision-making and a right to explanation," AI Mag., vol. 38, no. 3, pp. 50–57, 2017.',
}

TITLE = "Predicting Customer Churn in Subscription Services Using Interpretable Machine Learning"
SUBTITLE = "A Business Decision-Support Framework for Retention Management"
AUTHOR = "Steven Eleojo Onoja"
AFFIL = ["School of Computing", "Ulster University", "Belfast, United Kingdom", "onoja-se@ulster.ac.uk"]

ABSTRACT = ("Abstract—Retention teams do not act on a churn score. They act on a reason and an offer, and the models that rank customers best are usually the ones least "
"able to supply either. This paper measures what that trade costs rather than asserting it. Logistic regression, a random forest and gradient-boosted trees were "
"trained under one seven-stage protocol on two labelled public datasets from different sectors: 165,034 retail banking clients, 21.2% of whom had closed their "
"account, and 5,000 streaming subscribers, 50.3% of whom had cancelled. A third sector was abandoned once its file proved to carry no recoverable churn label, and the "
"reasoning is reported rather than dropped. Every model was scored on two axes, discrimination and probability calibration on one and the stability of its Shapley "
"explanations on the other, the latter audited against LIME on 100 stratified test records. On the banking data the ensemble gained 0.102 in Matthews correlation over "
"the transparent baseline and gave up 0.176 of agreement between the two explainers and 0.420 of rank agreement. On the streaming data there was no penalty at all, "
"which limits how far the finding generalises. Explanations feed an editable rule table giving every flagged customer an action, an owner and a value-weighted "
"priority, delivered through a dashboard for staff with no statistical training. The recommended banking model fails the four-fifths convention on all three sensitive "
"attributes; the decision layer explains demographic drivers but cannot act on them.")

KEYWORDS = ("Keywords—customer churn; interpretable machine learning; explainable artificial intelligence; decision-support systems; customer retention; Shapley values")

# item types: h1, h2, h5, body, bullet, eq, table, tablewide, fig, figwide
CONTENT = [
("h1","Introduction"),
("body","Subscription businesses live on the customers they already have, and the arithmetic is lopsided. Winning a new customer costs several times what holding an "
"existing one does, so small movements in retention compound into large movements in lifetime value {{hadden,verbeke}}. Lemmens and Gupta sharpen the point: "
"loss-aware targeting earns substantially more than the conventional kind, which locates the value in the decision a prediction informs rather than in the prediction "
"{{lemmens}}."),
("body","Machine learning has been the standard response for two decades, and benchmarking studies agree that tree ensembles beat simpler classifiers on tabular churn "
"data {{vafeiadis,ahmad}}. Adoption has not kept pace. Told that a customer's churn probability is 0.82, a retention manager still cannot choose between a discount, a "
"service call and a contract upgrade, and has no way to judge whether the model's reasoning is sound. Rudin argues that opaque models are routinely fielded where "
"unexplained errors are expensive, which makes this a deployment risk rather than an aesthetic complaint {{rudin}}; where the sector is regulated the pressure is "
"legal as well, since European rules give individuals a qualified right to meaningful information about automated decisions affecting them {{goodman}}."),
("body","Post-hoc attribution looks like the answer. SHAP and LIME give every input feature a quantified contribution to every individual prediction "
"{{lundberg,ribeiro}}, and churn studies increasingly include them {{elattar}}. What those studies rarely include is the step retention work needs: a repeatable route "
"from an attribution to an action a customer relationship management (CRM) team can execute and later defend. The explanation is produced for the analyst, and there "
"it stops."),
("body","Two questions organise what follows. How much explanation stability does a churn pipeline give up when it moves from a transparent baseline to a tuned "
"ensemble, and does that price survive a change of domain? And can per-customer attributions become auditable retention actions without the system making offers on "
"the basis of protected attributes? Neither can be answered by simulating part of a pipeline, so the framework below exists as working software and every figure "
"reported comes from a registered training run. The contributions are four."),
("bullet","A comparison of logistic regression, random forest and XGBoost under one protocol across two sectors, so that measured differences reflect the models "
"rather than uneven tuning effort."),
("bullet","A two-axis assessment reporting discrimination, calibration and explanation stability together, which prices the trade between them and finds the price is "
"not constant across domains."),
("bullet","An explanation-to-action taxonomy held as an editable rule table, with owners, value-weighted priority and an invariant preventing any offer resting on a "
"demographic driver."),
("bullet","A working decision-support system: batch scoring, model cards publishing the fairness audit, a versioned registry and a logged override trail."),

("h1","Related Work"),
("h2","What the Benchmarks Do Not Settle"),
("body","Churn has been treated as supervised classification since the earliest CRM surveys, which already noted the difficulty that has not gone away: getting from a "
"prediction to a management action {{hadden}}. Vafeiadis et al. benchmarked standard classifiers on telecommunications data and found boosted ensembles ahead "
"{{vafeiadis}}; Ahmad et al. reproduced the result at industrial scale, reporting areas under the receiver operating characteristic curve above 0.93 on operator data "
"covering millions of subscribers {{ahmad}}. De Caigny et al. built a hybrid of logistic regression and decision trees for a revealing reason, which was that their "
"users could not read a pure ensemble {{decaigny}}, and Alboukaey et al. showed that behavioural features computed over short windows beat static monthly aggregates "
"{{alboukaey}}. Imbalance runs through all of it: Burez and Van den Poel demonstrated that sampling strategy shifts reported performance as much as the choice of "
"classifier does {{burez}}, and SMOTE {{chawla}} remains the usual remedy."),
("body","Read together, these results settle less than their consistency suggests. A review of 240 studies published between 2020 and 2024 finds gradient boosting "
"dominant and deep learning of little help outside sequential inputs, but also finds that nearly every study evaluates one dataset from one sector {{imani}}. A "
"reported area under the curve therefore says nothing about transfer, and because datasets, splits and thresholds differ, the numbers are not comparable between "
"papers even when the metric shares a name. Imani et al. are candid about this, listing interpretability, imbalance handling and deployment realism as the field's "
"persistent gaps. The accuracy question looks closed; what remains open is whether findings travel between domains and whether anything downstream can act on them."),
("h2","The Case Against Explaining Black Boxes"),
("body","Explainable AI divides into models that can be read directly and post-hoc methods that explain a trained model from outside {{adadi,guidotti}}. Two dominate "
"applied work. LIME fits a sparse linear surrogate near a single instance, which is intuitive but sensitive to how that neighbourhood was sampled {{ribeiro}}. SHAP "
"assigns each feature its Shapley value and is alone in satisfying local accuracy, missingness and consistency at once {{lundberg}}; for tree ensembles TreeExplainer "
"computes exact values in polynomial time, which is what makes it affordable at scoring time {{lundberg2020}}."),
("body","The strongest objection to the design of this paper comes from inside that literature. Rudin's argument is not that post-hoc explanations are imperfect but "
"that they are the wrong instrument: they describe the model rather than the world, and they lend unearned authority to predictors nobody has verified {{rudin}}. "
"Empirical work supports her. Alvarez-Melis and Jaakkola show that both LIME and SHAP can return materially different attributions for inputs that differ trivially "
"{{alvarez}}, and Slack et al. construct classifiers whose post-hoc explanations hide the protected attribute the model is actually using {{slack}}. If explanations "
"can be fragile, and in the adversarial case actively deceptive, a decision layer built on them inherits the weakness."),
("body","The objection is treated here as a design constraint rather than waved away. A transparent baseline is kept throughout as a reference point, every "
"explanation is cross-checked against a second method before it reaches a business user, and features on which the two disagree are labelled as contested on screen. "
"Whether that is sufficient is taken up in Section VI, once there are numbers to argue with."),
("h2","Deciding Rather Than Predicting"),
("body","A separate strand asks what a churn model is for. Verbraken et al. formalised the expected maximum profit criterion, scoring a classifier by the campaign it "
"induces rather than by its error rate {{verbraken}}; Höppner et al. pushed the logic into training, deriving trees split on money rather than impurity {{hoppner}}; "
"and uplift modelling aims campaigns at customers whose behaviour an intervention would change rather than at those whose risk is merely high {{devriendt}}. "
"Calibration belongs here too, since a decision rule is only as good as the probabilities reaching it; ensembles distort them, and isotonic regression is the standard "
"non-parametric correction {{zadrozny,niculescu}}."),
("body","The two strands have grown up apart. Profit-aware papers seldom explain an individual prediction, and explanation papers seldom cost their recommendations. "
"Three gaps follow. The accuracy-interpretability relationship is usually presented as a trade-off to be accepted rather than a quantity to be measured, and "
"explanation stability is rarely reported at all. No widely cited study converts per-customer attributions into a structured, auditable set of retention actions for "
"non-specialists. And cross-domain validation is rare enough that the transferability of churn drivers remains genuinely unsettled {{imani}}. The framework below "
"addresses all three together, which is the point: measuring the trade is only useful if something downstream depends on the answer."),

("h1","Framework and Method"),
("h2","Architecture"),
("body","The system is a pipeline of five layers, shown in Fig. 1, each consuming the one above. One carries more weight than its position suggests: the feature layer "
"sorts predictors into behavioural, transactional, contractual, engagement and demographic groups, which gives every later explanation a vocabulary retention staff "
"already use and marks the one group on which no offer may rest."),
("fig","framework"),
("body","How this is built matters to the argument. The pipeline is one Python code base with two entry points, a modelling package that runs offline and a Django "
"application that imports it rather than reimplementing any of it, so the feature vocabulary, the calibration arithmetic and the intervention rules each have exactly "
"one definition; what the dashboard shows a manager is therefore what the training run measured. Each artefact stores the fitted pipeline, the calibrators, the "
"hyperparameters, both audits, the seed and a SHA-256 digest of the source file, and every score records the model version behind it, so a questioned decision can be "
"reproduced as it stood."),
("h2","Data"),
("body","Two labelled public datasets were used, drawn from sectors with different contractual dynamics and no shared column names, so that any claim about transfer "
"concerns the framework and not a common schema. Table I summarises them. The banking file describes 165,034 clients through credit score, country, gender, age, "
"tenure, balance, products held and estimated salary, of whom 21.2% have closed their account. The streaming file describes 5,000 subscribers through tier, hours "
"watched, days since last login, fee, payment method, profiles and average daily viewing, of whom 50.3% have cancelled. An unlabelled banking file of 110,023 records "
"demonstrates batch scoring, the realistic operational case, since a list arrives without the answer attached."),
("table","datasets"),
("body","A third sector was intended and abandoned. The e-commerce file obtained for the study has no churn label, and the obvious repair fails in both directions. "
"Define churn as the absence of a purchase within some window and the file's engagement score, correlated at r = −0.85 with days since last purchase, leaks the answer "
"straight into the predictors; remove both columns and the eleven remaining numeric predictors correlate with the derived label at |r| ≤ 0.03. Either way the result "
"would measure the construction rather than the customers, so two domains reported honestly seemed better than three manufactured. Surname was dropped from the "
"banking data rather than left unused: it identifies individuals and proxies for ethnicity and nationality."),
("body","The banking results below come from a stratified 20,000-record subsample preserving the 21.2% positive rate. That is a compute decision rather than a "
"statistical one, since the random forest grid alone requires 160 fits, and the cost belongs here rather than in a list at the end. Absolute performance on an eighth "
"of the available rows will understate what the full file supports, most of all for the ensembles, which gain most from volume. The comparison between models is less "
"exposed, because all three see the same rows, and the subsample is recorded in the artefact."),
("h2","Preparation"),
("body","Preparation is identical across datasets so that cross-domain comparison is not confounded by preprocessing. Duplicate identifiers are removed, continuous "
"fields median-imputed and min-max scaled, and categoricals one-hot encoded after missing values receive an explicit level rather than being silently filled, so that "
"absence stays visible to the explanation layer. Near-zero-variance predictors are dropped, as is one member of any pair correlated above 0.9 — on the banking data, a "
"balance-per-product ratio and a zero-balance flag, both restating the account balance. Correlated duplicates are worth removing for a second reason: they split one "
"Shapley attribution across columns that mean the same thing {{lundberg2020}}. The split is 80/20, stratified on the label."),
("body","SMOTE handles imbalance {{chawla}} under two constraints the literature treats as necessary for honest evaluation {{burez}}. Synthetic points come only from "
"training data, and the sampler sits inside the cross-validation pipeline so that it is fitted within each fold rather than before the split. Test partitions keep "
"their natural class balance, since a performance claim is only credible against the imbalance the model will meet in production."),
("h2","Models"),
("body","Three classifiers form a ladder of rising capacity and falling transparency, which is the trade the study is trying to price. Logistic regression is the "
"baseline, modelling churn probability as in (1), where x is the feature vector, β the coefficients and β subscript zero the intercept."),
("eq", [("p","i"),("(","n"),("x","i"),(") = 1 / (1 + exp(−(β","n"),("0","sub"),(" + β","n"),("T","sup"),("x","i"),(")))","n")]),
("body","Each coefficient is a log-odds contribution and the probabilities need little correction. Its weakness is the linearity assumption: interactions and threshold "
"effects, both common in churn data, must be hand-built or are missed. It is kept precisely because that weakness is fixed and known, which makes it a stable "
"reference point. Random forest averages decorrelated trees grown on bootstrap samples {{breiman}}, and XGBoost fits trees sequentially against the residuals with "
"explicit complexity penalties {{chen}}; both capture interactions without manual engineering, at the cost of direct readability and of well-scaled probabilities, and "
"boosting leads tabular churn benchmarks {{vafeiadis,ahmad,imani}}. Deep architectures were set aside, since on tabular data at this scale they seldom beat tuned "
"boosting while demanding more data and heavier explanation machinery {{imani}}."),
("h2","Protocol"),
("body","Every model runs the same seven stages, so that the comparison reflects the models and not the effort spent on each: prepare, tune, calibrate, choose a "
"threshold, refit and evaluate once, audit the explanations, register the artefact. Tuning is stratified ten-fold cross-validation scored on average precision with "
"SMOTE inside each fold, across fourteen candidates for logistic regression, a sixteen-point grid for the forest, and twenty-five Optuna trials under a "
"tree-structured Parzen estimator for XGBoost, whose space is too large to enumerate {{akiba}}. Seeds are fixed at 42 and recorded."),
("body","Calibration uses isotonic regression on out-of-fold probabilities from a five-fold split of the training partition {{zadrozny}}, for two reasons rather than "
"one. Ensembles distort probabilities in characteristic ways and need correcting before anything downstream consumes them {{niculescu}}, and isotonic regression is "
"monotone, so it repairs the probability without disturbing the Shapley ranking the decision layer reads. A method that improved calibration by reordering the drivers "
"would be useless here."),
("body","The operating threshold is a business decision in technical clothing. Rather than defaulting to 0.5, it is chosen on the validation folds to maximise F1 "
"subject to a precision floor of 0.60, a figure that in practice belongs to whoever owns the retention budget, since precision decides how much of that budget goes to "
"customers who were never going to leave. The threshold travels with the model. Section V-B shows what the choice is worth."),
("h2","Measuring Both Axes"),
("body","Accuracy is close to useless here: a classifier that never predicts churn is 79% accurate on the banking data. Precision and recall carry the operational "
"meaning instead, the first governing budget wasted on customers who were staying anyway and the second governing revenue that walks out unnoticed. Threshold-free "
"discrimination is reported as ROC-AUC, with the Matthews correlation coefficient alongside it, because ROC-AUC flatters models under imbalance and MCC is the more "
"conservative summary of a confusion matrix {{chicco}}. Calibration is the Brier score. A profit reading values the confusion matrix at the operating threshold "
"{{verbraken}}; its unit economics, 100 for a customer kept, 15 for an offer wasted and 200 for one lost, are placeholders that fix where the threshold lands, so "
"every monetary figure below demonstrates a mechanism rather than forecasting a return."),
("body","SHAP expresses each prediction as an additive attribution, shown in (2), where φ subscript zero is the base value equal to the mean model output and each "
"φ subscript j the Shapley value of feature j {{lundberg}}."),
("eq", [("g","i"),("(","n"),("z","i"),("′) = φ","n"),("0","sub"),(" + Σ","n"),("j","isub"),(" φ","n"),("j","isub"),(" ","n"),("z","i"),("′","n"),("j","isub")]),
("body","Explanation quality needs measures of its own and no convention has settled, so the choices here are stated with their reasoning. SHAP output is compared "
"against LIME on 100 test records per model, stratified across the risk range {{ribeiro}}, and three quantities are recorded. Overlap between the two methods' top "
"five features asks whether they point at the same evidence; rank agreement asks the harder question of whether they order it the same way; and LIME's agreement with "
"itself across random restarts separates disagreement between methods from noise inside one of them, which matters because an explainer that contradicts itself cannot "
"arbitrate anything {{alvarez}}. Five is the comparison depth because the dashboard surfaces three drivers and a margin of two absorbs reordering near the cut. "
"Features on which the two disagree in more than half the records where SHAP ranks them highly are recorded as contested, and any recommendation resting on one "
"carries a visible caveat."),

("h1","From Explanation to Action"),
("body","The three strongest drivers for a customer are grouped into the categories of Section III-A and matched against a rule table built with reference to the "
"retention-marketing literature {{lemmens,verbeke}}. Table II extracts from the eighteen rules implemented. Rules match a named feature or a whole category, are "
"evaluated in order so that specific rules precede general ones, and the first match on an actionable driver produces the recommendation."),
("table","interventions"),
("body","Two mechanisms turn the output into a work queue rather than a ranking. Each rule carries a weight that combines with the customer's calibrated probability "
"band to yield a priority of P1 to P3, and priority is then scaled by the value at stake against the median in the training data, bounded between 0.6 and 1.6. The "
"second exists because targeting the customers worth most to keep is not the same as targeting those most likely to leave, and a queue sorted on probability alone "
"assumes it is {{lemmens}}. The table is JSON, re-read whenever it changes on disk and published in the dashboard, so a retention manager can rewrite it without a "
"developer."),
("body","One rule is an invariant rather than a preference. Where a customer's leading driver is demographic, no automated offer rests on it: the recommendation falls "
"through to the strongest actionable driver, and if there is none the case goes to a human. The explanation still shows the demographic driver, because concealing it "
"would misrepresent the model. This is enforced in the decision layer and covered by a regression test, which is the difference between a policy and an intention."),
("body","Fig. 2 shows the resulting queue for a 400-record slice of the unlabelled banking file, each customer carrying a banded probability, three drivers in category "
"language and an intervention with its owner and priority. Of those customers 85 sit above the operating threshold and 29 are assigned P1, while the batch-level "
"driver profile puts 43% of total attribution on contract and products and 41% on demographic attributes — a campaign-level reading in its own right, since a dominant "
"category usually calls for one structural change rather than a hundred individual calls. A driver marked not used for offers is demographic; one marked disputed is a "
"feature on which SHAP and LIME disagreed during the audit, and 215 of the 400 explanations here carry at least one such caveat. Recommendations may be overridden, an "
"override requires a reason, and the reason, the person and the timestamp go to a log that is read-only in the administrative interface. The overview screen displays "
"the override rate, because a rate of zero is not reassurance; it is the signature of automation bias {{parasuraman}}."),
("figwide","dashqueue"),

("h1","Results"),
("h2","Predictive Performance"),
("body","Table III reports the single evaluation of each model on a test partition untouched by preparation, tuning, calibration or threshold selection, alongside the "
"explanation audit of the same model. On the banking data the ordering matches the benchmarking literature. XGBoost leads on every threshold-free measure, at ROC-AUC "
"0.883, MCC 0.570 and F1 0.665; the forest follows at 0.881, 0.543 and 0.644; logistic regression trails at 0.831, 0.468 and 0.541. Accuracy is the interesting column "
"precisely because it says nothing. All three sit between 0.842 and 0.848, half a percentage point apart, while MCC spreads across ten. A study reporting accuracy "
"alone would conclude that the choice of model does not matter here, and would be wrong."),
("tablewide","results"),
("body","Setting these numbers against published work needs care. Ahmad et al. report above 0.93 on telecommunications data at industrial scale {{ahmad}}, and most "
"tabular churn studies cluster in the high 0.80s to low 0.90s {{imani}}. The banking result sits at the lower edge of that range, and three differences account for "
"the distance rather than excuse it: a different sector, an eighth of the available rows, and a pipeline constrained to leave a calibrated, explainable model at the "
"end, which most published comparisons are not. Cross-paper comparison of areas under the curve is weak evidence in any case when splits, thresholds and preprocessing "
"differ — one of the field-level problems identified in Section II."),
("body","The streaming figures are a warning rather than a result, and are reported here as a finding about the dataset. ROC-AUC of 0.9999 and MCC of 0.984 are not how "
"churn behaves. A depth-five decision tree fitted to three of the file's columns — average daily viewing, days since last login and hours watched — recovers 90% of "
"the label under five-fold cross-validation, so the models are rediscovering the rule that generated the file. The dataset still earns its place, since it exercises "
"every layer on a second vocabulary and its near-balanced classes counterweight the banking imbalance, but the banking numbers are the ones that should carry weight. "
"The same caveat appears on the streaming model cards, where a reader might otherwise see 0.99 and believe it."),
("h2","Calibration and the Price of a Threshold"),
("body","Fig. 3(a) shows the recommended banking model before and after calibration. Raw boosted output is overconfident across the whole range; isotonic regression "
"pulls the curve onto the diagonal and improves the Brier score from 0.138 to 0.103 out of fold, with 0.099 on the test partition. Because the decision layer bands "
"probabilities and scales priority by them, this is not cosmetic. It is what makes a displayed 0.72 mean roughly what a retention manager will take it to mean."),
("fig","calthresh"),
("body","Fig. 3(b) prices the threshold. Holding the model fixed and sweeping the cut across its range, campaign value at the selected 0.329 is +1.56 per customer, "
"while at the conventional default of 0.5 it is −6.52. The 8.08 difference is produced entirely by a choice the model does not make. The curve also explains why the "
"precision floor is there: value rises as the threshold falls, because under the assumed economics a missed churner costs 200 and a wasted offer 15, so an "
"unconstrained optimiser would contact very nearly everyone. That is a defensible strategy only if those three numbers are right. The floor is where the budget "
"holder's judgement enters the system, and it is visible rather than buried in a default."),
("h2","The Cost of Opacity"),
("body","The second axis carries the paper's central claim. On the banking data the trade the literature usually asserts is present and monotone: as discrimination "
"improves across the ladder, agreement between the two explainers falls. Top-five overlap declines from 0.840 for logistic regression to 0.716 for the forest and "
"0.664 for XGBoost. Rank agreement, the stricter test, collapses from 0.720 to 0.486 to 0.300. Expressed as a price, the 0.102 of MCC that boosting buys over the "
"transparent baseline costs 0.176 of overlap and 0.420 of rank agreement."),
("fig","joint"),
("body","Fig. 4 shows why that sentence needs its qualifier. On the streaming data the pattern does not repeat. XGBoost holds both the highest MCC and the highest "
"explanation agreement, at 0.984 and 0.794, while the forest is worst on the explanation axis at 0.722. Where the underlying relationship is a clean threshold rule, a "
"well-regularised booster can recover it in a form that two different explainers describe alike. The generalisable claim is therefore narrower than the banking half "
"alone would support: complexity does not always cost interpretability, but the cost is real where it occurs, it varies with the data, and it can be measured instead "
"of assumed. Reporting only the banking result would have been overclaiming, and the discrepancy between the two datasets is as much a finding as the banking "
"trade-off itself."),
("body","Both axes feed a recommendation under a deliberately conservative rule. A model whose explanations fail the audit is disqualified whatever its discrimination, "
"and an ensemble beating the transparent baseline by less than one MCC point loses to it, on the principle that opacity should have to earn its place. XGBoost was "
"recommended on both datasets, by 0.102 MCC on banking and 0.199 on streaming. Had the banking margin been a tenth of that, the baseline would have won, and the "
"registry records the reasoning rather than only the outcome."),
("body","The audit also names where the two methods part company: for the recommended banking model, six of sixteen encoded predictors, with disagreement rates between "
"0.89 and 1.00 on the records where SHAP ranks them highly. Two of them, account balance and tenure, are drivers the taxonomy would otherwise act on, which is why the "
"dashboard flags them rather than suppressing them."),
("h2","What the Models Read"),
("body","The global Shapley ranking of the recommended banking model puts age first, at a mean absolute value of 1.19, just ahead of products held at 1.14 and well "
"clear of the activity flag at 0.59, with gender and country fifth and sixth. Three of the six strongest drivers are therefore demographic. That is what turns the "
"invariant of Section IV from a principle into a working necessity: on this data, a system acting on its strongest signals would be making offers on the basis of age, "
"sex and nationality in a substantial share of cases."),
("body","The streaming model reads an unrelated vocabulary: average daily viewing at 3.47, hours watched at 3.32 and days since last login at 3.03, all behavioural, "
"with no demographic attribute anywhere near the top. The same rule table routes both populations without modification, banking customers reaching sales and customer "
"success through product and contract rules, streaming customers reaching marketing through dormancy and usage rules. A decision layer that transfers between two "
"domains sharing no column names is the strongest evidence available here on the cross-domain question raised in Section I, and it is worth being precise about what "
"it shows: the mechanism transfers, which is a claim about the framework rather than about churn drivers."),
("h2","Fairness"),
("body","Table IV reports the fairness audit of the recommended model on each dataset. On banking it fails on all three declared sensitive attributes. Selection-rate "
"ratios are 0.50 for gender, 0.36 for country and 0.07 for age band, against the four-fifths convention of 0.80 {{barocas}}, with true-positive-rate gaps of 0.15, "
"0.29 and 0.67. Three demographic attributes are additionally flagged for appearing among the model's leading drivers, which is the proxy problem the fairness "
"literature has documented repeatedly {{mehrabi}}."),
("table","fairness"),
("body","Some of this belongs to the data rather than to the model. Base churn rates differ sharply: 37.6% of German clients in the test partition had closed their "
"account against 16.3% of French clients, and 54.1% of those aged 45 to 59 against 7.3% of the under-thirties. A model ignoring differences of that size would be a "
"worse model. The operational consequence survives the explanation, though, since a system flagging 5% of under-thirties and 70% of clients aged 45 to 59 distributes "
"retention spending very unevenly, and a customer who receives no offer receives none whether the cause is bias or base rate {{barocas}}. The streaming model passes "
"the same audit comfortably. Fairness here is a property of the training data at least as much as of the algorithm applied to it, which argues for auditing every "
"deployment rather than certifying a method once."),

("h1","Discussion"),
("h2","Is a Cross-Checked Explanation Good Enough?"),
("body","Section II left an objection standing. If post-hoc explanations describe the model rather than the world {{rudin}}, and can be unstable {{alvarez}} or "
"deliberately deceived {{slack}}, what is gained by auditing one against another? The honest answer is: less than the framework would like, and considerably more than "
"nothing."),
("body","Agreement between SHAP and LIME is evidence that an attribution is a property of the model rather than of one explainer's sampling. It is not evidence that "
"the attribution is true of the customer. Two methods can agree and both be wrong, and the adversarial constructions of Slack et al. are exactly the case where they "
"would be {{slack}}. So the audit does not refute Rudin. What it does is convert an unmeasured risk into a measured one and then act on the measurement, and the "
"banking numbers show why that is worth doing: rank agreement of 0.300 for the recommended model is poor, nothing in its discrimination hints at that, and without the "
"audit nobody would know. Six features were identified as contested and are labelled in the interface, which lets a manager treat those recommendations with the "
"scepticism they deserve."),
("body","A stronger response is available, and the results half support it. Rudin's remedy is to use an interpretable model wherever one will do. The selection rule "
"here implements a weak version of that, requiring an ensemble to beat the baseline by a full MCC point; on banking the margin was 0.102, so the ensemble won. Whether "
"it should have is arguable, and the argument is worth making. A reviewer could reasonably hold that 0.102 of MCC bought with a 0.420 collapse in rank agreement is a "
"poor trade for a system whose whole purpose is to supply reasons, and that the disqualification threshold should be set on the explanation axis as well as the "
"accuracy one. That is a defensible alternative design, and the rule table makes it a one-line change. What the framework insists on is that the trade be visible. "
"Where it should be struck is a business decision, like the precision floor."),
("h2","Which Limitations Carry the Most Weight"),
("body","The study has several limitations and they are not equally serious, so ranking them is more useful than listing them. Most serious is that every figure comes "
"from one seed and one split. The headline difference of 0.102 MCC is reported without an interval, so its precision is unknown, and the audit measures rest on 100 "
"records per model, which are themselves sample statistics. Nothing here establishes that the 0.027 MCC between the forest and XGBoost is distinguishable from noise, "
"and no such claim is made. The comparison the argument needs, baseline against best ensemble at 0.102, is large relative to the metric's spread and unlikely to "
"reverse, but that word is doing real work. Repeated splits with confidence intervals are the first thing further work should add, and they are cheap."),
("body","Next is the 20,000-record subsample, which depresses absolute performance for all three models and probably compresses the differences between them, since "
"ensembles gain most from volume. At least the direction of that bias is knowable: a full run would likely widen the accuracy gap and, if the pattern in Fig. 4 holds, "
"widen the explanation penalty with it, leaving the qualitative conclusion intact and the reported numbers conservative."),
("body","Third, the streaming dataset's synthetic label limits what the second domain can support. It shows that the pipeline and the taxonomy transfer but cannot "
"corroborate anything about real churn behaviour, so the absence of a trade-off there is weaker evidence than the presence of one on banking. Fourth, and least "
"threatening to the results while most limiting for practice, the taxonomy is untested: it encodes a judgement that a given driver is best answered by a given offer, "
"resting on the retention-marketing literature and on face validity rather than on evidence from firms that would use it. Editability was chosen for that reason, but "
"it moves the burden rather than discharging it, and the system's end-to-end value remains a hypothesis until measured against realised retention. Two further "
"constraints deserve naming without ranking: the unit economics are placeholders, so the monetary figures illustrate a mechanism, and the interface was designed "
"against personas rather than validated with users."),
("h2","What the Fairness Result Obliges"),
("body","A framework that audits fairness and then does nothing would be worse than one that never looked, since it would document the harm and proceed anyway. Three "
"things follow from the banking result and all three are implemented: the audit is published on the model card rather than filed {{mitchell}}, so anyone acting on a "
"score can see that the model selects 5% of one age band and 70% of another; the demographic invariant stops that disparity being converted directly into differential "
"offers; and an unmatched case reaches a person, with the override logged. None of this makes the disparity acceptable. It makes it visible and contestable, which is "
"close to the most a prediction system can do about a property of its own training data."),
("body","The remaining safeguards follow the same pattern of implementing rather than declaring: authentication on every screen showing customer risk, covered by a "
"test; a configuration flag that keeps the score, the drivers and the recommendation while discarding the personal data behind them; and purpose limitation applied "
"during feature selection, with surname the concrete case. The per-customer explanations the system already produces are then the natural mechanism for honouring a "
"right to meaningful information about an automated decision {{goodman}}."),
("h2","Where This Should Go Next"),
("body","The most valuable next step is not a better classifier. It is uplift modelling, which estimates the effect of an intervention on each customer and would move "
"the decision layer from risk-based to effect-based targeting {{devriendt}}, attacking the taxonomy's untested assumption at its root; profit-driven training "
"objectives would then align the models with the economics rather than applying them afterwards {{hoppner,verbraken}}. Cheapest by a wide margin, repeated-split "
"evaluation with intervals would settle the statistical question raised above. A field trial against a control group remains the standard the framework should "
"ultimately meet."),

("h1","Conclusion"),
("body","The useful frontier in churn prediction is no longer accuracy but the distance between a probability and a decision, and this paper set out to measure what "
"crossing that distance costs. Scoring both axes rather than one produced the central result: on the banking data, the 0.102 of MCC that gradient boosting buys over a "
"transparent baseline costs 0.176 of agreement between two explainers and 0.420 of rank agreement. The price is real, it did not recur on the second dataset, and the "
"field has more often assumed it than quantified it. Auditing fairness rather than presuming it produced the second, since the recommended banking model fails the "
"four-fifths convention on all three sensitive attributes. Neither finding depends on the taxonomy being right about which offer suits which driver, which remains "
"untested; both rest on a narrower claim, that a churn system should report what its explanations cost in the same breath as what its predictions gain."),

("h5","Acknowledgment"),
("body","The author thanks his project supervisor at the School of Computing, Ulster University, for detailed and constructive feedback on an earlier draft of this "
"paper, which materially improved its scope and rigour."),
]

TABLES = {
"datasets": {
 "caption": "Datasets Used, and the One Excluded",
 "widths": [1100, 900, 720, 780, 1440],
 "header": ["Dataset", "Domain", "Records", "Churn rate", "Role in the study"],
 "rows": [
  ["Bank customer churn", "Retail banking", "165,034", "21.2%", "Trained; 20,000-record stratified subsample"],
  ["Streaming subscriber churn", "Video on demand", "5,000", "50.3%", "Trained; synthetic label, see Section V-A"],
  ["Bank customer list", "Retail banking", "110,023", "n/a", "Unlabelled: dashboard batch scoring"],
  ["E-commerce features", "Electronic commerce", "6,000", "n/a", "Excluded: no label derivable without leakage"],
 ]},
"interventions": {
 "caption": "Extract from the Implemented Intervention Taxonomy",
 "widths": [1500, 2100, 1340],
 "header": ["Leading actionable driver", "Recommended intervention", "Business owner"],
 "rows": [
  ["Days since last login (usage)", "Win-back sequence: content prompt and reactivation offer", "Marketing"],
  ["Short tenure (contract)", "Structured onboarding contact within the first 90 days", "Customer success"],
  ["Single product held (contract)", "Bundle offer: cross-sell a second product", "Sales"],
  ["High monthly fee (money)", "Tariff review: loyalty pricing or a better-matched plan", "Pricing"],
  ["Any engagement driver", "Priority service recovery call within 48 hours", "Service recovery"],
  ["Any demographic driver", "No automated offer; refer for human review", "Retention lead"],
 ]},
"results": {
 "caption": "Predictive Performance and Explanation Audit on the Held-Out Test Partition",
 "widths": [1000, 1300, 850, 850, 850, 1000, 850, 900, 1100, 1100],
 "header": ["Dataset", "Model", "Acc.", "Prec.", "Rec.", "ROC-AUC", "MCC", "Brier", "Top-5 overlap", "Rank agr."],
 "rows": [
  ["Banking", "Logistic reg.", "0.842", "0.699", "0.441", "0.831", "0.468", "0.117", "0.840", "0.720"],
  ["Banking", "Random forest", "0.842", "0.613", "0.677", "0.881", "0.543", "0.101", "0.716", "0.486"],
  ["Banking", "XGBoost", "0.848", "0.624", "0.713", "0.883", "0.570", "0.099", "0.664", "0.300"],
  ["Streaming", "Logistic reg.", "0.892", "0.875", "0.917", "0.966", "0.785", "0.073", "0.778", "0.559"],
  ["Streaming", "Random forest", "0.989", "0.992", "0.986", "0.998", "0.978", "0.009", "0.722", "0.519"],
  ["Streaming", "XGBoost", "0.992", "0.994", "0.990", "1.000", "0.984", "0.005", "0.794", "0.566"],
 ]},
"fairness": {
 "caption": "Fairness Audit of the Recommended Model on Each Dataset",
 "widths": [900, 940, 640, 840, 780, 840],
 "header": ["Dataset", "Attribute", "Groups", "Selection ratio", "TPR gap", "Four-fifths met"],
 "rows": [
  ["Banking", "Gender", "2", "0.50", "0.15", "No"],
  ["Banking", "Country", "3", "0.36", "0.29", "No"],
  ["Banking", "Age band", "4", "0.07", "0.67", "No"],
  ["Streaming", "Gender", "3", "0.92", "0.01", "Yes"],
  ["Streaming", "Region", "6", "0.87", "0.03", "Yes"],
  ["Streaming", "Age band", "4", "0.93", "0.01", "Yes"],
 ]},
}

FIGURES = {
 "framework": {
   "file": "framework.png",
   "caption": "System architecture. Each layer consumes the outputs of the layer above it, ending in prioritised retention recommendations delivered "
              "through the dashboard.",
 },
 "dashqueue": {
   "file": "figures/fig08-dashboard-queue.png",
   "wide": True,
   "caption": "The retention queue, scoring a slice of the unlabelled banking file. Drivers marked not used for offers are demographic; those marked "
              "disputed are features on which SHAP and LIME disagreed during the audit.",
 },
 "calthresh": {
   "file": "figures/fig09-calibration-threshold.png",
   "caption": "Recommended banking model on the test partition. (a) Reliability before and after isotonic regression, ten quantile bins. (b) Campaign "
              "value per customer across the operating threshold, under the placeholder unit economics of Section III-F; the threshold is worth 8.08 per "
              "customer here, and the model does not choose it.",
 },
 "joint": {
   "file": "figures/fig03-joint-assessment.png",
   "caption": "The two assessment axes: discrimination horizontally, agreement between SHAP and LIME vertically. On banking (a) the relationship is "
              "monotone and negative; on streaming (b) it is not, so the trade-off is measurable rather than inevitable.",
 },
}
