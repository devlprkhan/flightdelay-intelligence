# FlightDelay Intelligence: Production-Style Flight Delay Risk Prediction

## Problem

Airline delays are a real operational problem affecting passengers, airline operations, airport capacity, crew/aircraft rotations, and downstream flights. The FAA explicitly uses airline schedules, historical capacities, weather, and operational metrics to anticipate and manage delays; its weather program is designed to help aviation operators make better planning decisions and improve on-time performance.

Build a system that predicts **whether a scheduled U.S. domestic flight will depart significantly late**, using only information that would realistically be available before departure.

The important framing is not:

> Can ML predict flight delays?

It is:

> **Given what was knowable before a flight departs, how accurately can we identify flights that are at high risk of delay so an operations team can prioritize intervention?**

For the initial production-style prediction cutoff, use **2 hours before scheduled departure**. This is a deliberate project-design choice rather than a claim that 2 hours is an industry-wide standard.

## Users

Primary users:

- Airline operations/control-center analysts.
- Airport operations teams.
- Airline customer-service/irregular-operations teams.

Secondary product consumer:

- Passengers who want a risk estimate for an upcoming flight.

The MVP should prioritize the **operations use case**, because it creates a genuine decision problem rather than turning the project into another consumer-facing prediction demo.

## ML Objective

- **Task:** Predict the probability that a scheduled flight will experience a departure delay of more than 15 minutes at the prediction cutoff.
- **Input:** Flight schedule, carrier, origin/destination, date/time, route characteristics, historical operational patterns, airport characteristics, and eventually weather information available before departure.
- **Target:** `departure_delay > 15 minutes`.
- **Output:** A calibrated probability/risk score plus a binary high-risk decision.
- **Problem Type:** Binary classification.
- **Primary Metric:** PR-AUC.
- **Secondary Metrics:** Precision, recall, F1, ROC-AUC, confusion matrix, recall at a chosen precision level, probability calibration, and threshold-specific business performance.

The 15-minute threshold is grounded in the U.S. Department of Transportation/BTS on-time reporting framework.

PR-AUC is intentionally added beyond the existing metric list because the project should evaluate ranking of high-risk flights under class imbalance, not merely overall classification accuracy.

## Why This Fits My Skills

This project naturally exercises the existing ML foundation without forcing every algorithm into the system.

Relevant existing skills:

- **EDA:** delay distributions, cancellations, carriers, airports, routes, seasonality, time-of-day effects, extreme delays, missingness, and operational anomalies.
- **Data cleaning:** missing values, inconsistent timestamps, duplicates, categorical inconsistencies, invalid records, and unusual operational observations.
- **Feature engineering:** date/time decomposition, airport/route features, historical delay rates, rolling statistics, carrier-route interactions, congestion proxies, and aircraft/flight rotation features where legitimately available.
- **Feature selection:** deciding which operational signals genuinely contribute predictive value.
- **Encoding:** carrier, airport, route, day-of-week, time buckets, etc.
- **Scaling:** useful for linear models, SVM, and other distance-sensitive approaches.
- **Dimensionality reduction:** useful primarily for exploratory analysis of high-cardinality representations, not necessarily as a production component.
- **Logistic regression:** interpretable baseline.
- **Decision trees:** simple nonlinear baseline.
- **Random forest:** robust nonlinear comparison.
- **Gradient boosting / XGBoost / LightGBM / CatBoost:** strong candidates for tabular operational data.
- **Ensembling/stacking:** optional advanced experiment after strong individual models are established.
- **Cross-validation:** required, but adapted to temporal data rather than blindly using random k-fold.
- **GridSearchCV / RandomizedSearchCV:** hyperparameter optimization.
- **Pipelines:** essential for leakage-safe preprocessing.
- **Model persistence:** required for deployment.
- **Domain knowledge:** critical for deciding what information is actually available at prediction time.

The project therefore tests whether ML judgment can be applied in practice rather than whether algorithms can merely be recalled.

## Skill Gaps

### Already Known

- Supervised learning.
- Classification.
- Feature engineering.
- Feature selection.
- Ensemble learning.
- Model evaluation.
- Hyperparameter tuning.
- Cross-validation.
- Pipelines.
- Data cleaning.
- Model persistence.
- EDA and metadata analysis.

### Partial

**Temporal validation**

Ordinary k-fold cross-validation is known, but this project requires understanding why random splits can create unrealistic estimates when data has time-dependent operational relationships.

**Data leakage**

Leakage is conceptually familiar, but flight data makes it much harder because actual operational information becomes available at different points in the timeline.

For example, BTS contains actual gate departure/arrival times, delay causes, and other post-event information. Those variables can be excellent predictors in hindsight but invalid features for a prediction made two hours before departure.

**Probability calibration**

A risk system should distinguish between probabilities such as `0.51` and `0.95`, rather than producing only class labels.

### New but Learnable

- Precision-recall curves and PR-AUC.
- Probability calibration.
- Temporal/rolling validation.
- Time-aware feature generation.
- Leakage auditing.
- Prediction-cutoff design.
- Basic model explainability for operational users.
- Monitoring model drift.
- Cost/threshold-based decision analysis.

These are extensions of the current ML foundation, not an entirely new specialization.

## Data

### Primary Dataset — BTS Airline On-Time Performance

Use the **U.S. Bureau of Transportation Statistics Airline Service Quality Performance / On-Time Performance dataset**.

The official BTS TranStats database contains individual domestic flight records and provides scheduled/actual operational information and delay-related measurements. BTS currently exposes data through 2026 and maintains historical records going back decades.

Relevant fields include:

- Flight date.
- Airline/carrier.
- Flight number.
- Origin airport.
- Destination airport.
- Scheduled departure/arrival.
- Actual departure/arrival.
- Cancellation/diversion information.
- Taxi-out/taxi-in.
- Air time.
- Distance.
- Delay-related fields.

### Secondary Dataset — NOAA Weather

Integrate historical airport weather from NOAA/NCEI.

Potential hourly weather variables include:

- Temperature.
- Dew point.
- Pressure.
- Visibility.
- Wind speed/direction.
- Wind gusts.
- Precipitation.
- Weather phenomena.
- Cloud information.

### Airport Metadata

Use FAA airport information for airport-level characteristics where useful and available.

### Realistic Data Challenges

Expect to deal with:

- Missing values.
- High-cardinality categorical variables.
- Multiple airport identifiers.
- Time-zone/local-time issues.
- Duplicate or anomalous records.
- Highly skewed delay distributions.
- Cancellations versus delays.
- Data appearing at different points in the operational timeline.
- Temporal dependence.
- Changing airline/airport behavior.
- Weather station matching.
- Airport/weather timestamp alignment.
- Historical features that must only use information available before the prediction cutoff.
- Distribution shift between earlier and later periods.

The temporal problem is especially important because realized operational outcomes can leak information from the future into training if the feature-generation process is not carefully designed.

## ML Approach

### Baseline

Start with a deliberately simple model:

**Logistic Regression**

Use a limited set of defensible pre-departure features.

The baseline should answer:

> How much predictive value exists before more complex nonlinear models are introduced?

### Candidate Models

Compare:

- Logistic Regression.
- Decision Tree.
- Random Forest.
- XGBoost.
- LightGBM.
- CatBoost.

SVM and KNN can be used only as controlled benchmarks where justified; they do not need to become part of the final system.

The objective is:

> **simple baseline → stronger nonlinear model → evidence-based final model**

### Feature Engineering

Investigate feature groups instead of blindly generating hundreds of variables.

**Temporal**

- Hour of departure.
- Day of week.
- Month.
- Season.
- Holiday/peak-travel indicators.
- Time-of-day congestion patterns.

**Flight**

- Scheduled elapsed time.
- Route distance.
- Carrier.
- Origin.
- Destination.
- Route frequency.
- Flight-number patterns where defensible.

**Historical operational**

- Recent carrier delay rate.
- Recent airport delay rate.
- Recent route delay rate.
- Recent time-of-day delay rate.
- Rolling delay statistics.
- Previous-flight/aircraft lineage features where information is genuinely available at prediction time.

**Airport**

- Airport activity/congestion proxies.
- Airport operational characteristics.
- Origin/destination historical behavior.

**Weather**

- Visibility.
- Wind.
- Gusts.
- Precipitation.
- Temperature.
- Pressure.
- Weather-condition indicators.

### Evaluation Strategy

Do **not** use a random train/test split as the final evaluation.

Use chronological splits such as:

```text
Earlier period  → Training
Later period    → Validation
Newest period   → Final Test
```

Within training, use rolling/temporal validation.

The final test set must represent a future period relative to training.

The central question is:

> How well does a model trained on the past predict flights it has never seen in a genuinely later period?

### Leakage Rules

These rules are mandatory.

The model must **not** use:

- Actual departure time.
- Actual arrival time.
- Arrival delay.
- Post-departure delay codes.
- Cancellation information that is unavailable at prediction time.
- Future flights' realized outcomes.
- Any statistic computed using future observations.
- Any feature derived from the target.

Every feature must answer:

> **Would an operations system genuinely know this value two hours before scheduled departure?**

## Real-World Constraints

### Prediction Cutoff

The system makes its prediction at:

**T − 2 hours before scheduled departure.**

Everything used by the model must be available at that point.

### Operational Asymmetry

A false negative and false positive do not necessarily have equal costs.

Missing a genuinely high-risk flight may be more costly than unnecessarily flagging a flight for attention.

Therefore, the final model should not simply use:

`probability > 0.5`

as its decision threshold.

Determine an operating threshold based on an explicit precision/recall or cost trade-off.

### Interpretability

Operational users should understand:

- Risk probability.
- Main factors contributing to risk.
- Historical risk context.
- Whether the prediction is unusually uncertain.

Avoid building a completely opaque risk score with no explanation.

### Drift

Airline operations, airport congestion, schedules, weather patterns, and network structures change.

A model that performs well on historical data may degrade later.

Measure performance across different time periods rather than reporting one global score.

### Scalability

Separate:

- Raw data.
- Cleaned data.
- Feature datasets.
- Training datasets.
- Model artifacts.
- Prediction outputs.

The project should behave like a real ML data workflow rather than a notebook containing everything.

## Expected Product

Build a **Flight Delay Risk Intelligence Dashboard/API**.

For each upcoming flight:

```text
Flight: AA123
Route: JFK → ORD
Scheduled Departure: 18:30

Delay Risk: 78%
Risk Level: HIGH

Key Risk Signals:
- High historical origin-airport delay rate
- Elevated evening congestion pattern
- Adverse weather conditions
- Elevated recent carrier/route delay rate

Recommended Action:
Prioritize for operational review
```

The system should also provide aggregate operational views:

```text
Today's Flights
────────────────────────────────────
High Risk       43
Medium Risk     81
Low Risk        214

Highest-Risk Airports
JFK   82%
ORD   78%
EWR   75%

Highest-Risk Routes
...
```

The UI is not the core achievement. The ML system underneath it is.

## Success Criteria

The project is successful when it demonstrates all of the following:

1. A meaningful improvement over a simple logistic-regression baseline.
2. Strong performance on a genuinely future, unseen time period.
3. No target leakage or future-information leakage.
4. Evidence that engineered features improve performance.
5. Evidence-based model selection rather than choosing the fanciest algorithm.
6. Useful precision/recall behavior at an operationally meaningful threshold.
7. Reasonably calibrated risk probabilities.
8. Performance analysis across carriers, airports, routes, seasons, and time periods.
9. Clear explanation of where the model fails.
10. A deployable inference pipeline that accepts flight information and returns a risk score.

A particularly strong result would demonstrate:

> **The model can identify a useful subset of future high-risk flights early enough for intervention, while maintaining an explicitly measured false-alert rate.**

Do **not** define success as achieving very high accuracy alone.

## Why This Is Not a Toy Project

This is representative of real ML work because the difficult part is not training XGBoost.

The difficult parts are:

**Problem formulation**

Define exactly what "delay" means and when the prediction must happen.

**Data integration**

Combine flight operations, airport metadata, and potentially weather observations from separate systems.

**Temporal reasoning**

Historical information can be useful while future information can make the model invalid.

**Leakage prevention**

BTS contains actual operational outcomes that can make prediction artificially easy if included incorrectly.

**Class imbalance**

Operationally important events are less common than ordinary flights, making accuracy a poor standalone measure.

**Feature engineering**

Important information is distributed across time, airports, carriers, routes, and weather rather than existing in one obvious column.

**Model selection**

Several algorithms can perform reasonably well; the job is to determine whether added complexity actually provides enough value.

**Distribution shift**

A future period may behave differently from the historical training period.

**Business decisions**

The model eventually affects which flights receive attention, so probability thresholds and false-positive/false-negative trade-offs matter.

**Deployment**

The final system must reproduce exactly the same preprocessing and feature logic used during training.

## Difficulty

**8.5/10**

The algorithms themselves are within the current skill level.

The difficulty comes from:

- Data volume.
- Temporal validation.
- Leakage prevention.
- External-data integration.
- Feature engineering.
- Operational framing.
- Calibration.
- Model drift.
- Building something that behaves like a real prediction system.

## Portfolio Value

This project demonstrates the ability to move beyond:

> I know XGBoost.

and toward:

> **I can formulate, build, validate, and deploy a real-world tabular ML system.**

A strong implementation demonstrates:

- Real-world data acquisition.
- Messy-data EDA.
- Data-quality engineering.
- Multi-source data integration.
- Feature engineering.
- Leakage-safe ML pipelines.
- Temporal validation.
- Model benchmarking.
- Hyperparameter optimization.
- Interpretable predictions.
- Probability calibration.
- Error analysis.
- Model persistence.
- API inference.
- Monitoring/drift thinking.
- Business-oriented metric selection.

That is substantially stronger portfolio evidence for an ML Engineer than another generic Kaggle leaderboard project.

## Scope

### MVP

Build the complete prediction system using **BTS flight data first**.

The MVP should contain:

- Historical flight dataset.
- Target construction.
- Data-quality analysis.
- Thorough EDA.
- Leakage analysis.
- Temporal train/validation/test split.
- Logistic-regression baseline.
- Tree/ensemble comparisons.
- Feature engineering.
- Hyperparameter tuning.
- Final evaluation.
- Error analysis.
- Saved model pipeline.
- Prediction API.
- Basic dashboard showing flight-level risk.

Do not add every external dataset immediately.

The first milestone is proving:

> **I can build a leakage-safe future-flight delay prediction system using real operational data.**

### Advanced

Add:

- NOAA weather integration.
- FAA airport characteristics.
- Rolling historical features.
- Airport/route congestion features.
- Aircraft rotation/lineage features when prediction-time availability can be reconstructed safely.
- Probability calibration.
- Cost-sensitive threshold optimization.
- SHAP or another appropriate explanation method.
- Carrier/airport-specific error analysis.
- Model comparison across multiple historical periods.
- Drift analysis.
- Automated retraining experiments.

The strongest advanced feature is **prediction-time-safe historical lineage**.

Flight operations are sequential: an aircraft's previous flight can affect the next flight. This creates potentially useful signal, but also creates significant leakage risk if future outcomes are accidentally incorporated.

### Production Direction

Evolve the project into a service with:

```text
Flight/Schedule Data
        ↓
Data Validation
        ↓
Feature Generation
        ↓
Prediction Service
        ↓
Delay Risk Model
        ↓
Risk Score
        ↓
Operational Dashboard
```

A realistic production version would add:

- Scheduled batch inference.
- Feature-store-like historical features.
- Model/version management.
- Prediction logging.
- Monitoring.
- Drift detection.
- Model-performance tracking once actual outcomes become available.
- Automated retraining.
- API authentication.
- Containerized deployment.
- Cloud deployment.

Preserve the distinction between:

**features known at prediction time**

and

**outcomes observed after prediction**

so production evaluation remains honest.

## Data Sources & References

- **U.S. Bureau of Transportation Statistics — Airline Service Quality Performance / On-Time Performance:** official flight-level operational data including scheduled and actual flight times and delay information. https://www.bts.gov/browse-statistical-products-and-data/bts-publications/airline-service-quality-performance-234-time
- **BTS TranStats — On-Time Data:** direct interface for selecting/download­ing monthly flight-level data. https://www.transtats.bts.gov/ONTIME/
- **BTS Technical Directive No. 40 — On-Time Reporting:** current reporting specification defining relevant operational and delay fields. https://www.bts.gov/explore-topics-and-geography/modes/aviation/number-40-technical-directive-reporting-time
- **NOAA/NCEI — Dataset Search / GHCNh:** historical hourly weather observations suitable for weather enrichment. https://www.ncei.noaa.gov/access/search/dataset-search/
- **NOAA/NCEI — Integrated Surface Database:** historical hourly weather source and documentation. https://www.ncei.noaa.gov/products/land-based-station/integrated-surface-database
- **FAA — Operational Performance Evaluation:** operational context around delay projection and airline schedule monitoring. https://www.faa.gov/about/office_org/headquarters_offices/ato/service_units/systemops/perf_analysis/ops_perf
- **FAA — NextGen Weather:** aviation operational context for weather-based decision support. https://www.faa.gov/nextgen/programs/weather
- **FAA / Data.gov — Aviation Facilities:** airport infrastructure and operational characteristics. https://catalog.data.gov/dataset/aviation-facilities
- **UC Berkeley — Flight-delay ML project:** useful practical reference on temporal validation, prediction cutoff design, feature engineering, leakage prevention, lineage features, and drift. https://www.ischool.berkeley.edu/projects/2025/air-travel-delay-prediction-feature-engineering-and-ml-approaches

## Final Recommendation

**Build FlightDelay Intelligence.**

This is the strongest fit for the current level because the **core ML is already within the existing capabilities**, while the surrounding engineering and statistical problems force meaningful growth.

The project combines:

**real government data + messy multi-source data + classification + serious feature engineering + temporal validation + leakage prevention + ensemble models + operational decision-making + deployable product**

Most importantly, it cannot be completed successfully by simply copying a tutorial.

The project repeatedly forces the question that separates beginner ML work from professional ML work:

> **Would this information actually have been available when my model was making the prediction?**

That is the central challenge of this project.
