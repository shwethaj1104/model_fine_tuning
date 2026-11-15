# Dashboard narrative and evidence

The portfolio dashboard leads with the author's actual contribution: fine-tuning Qwen3-0.6B for support triage. It connects measured model behavior to the intended support workflow without claiming measured business savings.

## What is emphasized

- The headline explicitly says the model was fine-tuned.
- High-contrast chips name the model, task-specific training, held-out ticket count, and selected epoch.
- The four cards prioritize improvements, with base and fine-tuned values directly below.
- Rates use percentage-point differences. F1 uses absolute F1 differences, never a percent-accuracy label.
- Plain-language descriptions connect structured answers, issue classification, environment detection, and team recommendation to support work.
- The project contribution is explained as customization, model selection, and a fair comparison on unseen requests.
- Correct category counts and unusable-output counts show concrete model behavior.

These outputs may support a future workflow; they do not establish reduced handling time, staff savings, customer satisfaction, or production ROI. Team recommendations follow the issue category, so that metric is not independent evidence of reasoning. Test data contains 66 AI-authored examples; impact-summary factuality still needs separate review. Invalid-schema responses count as incorrect for the task metrics.

## Guidance consulted

[Google PAIR: User Needs + Defining Success](https://pair.withgoogle.com/guidebook-v2/chapter/user-needs/) recommends starting with user needs and defining success around the intended user benefit. Applied here: communicate what support staff need—consistent records and correct issue identification—alongside the model metrics.

[Google PAIR: Data Collection + Evaluation](https://pair.withgoogle.com/guidebook-v2/chapter/data-collection/) connects evaluation with user needs and success metrics. Applied here: distinguish the authored-data benchmark from performance on live support traffic.

[NIST AI RMF: Measure](https://airc.nist.gov/airmf-resources/playbook/measure/) emphasizes documented evaluation context, metrics, validity, and human oversight. Applied here: retain dataset limitations and human-review boundaries instead of converting benchmark gains into unmeasured operational claims.
