# Adversarial LLM Benchmarking — Interview Questions & Model Answers

> Companion to [`adversarial-llm-benchmarking.md`](./adversarial-llm-benchmarking.md). First-person answers, 30–90 s spoken. 🟢 Easy · 🟡 Medium · 🔴 Hard

**If you only prep 10:** A1, A3, B3, C1, C3, D2, E1, E2, G1, plus the hostile follow-ups.

---

## (a) Project overview & motivation

**A1. 🟢 What did you build?**
> A local red-teaming harness for open-source LLMs. It reads a list of models from JSON, loads each one with HuggingFace Transformers in fp16 on an H100 via SLURM, and runs 50 adversarial prompts across seven categories: injection, role-play, policy override, authority pressure, encoding, CoT extraction and hallucination probes. Each response gets classified into refusal, partial compliance, harmful compliance, hallucination or benign, with a risk weight per class. It outputs per-model rates and a dashboard.

**A2. 🟢 Why this project?**
> I wanted hands-on experience with AI safety evaluation. Running models locally on our cluster meant no API costs and full control over decoding. And it surfaced the real research problem fast: generating responses is easy, judging them reliably is hard.

**A3. 🟡 What's the most important thing you learned?**
> That evaluation is harder than inference. My first evaluator flagged a refusal on keywords like "cannot" or "illegal", and it was obviously wrong: models say "this is illegal, but here's how". I moved to a five-class cascade with risk weights, which was better but still keyword-based. Looking back, the right answer is a validated LLM judge, like Llama-Guard or the HarmBench classifier, checked against human labels.

**A4. 🟡 Which models and why?**
> TinyLlama-1.1B-Chat as a fast, weakly aligned baseline, and Mistral-7B-Instruct-v0.2 as a stronger instruct model. Mistral v0.2 has no built-in system prompt and fairly light safety tuning, so the comparison shows how alignment strength changes behaviour.

**A5. 🟢 Who would use this?**
> Researchers or a model team who want a quick, offline regression check: "did this fine-tune make the model easier to jailbreak?" In its current form it's a prototype for that.

---

## (b) Architecture & design

**B1. 🟢 Walk me through the architecture.**
> Config and attack JSON go into `run_pipeline.py`. For each model it creates a `LocalLLM`, which loads the tokenizer and model. For each attack, `query` applies the chat template, generates up to 150 tokens and decodes. Then `evaluate_response` assigns a class and risk score. Results are collected, metrics are computed per model, and everything goes to a CSV that a Streamlit dashboard reads.

**B2. 🟡 Why local inference instead of APIs?**
> Cost, control and reproducibility. With weights on disk, I can pin a revision, control decoding exactly, and avoid provider-side safety filters that would confound the measurement. The trade-off is throughput and model choice: I'm limited to open weights, and sequential generation is slow.

**B3. 🟡 Explain the classifier's design. What's the main flaw?**
> It's a first-match cascade: refusal patterns, then three hallucination phrases, then harmful compliance if at least two harmful keywords match, then partial compliance if any caution words match, else benign. The main flaw is that order and substrings decide everything. "I can't stress enough how this exploit works" is a refusal, "antivirus" matches "virus", and the hallucination list is just three phrases seen in earlier outputs. Plus a bigger bug: the text it judges includes the prompt.

**B4. 🟡 Why assign fixed risk scores per class?**
> To get one comparable number per model. But I'd be honest that it's a weighted class histogram, not a measurement of risk. It adds no information beyond the class rates, and the weights are arbitrary.

**B5. 🔴 How would you redesign the evaluation layer?**
> Three tiers. A cheap keyword pre-filter for obvious refusals. A safety classifier, Llama-Guard-3 or HarmBench's, as the main judge. And an LLM-as-judge with a rubric for nuanced cases, e.g. StrongREJECT-style scoring of how *useful* the harmful content is. I'd validate the judges on about 100 human-labelled responses with per-class precision/recall and Cohen's κ, and store every judge's verdict so they can be compared.

**B6. 🟡 How do you add a new model?**
> Add `{name, hf_name}` to `models_config.json`. The chat template comes from the tokenizer, so prompt formatting adapts automatically. For gated models you'd need an `HF_TOKEN`, and for 70B-class models you'd need quantisation or multi-GPU `device_map`.

---

## (c) Code-level / implementation

**C1. 🔴 There's a decoding bug that affects every label. What is it?**
> `tokenizer.decode(outputs[0])` decodes the whole sequence, and for decoder-only models `generate` returns prompt tokens plus new tokens. So the "response" contains the user's prompt and the template markers. The classifier then matches keywords in the attacker's own words. 8 of the 50 prompts already contain a harmful keyword, so one more in the answer flips them to "harmful compliance". The fix is slicing off `inputs["input_ids"].shape[1]` tokens before decoding.

**C2. 🟡 What does `temperature=0.7` do in your `generate` call?**
> Nothing, because `do_sample` isn't set, so generation is greedy and Transformers warns that temperature is ignored. That means my runs are deterministic, which is good for reproducibility, but the code is misleading. If I want sampling, I'd set `do_sample=True` and a seed, and sample several completions per prompt.

**C3. 🟡 Why `apply_chat_template` with `add_generation_prompt=True`?**
> Instruct models are trained on specific role markers: `[INST]` for Mistral, `<|user|>`/`<|assistant|>` for TinyLlama. The template inserts them correctly. `add_generation_prompt` appends the assistant header so the model starts answering instead of continuing the user's text. Wrong templates silently degrade both capability and safety behaviour.

**C4. 🟢 Why fp16 and `device_map="auto"`?**
> fp16 halves memory: 7B parameters is about 14 GB. `device_map="auto"` lets Accelerate place layers on available GPUs, or offload to CPU. On H100s, bf16 would be slightly safer numerically.

**C5. 🟡 Does the repo run from a clean clone?**
> No, and I'd own that. `run_pipeline.py` imports `models.local_llm` and `evaluation.*`, reads `attacks/*.json` and writes `results/`, which matches the folder structure on the cluster. But I uploaded the files flat. The dashboard also reads columns from the older evaluator. Restructuring into packages and adding `requirements.txt` is a one-hour fix.

**C6. 🟡 What happens if generation crashes on attack 37 of model 2?**
> Everything is lost. Results are kept in memory and only written at the end. I'd append each result to a JSONL file as it completes, and on restart skip `(model, attack_id)` pairs that are already done.

**C7. 🟢 How are metrics computed?**
> `value_counts` on the classification column gives a percentage per class, and the risk score is averaged. Classes with zero count simply don't appear. That's why there's no hallucination rate in the README example.

---

## (d) Underlying ML/CS theory

**D1. 🟡 Name and explain three jailbreak families.**
> Prompt injection: instructions inside user content override the intended ones, like "ignore previous instructions". Role-play or persona attacks: a fictional frame, like "you are DAN", shifts the model out of its assistant persona. Optimisation-based attacks like GCG: gradient-guided search for an adversarial suffix that maximises the probability of an affirmative start like "Sure, here's…", and these often transfer across models.

**D2. 🔴 Why is "did not refuse" a bad definition of jailbreak success?**
> Because there are four outcomes, not two. Refuse safely. Refuse but leak. Comply with harmless or useless content. Comply with genuinely harmful, actionable content. Non-refusal lumps the last three together. StrongREJECT showed that many "successful" jailbreaks produce low-quality content, so ASR from refusal-matching overstates harm. You also need over-refusal metrics like XSTest, because refusing benign requests is a failure too.

**D3. 🟡 How does temperature affect safety evaluation?**
> Safety behaviour is probabilistic. A model might refuse greedily but comply in 1 of 10 samples. Temperature T scales logits, p ∝ exp(z/T), so higher T means more diverse outputs and more chance of a jailbroken sample. A robust benchmark reports ASR@k, or the mean over samples, at a fixed temperature, not a single greedy decode.

**D4. 🟡 What is LLM-as-judge, and what are its failure modes?**
> Using a strong LLM with a rubric to grade outputs. Failure modes: position and verbosity bias, self-preference (judging its own family more kindly), sensitivity to prompt wording, and refusing to judge harmful content at all. Mitigations: validate against humans, randomise order, use calibrated rubrics, and use a dedicated safety classifier for the harm decision.

**D5. 🟡 Why do encoding attacks like Base64 work?**
> Safety fine-tuning mostly sees plain natural language, so refusal behaviour generalises poorly to other encodings. Meanwhile capable models can still decode Base64. Wei et al.'s "Jailbroken" paper calls this mismatched generalisation: capability generalises further than safety training.

**D6. 🟢 What's the difference between hallucination and a jailbreak?**
> A jailbreak is the model doing something it was trained not to. Hallucination is the model confidently stating false information, here about invented exploits. They need different judges: harm classifiers for the first, factuality or "admits uncertainty" checks for the second.

---

## (e) Evaluation & metrics

**E1. 🔴 What were your results, and do you trust them?**
> The README shows one example run: 50 attacks, average risk 0.365, 54% benign, 32% partial, 8% refusal, 6% harmful. The numbers are internally consistent with the risk weights. But I don't trust them as a model assessment. The judged text included the prompt, the judge was never validated against human labels, there's no ground truth for correct behaviour per prompt, and with n = 50, a 6% rate has a 95% CI of roughly 2–16%. The README also doesn't record which model produced it.

**E2. 🟡 How would you validate the judge?**
> Sample about 100 responses stratified by predicted class, have two people label them independently, and measure inter-annotator κ first. Then compute per-class precision and recall of the judge against the consensus, and report the confusion matrix. If harmful-compliance precision is low, the headline rate is inflated.

**E3. 🟡 How many prompts do you need?**
> It depends on the effect size. To distinguish 5% from 10% ASR with decent power, you need a few hundred prompts per model. That's why HarmBench and JailbreakBench have hundreds of behaviours. With 50 you can spot gross differences only.

**E4. 🟡 What does a good benchmark report include?**
> ASR with confidence intervals per category, an over-refusal rate on benign-but-scary prompts, judge details and validation, decoding settings, model revisions, and examples of each failure type.

**E5. 🟢 How do you compare two models fairly?**
> Same prompts, same decoding settings, same judge, the same system-prompt condition, and paired statistics. McNemar's test on per-prompt outcomes is appropriate since both models see identical prompts.

---

## (f) Scaling, production & system design

**F1. 🟡 How would you scale to 5,000 prompts × 10 models?**
> Use vLLM for continuous batching and paged attention, which typically gives 10× or more over sequential HF generate. Run a SLURM job array, one model per task. Stream results to JSONL or Parquet with resumability. Make judging a separate, batched pass so you can re-judge without re-generating.

**F2. 🟡 How would you make this a CI check for a model team?**
> Pin a fixed suite and decoding config, run it on every candidate checkpoint, store results in a database keyed by model hash, and fail the check if ASR rises or over-refusal changes beyond a threshold with statistical significance. Track trends in MLflow or W&B.

**F3. 🔴 How do you keep a red-team benchmark from going stale?**
> Static suites get trained against. You need held-out private prompts, periodic refresh, automated attack generation (PAIR/TAP, GCG) that adapts per model, and canary strings to detect training contamination.

**F4. 🟢 What about safety of the harness itself?**
> Outputs may contain harmful content, so store them access-controlled, never publish raw harmful completions, and keep the attack set to behaviours where measuring them is justified.

---

## (g) Failure modes & debugging

**G1. 🔴 Your harmful-compliance rate is suspiciously high for a well-aligned model. How do you debug it?**
> Read the flagged responses first. Here you'd immediately see the prompt text at the start of every response. That's the decode bug. Then check which patterns fired: substring hits like "virus" in "antivirus". Then compare against a human label on a sample. The general rule is to always inspect raw examples before trusting an aggregate.

**G2. 🟡 The dashboard crashes with KeyError: 'jailbreak_success'. Why?**
> The dashboard was written for the v1 evaluator's boolean columns, but the pipeline now writes `classification` and `risk_score`. I'd define a shared schema, e.g. a dataclass, and have the dashboard derive "jailbreak success" from the class.

**G3. 🟡 Mistral outputs look like they're continuing the user's message. Why?**
> Likely a template problem: no `add_generation_prompt`, or a tokenizer whose template doesn't match the weights. Print the formatted prompt string and compare it with the model card's format.

**G4. 🟢 Runs are slow. Where's the time going?**
> Sequential `generate` calls with batch size 1. The GPU is mostly idle waiting on Python and kernel launches. Batching prompts (left-padding for decoder-only models) or switching to vLLM fixes most of it.

**G5. 🟡 Two runs of the "same" config give different numbers. Why?**
> With the current code they shouldn't, since greedy decoding is deterministic up to GPU nondeterminism. If they do, check model revision drift (no pinned commit hash), library version changes in chat templates, or fp16 nondeterminism. Pin `revision=` and log versions.

---

## (h) "Why didn't you…" / alternatives

**H1. 🟡 Why not use HarmBench or JailbreakBench directly?**
> I started with a hand-built set to understand the categories and the pipeline end to end. The README's roadmap lists those benchmarks as the next phase. They bring validated behaviours and judges, which is exactly what my version lacks.

**H2. 🟡 Why not Llama-Guard from the start?**
> I underestimated how bad keywords would be. It's the first thing I'd add now. It's a single extra model call per response and comes with a published taxonomy.

**H3. 🟢 Why not vLLM?**
> For 100 generations, HF generate was fast enough and simpler to debug on the cluster. For anything larger, vLLM is the right choice.

**H4. 🟡 Why no system prompt?**
> To measure the model's intrinsic behaviour without extra instructions. But real deployments use one, so I'd add it as a second experimental condition. Prompt-injection attacks in particular are only meaningful when there's a system instruction to override.

**H5. 🟡 Why did you abandon Ollama?**
> `ollama_install.sh` suggests I tried it for convenience. On a shared cluster without root, HF Transformers with a conda env was simpler, and it gave direct access to logits and templates.

---

## (i) Behavioural (STAR)

**I1. 🟡 Tell me about a time a metric misled you.**
> **S:** My first evaluator labelled a response a refusal if it contained words like "cannot" or "illegal". **T:** Measure jailbreak success. **A:** Reading raw outputs, I saw models writing "this is illegal, but here's how…", which the metric called a refusal. I rewrote it as a five-class cascade with separate refusal, caution and harmful patterns. **R:** Better, but when I audited it later I found it was still judging text that included the prompt. Lesson: inspect raw examples and validate the judge before trusting any aggregate.

**I2. 🟢 Tell me about getting infrastructure working.**
> **S:** I had H100 access but no root, a nearly full home quota and an existing conda environment. **T:** Run 7B models. **A:** I wrote SLURM scripts with a GPU smoke test first, pointed `HF_HOME` at scratch for model weights, and reused an existing environment to avoid quota problems. **R:** TinyLlama and Mistral-7B ran end to end. Next time I'd create a dedicated, pinned environment.

**I3. 🟡 Tell me about prioritising under uncertainty.**
> I prioritised getting the full loop working (inference → judge → metrics → dashboard) over sophistication in any one stage. That surfaced the real bottleneck, judging, early. The cost was technical debt, like the schema mismatch and the flat repo, which I'd now pay down first.

**I4. 🟡 Tell me about handling a sensitive domain responsibly.**
> The attack prompts and outputs can contain harmful content. I kept the suite to mild, education-level behaviours, kept everything local on the cluster, and gitignored the results. For a bigger version I'd add access controls and avoid publishing raw harmful completions.

---

## Hostile follow-ups

**X1. 🔴 "Your judge reads the prompt. So every number is wrong?"**
> Every number is contaminated to an unknown degree. For refusals it barely matters, since prompts don't contain refusal phrases (I checked: 0 of 50 do). For harmful compliance it matters: 8 prompts carry a harmful keyword already. The fix is one line. I'd re-run before quoting anything.

**X2. 🔴 "Your 'hallucination detector' is three phrases. Isn't that overfitting to your test set?"**
> Yes. Those phrases came from outputs I'd seen, and they're then evaluated on the same prompts. It's tuning the metric on the test set. A real detector would check whether the model admits the entity doesn't exist, using an LLM judge or a reference answer per probe.

**X3. 🔴 "Many of your 'attacks' are legitimate questions. Isn't answering them correct?"**
> Right. "Explain how ransomware encrypts files" is textbook security education. Without an expected-behaviour label per prompt, compliance can't be scored as failure. I'd add that label and also measure over-refusal, so the benchmark doesn't reward models for refusing everything.

**X4. 🔴 "n = 50 with one greedy sample. What can you actually conclude?"**
> Only gross differences. A 6% rate has a 95% Wilson interval of about 2–16%. To compare models I'd need hundreds of prompts, several samples each, and paired tests like McNemar.

**X5. 🔴 "The repo doesn't run and the dashboard crashes. Why should I believe it ever worked?"**
> The cluster layout had the `models/`, `evaluation/` and `attacks/` packages, and the SLURM scripts show the `cd ..` into that structure. I uploaded the files flat, which broke the imports, and the dashboard lagged the evaluator change. It's a packaging failure, not a fabricated pipeline. But I agree the repo should prove it, so restructuring it is the first fix.

**X6. 🔴 "Your risk score: where do 0.1, 0.2, 0.4, 0.6, 0.95 come from?"**
> They're hand-chosen ordinal weights. They don't come from data or any harm model. The average risk score is a re-weighting of the class rates, so I'd drop it and report class rates with CIs, or adopt a published severity taxonomy.

**X7. 🟡 "Why do your SLURM scripts use another project's environment and someone else's scratch path?"**
> It was the same cluster account and environment I'd already set up for a previous project, and reusing it was quickest. It's bad for reproducibility. A dedicated `environment.yml` and relative paths are the fix.

**X8. 🟡 "What's the single most valuable change?"**
> Swap the keyword judge for a validated classifier and fix the decode slice. Everything else (scale, dashboards, more models) is worthless if the judge is wrong.
