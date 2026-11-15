# What to do next — simple guide

We are building a support-ticket assistant. We will first measure the original model, then teach it our task, then check whether it improved. The dashboard comes after those checks.

## Step 1: Finish and review the data

**What happens:** We prepare tickets with correct JSON answers in three files:

- `data/train.jsonl`: examples the model learns from.
- `data/validation.jsonl`: examples used to check progress and choose training settings.
- `data/test.jsonl`: examples reserved for the final comparison.

**Your action:** After the supervisor finishes, read examples from each category and check that their labels and impact summaries make sense. The data is AI-authored fictional data; AI review is not human review.

**Purpose:** Incorrect answers in the dataset teach the model incorrect behavior. Related versions of one ticket must stay in the same split so the final check stays useful.

**Ready when:** Dataset checks pass, reviewer findings are resolved, and we have recorded the dataset version. A small pilot dataset can start the experiment; it does not prove real-world quality.

## Step 2: Prepare the Colab notebook

**What happens:** I implement a notebook in this project. You open it in Google Colab and select an available GPU runtime. The notebook downloads Qwen onto Colab's machine, not your laptop.

**Your action:** Open the notebook when it is ready and connect to a GPU. You do not need to buy or own a GPU. Free GPU availability can vary.

**Purpose:** Colab supplies the computing power for the experiment.

**Ready when:** The notebook can load the model and produce one prediction. It must also check that the prompt, ticket, and expected answer fit within the training length. We must not silently cut answers off.

**Current status:** `training/baseline_colab.ipynb` and `evaluation/run_model.py` are prepared. Follow [the Colab guide](COLAB_GUIDE.md). They still need their first real GPU smoke run.

## Step 3: Measure the original model

**What happens:** Run the original Qwen model on validation tickets and save its answers. Score those answers using our evaluator.

**Your action:** Run the baseline section of the notebook and keep its output files.

**Purpose:** These starting scores are the baseline. Without them, we cannot tell whether fine-tuning added value.

**Ready when:** We have saved raw predictions, scores, the exact model version, and run settings. Check routing accuracy, JSON validity, and environment extraction. Impact summaries need separate human review.

## Step 4: Fine-tune the model

**What happens:** The notebook shows the model tickets and correct answers from `train.jsonl`. LoRA trains a small adapter containing our learned changes; it does not replace all the original weights.

**Your action:** Run the training section. Save checkpoints outside Colab's temporary runtime so work is not lost if it disconnects.

**Purpose:** Teach the model to follow our output format and ticket-labeling rules more consistently.

**Ready when:** A short training run completes and the adapter can be saved and loaded. Start small before spending more GPU time.

## Step 5: Choose the best saved version

**What happens:** Compare saved checkpoints on `validation.jsonl`. If needed, adjust training settings and repeat using training data only.

**Your action:** Review validation results and representative mistakes with me.

**Purpose:** Choose settings without using the final test questions to guide training.

**Ready when:** We choose one adapter and freeze the prompt and prediction settings. Do not tune based on test results.

## Step 6: Run the final comparison

**What happens:** Run both the original model and the selected adapter on the same tickets in `test.jsonl`.

**Your action:** Run the final evaluation section and inspect the comparison table and mistakes.

**Purpose:** Find out whether fine-tuning helped on examples held out from training and tuning.

**Ready when:** We have actual scores and reviewed impact summaries. Report results honestly even if fine-tuning does not help. File checks are not model scores, and generated confidence numbers are not reliable probabilities by default.

## Step 7: Build the application

**What happens:** React lets a user enter a ticket. Python/FastAPI loads the original model plus our adapter and returns a prediction. Store examples and evaluation reports in files; no database is needed.

**Your action:** Run the app locally and try several tickets.

**Purpose:** Show how the trained model solves a practical workflow.

**Ready when:** The app handles valid tickets, unclear requests, and errors. The model loads once; it does not retrain for each request. Local inference downloads weights onto the computer running Python and uses that computer's hardware.

## Step 8: Publish the portfolio project

**What happens:** Publish code and setup instructions on GitHub, the adapter on Hugging Face, and a demo recording and real evaluation results in your portfolio.

**Your action:** Create/select the publishing accounts when needed. Keep tokens in environment variables or notebook secrets, never in chat or committed code.

**Purpose:** Let reviewers understand your engineering decisions and reproduce the project.

**Ready when:** Another person can follow the instructions, download the model and adapter, and run the app. Hosting files on Hugging Face does not automatically create a live prediction API. Decide optional website/backend hosting separately.

## What you should do right now

You confirmed the data looks good and ran/downloaded the original validation baseline. The three-epoch pilot is complete and checkpoint-32 (epoch 2) is selected. Its adapter/results are saved locally. Next open `training/final_test_colab.ipynb` and follow [the final test guide](FINAL_TEST_GUIDE.md) to compare both models on the held-out test set. No test results have been generated locally.
