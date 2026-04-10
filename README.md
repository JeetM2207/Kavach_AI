# 🛡️ Kavach AI: Two-Stage SMS Fraud Detector

[![Model](https://img.shields.io/badge/Model-Llama--3.2--1B-blue)](#)
[![Library](https://img.shields.io/badge/Library-HuggingFace-yellow)](#)
[![Framework](https://img.shields.io/badge/Framework-PyTorch-ee4c2c)](#)
[![Alignment](https://img.shields.io/badge/Alignment-DPO-green)](#)

Kavach AI is an edge-optimized, two-stage NLP pipeline designed to detect nuanced Indian financial and phishing SMS scams. Built to overcome the high false-positive rates of traditional keyword-based filters, this system utilizes a high-speed gatekeeper model and a fine-tuned generative reasoning engine to accurately classify and explain fraudulent messages.

## 📌 The Problem
Traditional SMS spam filters in India often rely on simple keyword matching (e.g., flagging messages containing "INR", "Bank", or "Account"). This leads to extreme alert fatigue, as legitimate banking alerts and safe transactional messages are routinely blocked. Kavach AI solves this by introducing deep contextual reasoning to differentiate between a safe transaction and a socially engineered phishing attempt.

## 🏗️ Architecture: The Two-Stage Pipeline

To balance inference speed and analytical depth, Kavach AI uses a Two-Stage architecture:

1. **Stage 1 (The Gatekeeper):** A lightweight `AI4Bharat IndicBERT` model. It acts as a high-speed router, instantly classifying obvious spam or clearly safe messages.
2. **Stage 2 (The Reasoning Engine):** A `Llama-3.2-1B-Instruct` model fine-tuned via QLoRA. When the Gatekeeper flags a message as "suspicious" or "borderline," the Llama model takes over to perform forensic analysis and generate a structured explanation of *why* the message is a scam.

## 🧠 Training Methodology (SFT + DPO)

The Stage 2 Llama model was aligned using a state-of-the-art RLHF pipeline:

* **Supervised Fine-Tuning (SFT):** The base model was fine-tuned on a custom dataset using Parameter-Efficient Fine-Tuning (PEFT) with LoRA adapters to adapt to the SMS domain.
* **Direct Preference Optimization (DPO):** To prevent the model from writing rambling paragraphs, DPO was applied using a dataset of `chosen` (highly structured, bulleted explanations) and `rejected` (unstructured or hallucinated) responses. 
    * **Metric Achieved:** The DPO training successfully dropped validation loss to **0.0012**, resulting in a model that natively outputs clean, readable, and highly accurate forensic breakdowns.

## 🚀 Quick Start

### Installation
```bash
pip install torch transformers peft accelerate

Inference Code

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Load Tokenizer
tokenizer = AutoTokenizer.from_pretrained("Jeet2207/kavach-llama-dpo")

# Load Base Model & DPO Adapters
base_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-1B-Instruct", 
    torch_dtype=torch.float16, 
    device_map="auto"
)
model = PeftModel.from_pretrained(base_model, "Jeet2207/kavach-llama-dpo")
model.eval()

# Generate Explanation
sms = "Dear customer, your SBI account is blocked. Update PAN: [http://sbi-kyc.co](http://sbi-kyc.co)"
prompt = f"Below is an SMS. Explain why it is a scam.\n\n### SMS:\n{sms}\n\n### Explanation:\n"

inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=250, do_sample=False)
    
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## 📊 Example Output
Input SMS:

"Sir aapka sim band ho jayega 24 ghante mein: http://sim-verify.in"

Kavach AI (DPO-Aligned) Output:

This SMS is a scam because it tricks the recipient into revealing their SIM card details, which can be used for identity theft or other malicious purposes. Here's why:

Request for SIM verification: Telecom operators never send unverified HTTP links for KYC or SIM verification.

False Urgency: The 24-hour deadline is a psychological tactic designed to induce panic and force a rapid, unthinking click.

Suspicious URL: The domain sim-verify.in is unofficial and not affiliated with any legitimate Indian telecom provider.

⚠️ Limitations & Future Work
While DPO drastically improved the structural readability of the explanations, testing revealed that the lightweight 1B parameter model can occasionally over-fit to the preference dataset's structure. In rare edge cases, it may hallucinate specific scam triggers (e.g., claiming the SMS "asked for a name" when it only provided a phone number). Future iterations will focus on expanding the DPO dataset to heavily penalize these specific hallucination patterns.

👨‍💻 Author
Jeet Manseta B.Tech Computer Science & Engineering | The LNM Institute of Information Technology (LNMIIT)


