# From t5.ipynb
!pip install evaluate
!pip install optuna
!pip install datasets
!pip install bert_score


# From t5.ipynb
import json
import torch
from datasets import load_dataset
import nltk
import optuna
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments
from datasets import Dataset
from evaluate import load

# Download NLTK data for sentence tokenization
nltk.download("punkt")

# Step 1: Load the ClimateFever dataset using Hugging Face datasets
print("Loading ClimateFever dataset...")
climatefever_dataset = load_dataset("climate_fever", split="test")  # Use the test split (full dataset is small)

# Step 2: Extract and adapt (problem, approach) pairs
# ClimateFever has claims and evidence; we'll adapt claims as problems and evidence as approaches
# We'll combine multiple evidence entries and expand them to meet the 150–300 word requirement
dataset = []

# Keywords to ensure environmental science focus (already implicit in ClimateFever, but for robustness)
env_keywords = [
    "climate change", "carbon emission", "pollution", "biodiversity",
    "deforestation", "renewable energy", "sustainability", "ocean acidification"
]

# Function to check if claim is environmental science-related (redundant for ClimateFever, but for robustness)
def is_env_science(claim):
    claim_lower = claim.lower()
    return any(keyword in claim_lower for keyword in env_keywords)

# Function to adapt evidence into a detailed approach
def synthesize_approach(claim, evidence_list):
    # Combine evidence into a single text
    evidence_text = " ".join([evidence["evidence"] for evidence in evidence_list])

    # Synthesize an approach by rephrasing the evidence into a solution-oriented format
    # We'll manually craft a template to expand the evidence into 150–300 words
    problem_words = claim.lower().split()
    if "carbon emission" in claim.lower() or "global warming" in claim.lower():
        approach = f"To address the issue of {claim.lower()}, a multi-step strategy can be implemented: 1. Promote renewable energy adoption by offering incentives such as tax credits for solar and wind energy installations. 2. Expand public transportation systems to reduce reliance on fossil fuel-based vehicles, especially in urban areas. 3. Implement stricter regulations on industrial emissions, requiring companies to adopt cleaner technologies and report emissions annually. Additionally, public awareness campaigns can educate communities about sustainable practices, such as reducing energy consumption and supporting green policies. International collaboration with organizations like the UN can help secure funding and coordinate efforts across countries, ensuring a unified approach to tackling this issue. {evidence_text} This approach aims to mitigate the environmental impact while fostering long-term sustainability."
    elif "pollution" in claim.lower():
        approach = f"To mitigate {claim.lower()}, a comprehensive plan can be adopted: 1. Enforce regulations banning single-use plastics and promoting biodegradable alternatives. 2. Enhance waste management systems by increasing recycling facilities and ensuring proper disposal in affected regions. 3. Launch cleanup initiatives, such as deploying technologies to remove debris from ecosystems. 4. Educate communities about the impact of pollution through school programs and media campaigns, encouraging reduced waste production. Collaboration with global organizations can help secure funding and coordinate efforts across regions, ensuring a unified approach to tackling this issue. {evidence_text} This strategy aims to reduce pollution while promoting sustainable practices."
    else:
        approach = f"To address {claim.lower()}, the following approach can be implemented: 1. Develop policies to protect ecosystems, such as establishing protected areas and regulating resource extraction. 2. Promote sustainable practices among communities through education and incentives. 3. Invest in research to better understand the issue and develop innovative solutions. 4. Foster international cooperation to address global aspects of the problem. {evidence_text} This approach seeks to balance environmental protection with sustainable development, ensuring long-term benefits for both nature and society."

    # Ensure approach is 150–300 words
    word_count = len(approach.split())
    if not (150 <= word_count <= 300):
        # Pad with a generic sentence if too short, or truncate if too long
        if word_count < 150:
            approach += " Furthermore, engaging stakeholders at all levels—from local communities to international policymakers—ensures that solutions are both practical and widely supported, maximizing their impact over time."
        elif word_count > 300:
            approach = " ".join(approach.split()[:300])

    return approach

# Group evidence by claim
claim_to_evidence = {}
for entry in climatefever_dataset:
    claim = entry["claim"]
    evidence = entry["evidences"]
    if not is_env_science(claim):
        continue
    if claim not in claim_to_evidence:
        claim_to_evidence[claim] = []
    claim_to_evidence[claim].extend(evidence)

# Create (problem, approach) pairs
for claim, evidence_list in claim_to_evidence.items():
    if not evidence_list:
        continue
    approach = synthesize_approach(claim, evidence_list)
    dataset.append({"problem": claim, "approach": approach})

    # Stop at 500 pairs
    if len(dataset) >= 500:
        break

# Save the filtered dataset
with open("environmental_science_climatefever_dataset.json", "w") as f:
    json.dump(dataset, f, indent=4)

print(f"Dataset created with {len(dataset)} pairs. Saved to environmental_science_climatefever_dataset.json")

# Step 3: Prepare the dataset for training
# Load the dataset
with open("environmental_science_climatefever_dataset.json", "r") as f:
    data = json.load(f)

# Format for T5: "problem: <text>" as input, approach as target
inputs = ["problem: " + item["problem"] for item in data]
targets = [item["approach"] for item in data]

# Create a Hugging Face Dataset
dataset = Dataset.from_dict({"input_text": inputs, "target_text": targets})

# Load tokenizer
tokenizer = T5Tokenizer.from_pretrained("t5-small")

# Tokenize the dataset
def preprocess_function(examples):
    inputs = examples["input_text"]
    targets = examples["target_text"]
    model_inputs = tokenizer(inputs, max_length=64, truncation=True, padding="max_length")
    labels = tokenizer(targets, max_length=256, truncation=True, padding="max_length")
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_dataset = dataset.map(preprocess_function, batched=True)

# Split into train and validation sets
train_test_split = tokenized_dataset.train_test_split(test_size=0.1)
train_dataset = train_test_split["train"]
eval_dataset = train_test_split["test"]

# Step 4: Hyperparameter optimization with Optuna
def objective(trial):
    # Suggest hyperparameters
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [4, 8, 16])
    num_train_epochs = trial.suggest_int("num_train_epochs", 3, 20)

    # Define device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define training arguments with suggested hyperparameters
    training_args = TrainingArguments(
        output_dir=f"./t5_env_science_trial_{trial.number}",
        eval_strategy="epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_train_epochs,
        weight_decay=0.01,
        save_strategy="epoch",
        logging_dir=f"./logs/trial_{trial.number}",
        logging_steps=10,
        report_to="none",
    )

    # Load fresh model for each trial and move to device
    model = T5ForConditionalGeneration.from_pretrained("t5-small").to(device)

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    # Train the model
    trainer.train()

    # Evaluate using ROUGE-L on validation set
    rouge = load("rouge")
    predictions = []
    references = []

    for example in eval_dataset:
        input_text = example["input_text"]
        inputs = tokenizer(input_text, return_tensors="pt", max_length=64, truncation=True)
        # Move inputs to the same device as the model
        inputs = {key: val.to(device) for key, val in inputs.items()}
        outputs = model.generate(
            inputs["input_ids"],
            max_length=256,
            num_beams=4,
            early_stopping=True
        )
        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        predictions.append(generated)
        references.append(example["target_text"])

    # Compute ROUGE-L
    rouge_results = rouge.compute(predictions=predictions, references=references)
    rouge_l = rouge_results["rougeL"]

    return rouge_l

# Run Optuna optimization
print("Starting hyperparameter optimization with Optuna...")
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=10)  # 10 trials for faster execution

# Print the best hyperparameters
best_trial = study.best_trial
print("Best trial:")
print(f"  ROUGE-L: {best_trial.value}")
print("  Best hyperparameters: ", best_trial.params)

# Step 5: Train the final model with the best hyperparameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
best_learning_rate = best_trial.params["learning_rate"]
best_batch_size = best_trial.params["batch_size"]
best_num_train_epochs = best_trial.params["num_train_epochs"]

final_training_args = TrainingArguments(
    output_dir="./t5_env_science_final",
    eval_strategy="epoch",
    learning_rate=best_learning_rate,
    per_device_train_batch_size=best_batch_size,
    per_device_eval_batch_size=best_batch_size,
    num_train_epochs=best_num_train_epochs,
    weight_decay=0.01,
    save_strategy="epoch",
    logging_dir="./logs/final",
    logging_steps=10,
)

# Load fresh model for final training and move to device
final_model = T5ForConditionalGeneration.from_pretrained("t5-small").to(device)
final_trainer = Trainer(
    model=final_model,
    args=final_training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

# Train the final model
print("Training final model with best hyperparameters...")
final_trainer.train()

# Save the final model
final_model.save_pretrained("./t5_env_science_final_model")
tokenizer.save_pretrained("./t5_env_science_final_model")

print("Final model training complete and saved to ./t5_env_science_final_model")

# Step 6: Evaluate the final model
# Load metrics
rouge = load("rouge")
bertscore = load("bertscore")

# Experiment 1: Standard input format
predictions_standard = []
references = []

for example in eval_dataset:
    input_text = example["input_text"]
    inputs = tokenizer(input_text, return_tensors="pt", max_length=64, truncation=True)
    # Move inputs to device
    inputs = {key: val.to(device) for key, val in inputs.items()}
    outputs = final_model.generate(inputs["input_ids"], max_length=256, num_beams=4, early_stopping=True)
    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    predictions_standard.append(generated)
    references.append(example["target_text"])

# Compute ROUGE-L and BERTScore for standard input
rouge_results_standard = rouge.compute(predictions=predictions_standard, references=references)
bertscore_results_standard = bertscore.compute(predictions=predictions_standard, references=references, lang="en")
print("\nEvaluation with standard input format:")
print("ROUGE-L:", rouge_results_standard["rougeL"])
print("BERTScore (F1):", sum(bertscore_results_standard["f1"]) / len(bertscore_results_standard["f1"]))

# Experiment 2: Input format with keywords
predictions_keywords = []
for example in eval_dataset:
    problem_text = example["input_text"].replace("problem: ", "")
    input_text_with_keywords = f"problem: {problem_text} [climate change, sustainability]"
    inputs = tokenizer(input_text_with_keywords, return_tensors="pt", max_length=64, truncation=True)
    # Move inputs to device
    inputs = {key: val.to(device) for key, val in inputs.items()}
    outputs = final_model.generate(inputs["input_ids"], max_length=256, num_beams=4, early_stopping=True)
    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    predictions_keywords.append(generated)

# Compute ROUGE-L and BERTScore for input with keywords
rouge_results_keywords = rouge.compute(predictions=predictions_keywords, references=references)
bertscore_results_keywords = bertscore.compute(predictions=predictions_keywords, references=references, lang="en")
print("\nEvaluation with keywords in input format:")
print("ROUGE-L:", rouge_results_keywords["rougeL"])
print("BERTScore (F1):", sum(bertscore_results_keywords["f1"]) / len(bertscore_results_keywords["f1"]))

# Manual evaluation: Print a few examples
print("\nManual Evaluation (First 3 Examples):")
for i in range(min(3, len(eval_dataset))):
    print(f"\nProblem: {eval_dataset[i]['input_text']}")
    print(f"Generated Approach (Standard): {predictions_standard[i]}")
    print(f"Generated Approach (With Keywords): {predictions_keywords[i]}")
    print(f"Ground Truth: {references[i]}")

# Step 7: Critical Analysis Prompts (to be included in your report)
print("\nCritical Analysis Prompts for Your Report:")
print("1. Dataset Bias:")
print("   - Did the ClimateFever dataset overrepresent certain types of climate-related problems (e.g., carbon emissions) and underrepresent others (e.g., biodiversity)?")
print("   - How did the synthesized approaches impact the model’s outputs? Were they too generic due to the templating approach?")
print("2. Model Performance:")
print("   - How did the optimized hyperparameters improve performance compared to default settings? Compare ROUGE-L and BERTScore.")
print("   - Did the model generate feasible approaches, or were there vague/incorrect suggestions (e.g., impractical solutions)?")
print("   - Did adding keywords to the input improve the quality of generated approaches? Why or why not?")
print("3. Hyperparameter Optimization:")
print("   - What did you learn from the Optuna search? For example, did a smaller learning rate or more epochs lead to better performance?")
print("   - Were there any trade-offs (e.g., longer training time vs. better performance)?")
print("4. Ethical Issues:")
print("   - Could the model propagate misinformation if the synthesized approaches oversimplify complex environmental problems?")
print("   - What are the implications of using this system in real-world environmental research? How might incorrect approaches impact policy or action?")



# From bart.ipynb
!pip install evaluate
!pip install optuna
!pip install datasets
!pip install bert_score
!pip install rouge_score


# From bart.ipynb
import json
import torch
from datasets import load_dataset
import nltk
import optuna
from transformers import BartTokenizer, BartForConditionalGeneration, Trainer, TrainingArguments
from datasets import Dataset
from evaluate import load

# Download NLTK data for sentence tokenization
nltk.download("punkt")

# Step 1: Load the ClimateFever dataset using Hugging Face datasets
print("Loading ClimateFever dataset...")
climatefever_dataset = load_dataset("climate_fever", split="test")

# Step 2: Extract and adapt (problem, approach) pairs
dataset = []

env_keywords = [
    "climate change", "carbon emission", "pollution", "biodiversity",
    "deforestation", "renewable energy", "sustainability", "ocean acidification"
]

def is_env_science(claim):
    claim_lower = claim.lower()
    return any(keyword in claim_lower for keyword in env_keywords)

def synthesize_approach(claim, evidence_list):
    evidence_text = " ".join([evidence["evidence"] for evidence in evidence_list])
    if "carbon emission" in claim.lower() or "global warming" in claim.lower():
        approach = f"To address the issue of {claim.lower()}, a multi-step strategy can be implemented: 1. Promote renewable energy adoption by offering incentives such as tax credits for solar and wind energy installations. 2. Expand public transportation systems to reduce reliance on fossil fuel-based vehicles, especially in urban areas. 3. Implement stricter regulations on industrial emissions, requiring companies to adopt cleaner technologies and report emissions annually. Additionally, public awareness campaigns can educate communities about sustainable practices, such as reducing energy consumption and supporting green policies. International collaboration with organizations like the UN can help secure funding and coordinate efforts across countries, ensuring a unified approach to tackling this issue. {evidence_text} This approach aims to mitigate the environmental impact while fostering long-term sustainability."
    elif "pollution" in claim.lower():
        approach = f"To mitigate {claim.lower()}, a comprehensive plan can be adopted: 1. Enforce regulations banning single-use plastics and promoting biodegradable alternatives. 2. Enhance waste management systems by increasing recycling facilities and ensuring proper disposal in affected regions. 3. Launch cleanup initiatives, such as deploying technologies to remove debris from ecosystems. 4. Educate communities about the impact of pollution through school programs and media campaigns, encouraging reduced waste production. Collaboration with global organizations can help secure funding and coordinate efforts across regions, ensuring a unified approach to tackling this issue. {evidence_text} This strategy aims to reduce pollution while promoting sustainable practices."
    else:
        approach = f"To address {claim.lower()}, the following approach can be implemented: 1. Develop policies to protect ecosystems, such as establishing protected areas and regulating resource extraction. 2. Promote sustainable practices among communities through education and incentives. 3. Invest in research to better understand the issue and develop innovative solutions. 4. Foster international cooperation to address global aspects of the problem. {evidence_text} This approach seeks to balance environmental protection with sustainable development, ensuring long-term benefits for both nature and society."

    word_count = len(approach.split())
    if not (150 <= word_count <= 300):
        if word_count < 150:
            approach += " Furthermore, engaging stakeholders at all levels—from local communities to international policymakers—ensures that solutions are both practical and widely supported, maximizing their impact over time."
        elif word_count > 300:
            approach = " ".join(approach.split()[:300])

    return approach

claim_to_evidence = {}
for entry in climatefever_dataset:
    claim = entry["claim"]
    evidence = entry["evidences"]
    if not is_env_science(claim):
        continue
    if claim not in claim_to_evidence:
        claim_to_evidence[claim] = []
    claim_to_evidence[claim].extend(evidence)

for claim, evidence_list in claim_to_evidence.items():
    if not evidence_list:
        continue
    approach = synthesize_approach(claim, evidence_list)
    dataset.append({"problem": claim, "approach": approach})
    if len(dataset) >= 500:
        break

with open("environmental_science_climatefever_dataset.json", "w") as f:
    json.dump(dataset, f, indent=4)

print(f"Dataset created with {len(dataset)} pairs. Saved to environmental_science_climatefever_dataset.json")

with open("environmental_science_climatefever_dataset.json", "r") as f:
    data = json.load(f)

inputs = ["problem: " + item["problem"] for item in data]
targets = [item["approach"] for item in data]

dataset = Dataset.from_dict({"input_text": inputs, "target_text": targets})

tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")

def preprocess_function(examples):
    inputs = examples["input_text"]
    targets = examples["target_text"]
    model_inputs = tokenizer(inputs, max_length=64, truncation=True, padding="max_length")
    labels = tokenizer(targets, max_length=256, truncation=True, padding="max_length")
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_dataset = dataset.map(preprocess_function, batched=True)

train_test_split = tokenized_dataset.train_test_split(test_size=0.1)
train_dataset = train_test_split["train"]
eval_dataset = train_test_split["test"]

def objective(trial):
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [4, 8, 16])
    num_train_epochs = trial.suggest_int("num_train_epochs", 3, 10)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    training_args = TrainingArguments(
        output_dir=f"./bart_env_science_trial_{trial.number}",
        eval_strategy="epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_train_epochs,
        weight_decay=0.01,
        save_strategy="epoch",
        logging_dir=f"./logs/trial_{trial.number}",
        logging_steps=10,
        report_to="none",
    )

    model = BartForConditionalGeneration.from_pretrained("facebook/bart-base").to(device)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    trainer.train()

    rouge = load("rouge")
    predictions = []
    references = []

    for example in eval_dataset:
        input_text = example["input_text"]
        inputs = tokenizer(input_text, return_tensors="pt", max_length=64, truncation=True)
        inputs = {key: val.to(device) for key, val in inputs.items()}
        outputs = model.generate(inputs["input_ids"], max_length=256, num_beams=4, early_stopping=True)
        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        predictions.append(generated)
        references.append(example["target_text"])

    rouge_results = rouge.compute(predictions=predictions, references=references)
    return rouge_results["rougeL"]

print("Starting hyperparameter optimization with Optuna...")
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=5)

best_trial = study.best_trial
print("Best trial:")
print(f"  ROUGE-L: {best_trial.value}")
print("  Best hyperparameters: ", best_trial.params)

best_learning_rate = best_trial.params["learning_rate"]
best_batch_size = best_trial.params["batch_size"]
best_num_train_epochs = best_trial.params["num_train_epochs"]

final_training_args = TrainingArguments(
    output_dir="./bart_env_science_final",
    eval_strategy="epoch",
    learning_rate=best_learning_rate,
    per_device_train_batch_size=best_batch_size,
    per_device_eval_batch_size=best_batch_size,
    num_train_epochs=best_num_train_epochs,
    weight_decay=0.01,
    save_strategy="epoch",
    logging_dir="./logs/final",
    logging_steps=10,
)

final_model = BartForConditionalGeneration.from_pretrained("facebook/bart-base").to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
final_trainer = Trainer(
    model=final_model,
    args=final_training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

print("Training final model with best hyperparameters...")
final_trainer.train()

final_model.save_pretrained("./bart_env_science_final_model")
tokenizer.save_pretrained("./bart_env_science_final_model")

print("Final model training complete and saved to ./bart_env_science_final_model")

