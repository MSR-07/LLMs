# LLMs 🚀

This repository contains experiments, code, and notes on **Large Language Models (LLMs)**.  
I use [Hugging Face Transformers](https://huggingface.co/transformers/) and [PyTorch](https://pytorch.org/) for most of the projects.

---

## ⚡ Setup

Clone the repository:

```bash
git clone https://github.com/USERNAME/LLMs.git
cd LLMs

## Create and activate a virtual environment:
python -m venv .env
# macOS / Linux
source .env/bin/activate
# Windows
.env\Scripts\activate
## Install dependencies:
pip install -r requirements.txt

🔬 Experiments
1. Sentiment Analysis Pipeline

Using Hugging Face’s built-in pipeline to classify text as positive or negative.

from transformers import pipeline

classifier = pipeline("sentiment-analysis")
result = classifier("I've been waiting for a HuggingFace course my whole life.")
print(result)
## Output Example:

[{'label': 'POSITIVE', 'score': 0.9998}]

2. Fine-Tuning Experiments

Train a pre-trained model on a custom dataset.

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer

# Load dataset
dataset = load_dataset("imdb")
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

def tokenize(batch):
    return tokenizer(batch["text"], padding=True, truncation=True)

dataset = dataset.map(tokenize, batched=True)
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    per_device_train_batch_size=16,
    num_train_epochs=2,
    save_total_limit=2
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"].shuffle().select(range(5000)),
    eval_dataset=dataset["test"].shuffle().select(range(2000))
)

trainer.train()

3. Model Evaluations

Compare multiple models for performance metrics like accuracy and F1-score.

from sklearn.metrics import accuracy_score, f1_score

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="weighted")
    }


📂 Project Structure
LLMs/
│── notebooks/        # Jupyter notebooks for experiments
│── scripts/          # Python scripts for training/evaluation
│── data/             # Sample datasets (optional)
│── requirements.txt  # Dependencies
│── .gitignore
│── README.md

🛠 Requirements

transformers

datasets

torch

numpy<2

scikit-learn

jupyter

🤝 Contributing

Contributions, suggestions, and pull requests are welcome!
Feel free to fork this repo and experiment.

📜 License

This project is licensed under the MIT License.


---

If you want, I can also **write a matching `requirements.txt`** with the correct versions of `torch`, `transformers`, `numpy`, etc., so others can run your repo without errors.  

Do you want me to do that next?
