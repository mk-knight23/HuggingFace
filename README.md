# Text Summarizer - AI-Powered Dialogue Summarization

A fine-tuned PEGASUS model for generating concise summaries from conversational text using the SAMSum dataset.

## 🚀 Quick Start

### Prerequisites
```bash
pip install transformers datasets torch nltk rouge_score accelerate
```

### Run Inference
```python
from transformers import pipeline

# Load the trained model
summarizer = pipeline("summarization", model="pegasus-samsum-model")

# Generate summary
dialogue = """Person A: Hey, are we still meeting for lunch?
Person B: Yes, at 1 PM at the usual place.
Person A: Great, see you then!"""

summary = summarizer(dialogue)[0]["summary_text"]
print(summary)  # Output: "Meeting for lunch at 1 PM at the usual place."
```

## 📊 Project Overview

- **Model**: Google PEGASUS fine-tuned on SAMSum dataset
- **Task**: Conversational text summarization
- **Input**: Dialogue/conversation text
- **Output**: Concise summary (max 128 tokens)

## 🏗️ Training

```python
# Train the model (optional - pre-trained version available)
python train_model.py
```

## 📁 Project Structure

```
├── Text_Summarizer_project.ipynb  # Main training notebook
├── pegasus-samsum-model/          # Trained model
├── tokenizer/                     # Model tokenizer
└── README.md                      # This file
```

## 🎯 Key Features

- **Abstractive Summarization**: Generates new text rather than extracting
- **Conversational Focus**: Optimized for dialogue summarization
- **GPU Acceleration**: CUDA support for faster training/inference
- **Production Ready**: Saved model can be loaded without retraining

## 📈 Performance

- **ROUGE-1**: ~47.6 F1 score
- **ROUGE-2**: ~24.0 F1 score  
- **ROUGE-L**: ~39.4 F1 score

## 🔧 Requirements

- Python 3.7+
- PyTorch
- Transformers library
- 8GB+ GPU memory (recommended)

## 🚀 Usage Examples

### Basic Usage
```python
# Single conversation
summary = summarizer(long_conversation_text)[0]["summary_text"]

# Batch processing
dialogues = [conv1, conv2, conv3]
summaries = summarizer(dialogues)
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📄 License

This project is open source and available under the MIT License.
