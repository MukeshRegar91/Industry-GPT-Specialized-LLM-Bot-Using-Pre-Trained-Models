# IndustryGPT: Specialized LLM Bot Using Pre-Trained Models

**Category:** Deep Learning for NLP  
**Domain:** Information Technology (IT Support)  
**Base model:** [`google/flan-t5-small`](https://huggingface.co/google/flan-t5-small)  

IndustryGPT is an IT-support question-answering chatbot that combines a **fine-tuned FLAN-T5 Small** language model with **TF-IDF retrieval**. It is designed to assist with common issues involving Windows, printers, VPNs, Wi-Fi, Outlook, Teams, drivers, and related software.

The project covers dataset preparation, exploratory data analysis, model fine-tuning, evaluation, retrieval-based responses, and a Streamlit chat interface.

## Project Objectives

- Adapt a pre-trained language model to IT-support question answering.
- Retrieve existing troubleshooting answers when a close match is available.
- Generate an answer when retrieval does not find a sufficiently similar question.
- Evaluate the fine-tuned model against the unmodified base model.
- Demonstrate the chatbot through a Streamlit interface.

## Dataset

**Source:** [Benjamin Macklin — IT Support (Hugging Face)](https://huggingface.co/datasets/benjaminmacklin/IT_Support)

The notebook downloads six JSONL files and primarily uses troubleshooting and step-based instruction/answer records, with a sample of conversational examples. Its preprocessing repairs some malformed JSON lines, removes missing values and duplicate instructions, and produces **17,158 cleaned Q&A pairs**.

| Dataset split | Examples |
|---|---:|
| Training | 1,200 |
| Validation | 218 |
| Test | 218 |

The training examples are sampled across IT topics to reduce dominance by the most frequent categories.

## Methodology

1. **Data collection and EDA:** Download JSONL data, clean records, inspect question/answer lengths, and explore IT-support topics.
2. **Preprocessing:** Build topic-aware samples and training, validation, and test splits.
3. **Fine-tuning:** Train `google/flan-t5-small` (approximately 77M parameters) for **3 epochs**, batch size **8**, and learning rate **3e-4**.
4. **Retrieval:** Build a TF-IDF index over the **17,158** cleaned questions using unigrams and bigrams.
5. **Evaluation:** Compare base and fine-tuned models using ROUGE-1 and ROUGE-L on **60** sampled held-out test questions.
6. **Chatbot:** Handle greetings and basic keyword-based refusals; retrieve a stored answer when cosine similarity is **at least 0.45**; otherwise generate a response with the fine-tuned model.
7. **Interface:** Provide an interactive chat UI using Streamlit.

## Model Results

| Metric | Base FLAN-T5 Small | Fine-tuned FLAN-T5 Small |
|---|---:|---:|
| ROUGE-1 F1 | 0.1306 | **0.3641** |
| ROUGE-L F1 | 0.1119 | **0.2687** |

The fine-tuned model improved lexical overlap with reference answers on the sampled evaluation set. **These scores evaluate model-generated answers, not the complete retrieval-plus-generation application, and do not establish that all troubleshooting responses are correct.**

## Tech Stack

Python · PyTorch · Hugging Face Transformers · Hugging Face Hub · scikit-learn · Pandas · NumPy · Matplotlib · ROUGE Score · Streamlit

## Project Structure

```text
IndustryGPT_Project/
├── IndustryGPT Master.ipynb          # End-to-end analysis, training and evaluation
├── app/
│   └── streamlit_app.py              # Streamlit chat interface
├── data/                             # Downloaded and processed data
├── models/
│   ├── flan-t5-small-itsupport/      # Fine-tuned model and tokenizer
│   ├── retrieval_vectorizer.pkl
│   ├── retrieval_matrix.pkl
│   └── retrieval_corpus.csv
├── outputs/                          # EDA plots and evaluation artifacts
└── requirements.txt                  # Project dependencies
```

> Folder contents shown here reflect the notebook's expected output paths and the separate Streamlit app described in the project. Check that these artifacts exist in your repository before publishing or running the app.

## Run the Project

### 1. Open the notebook

Open `IndustryGPT Master.ipynb` in [Google Colab](https://colab.research.google.com/) or Jupyter and run its cells in order. The notebook downloads and processes the dataset, fine-tunes the model, builds the retrieval index, evaluates the model, and demonstrates chatbot responses.

**Note:** The fine-tuning step can take a while on CPU. If you already have saved trained model files, you can reuse them rather than retraining.

### 2. Run the Streamlit application

From the project root, with the required Python dependencies installed and model/retrieval artifacts present:

```bash
pip install -r requirements.txt
python -m streamlit run app/streamlit_app.py
```

Streamlit typically opens at `http://localhost:8501` when running on your own machine.

**Colab demo:** If running inside Google Colab, Streamlit runs on Colab's remote runtime, not directly on your computer. A temporary tunnel can make it accessible in a browser; the tunnel and runtime must remain active during the demonstration.

## Example Questions

- "My printer is showing offline. How do I fix it?"
- "My VPN keeps disconnecting on Windows 11."
- "My laptop cannot connect to Wi-Fi. What should I check?"
- "How do I troubleshoot Bluetooth audio issues?"

## Limitations and Future Work

- The model was fine-tuned on a relatively small training subset due to CPU constraints.
- TF-IDF uses lexical similarity and can retrieve an answer for the wrong technical context.
- Generated answers can be incomplete, repetitive, or inaccurate; important fixes should be verified.
- The keyword-based safety check is basic and is not a comprehensive security safeguard.
- Future improvements include semantic search with sentence embeddings and FAISS, larger-scale training, stronger safety checks, and multi-turn conversation memory.

## Acknowledgments

- [Google FLAN-T5 Small](https://huggingface.co/google/flan-t5-small)
- [Benjamin Macklin IT Support dataset](https://huggingface.co/datasets/benjaminmacklin/IT_Support)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/index)
- [Streamlit](https://streamlit.io/)

---

*IndustryGPT is a learning/capstone project and is not a substitute for professional IT support.*
