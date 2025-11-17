# Build LLM from Scratch

Implementation of a ChatGPT-like Large Language Model from scratch, following the book **"Build a Large Language Model (From Scratch)"** by Sebastian Raschka.

## Description

This repository contains a step-by-step implementation of a Large Language Model (LLM) similar to ChatGPT. The goal is to deeply understand how these models work by building them from scratch, without relying on high-level frameworks.

## Project Structure

```
Build_LLM/
├── stage_1/
│   ├── chapter_2.ipynb           # Tokenization and data preparation
│   ├── the-verdict.txt            # Sample text for training
│   └── tokenization_tree.png      # Tokenization process visualization
└── README.md
```

## Implemented Content

### Stage 1 - Fundamentals

#### Chapter 2: Tokenization and Data Preparation

This chapter covers the fundamental concepts of text preprocessing for LLMs:

**1. Text Preprocessing**
- Token separation while preserving punctuation marks
- Maintaining capitalization for better context understanding
- Whitespace handling

**2. Basic Tokenization (SimpleTokenizerV1)**
- Vocabulary creation from text
- Word-to-ID mapping
- Encoding and decoding methods

**3. Special Token Handling (SimpleTokenizerV2)**
- `<|endoftext|>`: Sequence separator
- `<|unk|>`: Unknown tokens (out-of-vocabulary words)
- Comparison with other special tokens: `[BOS]`, `[EOS]`, `[PAD]`

**4. Byte Pair Encoding (BPE)**
- Implementation using `tiktoken` (GPT-2 tokenizer)
- Efficient handling of unknown words through subwords
- Advantages over simple tokenization

**5. Input-Target Pairs Creation**
- Generating training data for next-word prediction
- Using sliding windows
- DataLoader implementation with PyTorch

**6. Embeddings**
- **Token Embeddings**: Vector representation of tokens
- **Positional Embeddings**: Absolute position encoding
- Combining both embeddings for final model input

## Technologies Used

- **Python**: Main language
- **PyTorch**: Deep learning framework
- **tiktoken**: OpenAI's BPE tokenizer
- **Matplotlib**: Process visualization
- **Jupyter Notebook**: Interactive development environment

## Requirements

```bash
pip install torch tiktoken matplotlib jupyter
```

## Key Concepts Implemented

### Tokenization
The process of converting text into tokens (processable units):
- Text → Tokens → Numeric IDs → Embeddings

### Byte Pair Encoding (BPE)
Algorithm that breaks down unknown words into subunits:
```
"someunkonwnPlace" → ["some", "unk", "on", "wn", "Place"]
```

### Sliding Window for Training
Creating multiple training examples from a single text:
```
Text: "and established himself in a"
Context: [290] → Target: 4920 ("established")
Context: [290, 4920] → Target: 2241 ("himself")
Context: [290, 4920, 2241] → Target: 287 ("in")
```

### Embeddings
- **Vocabulary**: 50,257 tokens (GPT-2)
- **Embedding dimension**: 256
- **Context length**: 4 tokens (configurable)

## Usage Examples

### Simple Tokenization
```python
tokenizer = SimpleTokenizerV2(vocab)
text = "Hello, do you like tea?"
ids = tokenizer.encode(text)
decoded = tokenizer.decode(ids)
```

### BPE Tokenization (GPT-2)
```python
import tiktoken
tokenizer = tiktoken.get_encoding("gpt2")
text = "Hello, do you like tea?"
tokens = tokenizer.encode(text)
# [15496, 11, 466, 345, 588, 8887, 30]
```

### DataLoader Creation
```python
dataloader = create_data_loader_v1(
    raw_text,
    batch_size=8,
    max_length=4,
    stride=4,
    shuffle=False
)
```

## Visualizations

The project includes tokenization process visualizations showing:
- Original text
- Individual tokens with their IDs
- Text reconstruction from tokens

![Tokenization Process](stage_1/tokenization_tree.png)

## Next Steps

- Model architecture implementation (Transformer)
- Self-Attention mechanism
- Feed-Forward Networks
- Model training
- Fine-tuning and evaluation

## Resources

- **Reference book**: "Build a Large Language Model (From Scratch)" - Sebastian Raschka
- **Official book repository**: [LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch)

## Technical Notes

### GPT vs Other Models Differences
- GPT uses only `<|endoftext|>` (also for padding)
- Doesn't use `<|unk|>` thanks to BPE
- Doesn't require separate `[BOS]`, `[EOS]`, `[PAD]` tokens

### Important Parameters
- **vocab_size**: 50,257 (GPT-2 standard)
- **output_dim**: 256 (embedding dimension)
- **context_length**: Maximum sequence length
- **stride**: Sliding window step for creating batches

## Learning Outcomes

By working through this implementation, you will understand:

- How text is converted into numerical representations
- The importance of positional information in transformers
- How BPE tokenization handles out-of-vocabulary words
- Data preparation pipelines for LLM training
- The relationship between tokens, embeddings, and model inputs

## Author

This project is developed following Sebastian Raschka's book as a learning guide.

## License

This is an educational project based on the book "Build a Large Language Model (From Scratch)".
