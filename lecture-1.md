# How Transformer Models Work and Solve Tasks

This notes provides an overview of the Transformer architecture, its history, core concepts, and its application in solving various tasks in Natural Language Processing (NLP), speech, and computer vision, based on the provided sources.

## 1. Introduction and History of Transformers

The Transformer architecture was introduced in June 2017, initially focusing on translation tasks. It is primarily composed of two blocks: an **encoder**, which builds a feature representation of an input, and a **decoder**, which uses that representation to generate a target sequence. A key feature of these models is their use of **attention layers**, which allow the model to focus on specific parts of the input sequence when processing information.

All major Transformer models (like GPT, BERT, T5) are trained as **language models** on large amounts of raw text using a self-supervised learning approach, which does not require human-annotated labels.

### 1.1. A Brief Timeline of Key Models

The history of Transformer models includes several influential releases:
*   **June 2018**: **GPT**, the first pretrained Transformer model used for fine-tuning on various NLP tasks.
*   **October 2018**: **BERT**, a large pretrained model designed for better sentence summarization.
*   **February 2019**: **GPT-2**, a larger version of GPT.
*   **October 2019**: **T5**, a multi-task, sequence-to-sequence Transformer model.
*   **May 2020**: **GPT-3**, an even larger version of GPT-2 capable of zero-shot learning.
*   **January 2022**: **InstructGPT**, a version of GPT-3 trained to better follow instructions.
*   **January 2023**: **Llama**, a large language model capable of generating text in multiple languages.
*   **March 2023**: **Mistral**, a 7-billion-parameter model that outperforms Llama 2 13B, using grouped-query and sliding window attention for efficiency.
*   **May 2024**: **Gemma 2**, a family of lightweight open models (2B to 27B parameters) using interleaved local-global and group-query attention.
*   **November 2024**: **SmolLM2**, a state-of-the-art small language model (135M to 1.7B parameters) designed for mobile and edge devices.

## 2. Core Architectural Concepts

### 2.1. Encoder-Decoder Structure

The Transformer model consists of an encoder and a decoder, each of which can be used independently depending on the task:
*   **Encoder-only models (e.g., BERT)**: These models are optimized to understand the input. They are well-suited for tasks like sentence classification and named entity recognition.
*   **Decoder-only models (e.g., GPT, Llama)**: These models are optimized for generating outputs. They are good for generative tasks like text generation.
*   **Encoder-decoder models (e.g., T5, BART)**: Also known as sequence-to-sequence models, they are used for generative tasks that require an input, such as translation or summarization.

### 2.2. Attention Layers

A foundational component of the Transformer is the **attention layer**. The title of the original paper was "Attention Is All You Need". This layer instructs the model to pay specific attention to certain words in a sentence when processing the representation of each word. A word's meaning is deeply affected by its context, which can be other words before or after it. For example, in translation, the conjugation of a verb may depend on the subject, and the form of a demonstrative pronoun ("this") may depend on the gender of the noun it modifies. The attention mechanism allows the model to weigh the importance of different words in the input sequence to make these determinations.

### 2.3. Terminology: Architecture vs. Checkpoint
It is important to distinguish between a few key terms:
*   **Architecture**: The skeleton of the model—the definition of its layers and operations.
*   **Checkpoints**: The specific weights that are loaded into a given architecture. For example, BERT is an architecture, while `bert-base-cased` is a checkpoint.
*   **Model**: An umbrella term that can refer to either the architecture or the checkpoint.

## 3. Training and Transfer Learning

### 3.1. Language Model Training

Transformers are trained as language models, giving them a statistical understanding of language. There are two main self-supervised approaches:
1.  **Causal Language Modeling (CLM)**: The model predicts the next word in a sentence based on the *n* previous words. This is used by **decoder models** like GPT, which can only use context from the left (previous tokens).
2.  **Masked Language Modeling (MLM)**: The model predicts a masked word within a sentence. This is used by **encoder models** like BERT, allowing them to learn from bidirectional context (words before and after the mask).

### 3.2. Transfer Learning: Pretraining and Fine-Tuning

The common practice for using Transformers is **transfer learning**.
*   **Pretraining**: This is the initial training of a model from scratch on a very large corpus of data. This process is extremely costly in terms of time, data, and compute resources, and it has a significant environmental impact.
*   **Fine-tuning**: After a model is pretrained, it undergoes fine-tuning, which is additional training on a smaller, task-specific dataset with human-annotated labels.

**Benefits of fine-tuning** include:
*   Leveraging knowledge already acquired during pretraining.
*   Requiring significantly less data, time, and resources.
*   Achieving better results than training from scratch, unless you have vast amounts of data.

This is why **sharing pretrained models** is crucial; it reduces the overall compute cost and carbon footprint for the community.

## 4. How Transformers Solve Tasks

Most tasks follow a similar pattern: input data is prepared and processed through a model, and the output is interpreted for the specific task. The main difference lies in the model architecture variant used and how the output is processed.

### 4.1. NLP Tasks

*   **Text Generation (Decoder-only)**: Models like **GPT-2** are pretrained using causal language modeling, making them excellent at generating text. They use *masked self-attention*, which prevents the model from attending to future tokens, ensuring it only uses the preceding context to generate the next word.
*   **Text Classification (Encoder-only)**: For a model like **BERT**, a special `[CLS]` token is added to the beginning of the input sequence. A sequence classification head (a linear layer) is added on top of the base model, which takes the final hidden state of the `[CLS]` token as input to produce classification logits.
*   **Token Classification (Encoder-only)**: To use **BERT** for tasks like Named Entity Recognition (NER), a token classification head is added. This linear layer takes the final hidden states of *all* tokens and converts them into logits for each token's label.
*   **Question Answering (Encoder-only)**: With **BERT**, a span classification head is added to compute the start and end logits for the answer's span within the context text.
*   **Summarization & Translation (Encoder-Decoder)**: Models like **BART** and **T5** are ideal for these sequence-to-sequence tasks. The encoder processes the source text, and the decoder generates the condensed summary or the translated text.

### 4.2. Modalities Beyond Text

Transformers are not limited to text and can be applied to speech, audio, and computer vision.

*   **Speech and Audio**: **Whisper** is an encoder-decoder model pretrained on 680,000 hours of labeled audio data. The encoder processes a log-Mel spectrogram of the audio, and the decoder autoregressively predicts the corresponding text tokens for tasks like automatic speech recognition.
*   **Computer Vision**: For image classification, the **Vision Transformer (ViT)** splits an image into a sequence of non-overlapping patches, which are treated like tokens. A special `[CLS]` token and positional embeddings are added before the sequence is passed to a Transformer encoder. The final hidden state of the `[CLS]` token is used by a classification head to predict the image's class.

# The Attention Mechanism in Detail

This notes provides a detailed explanation of the attention mechanism, its core components (queries, keys, and values), the step-by-step calculation process, and key architectural variants such as multi-head and causal attention. The information is drawn from the provided sources.

## 1. Introduction: The Concept of Attention

Traditional deep learning models like CNNs struggle with inputs of variable size. The **attention mechanism** was introduced to address this, particularly for long sequences where it's difficult for a network to keep track of all information.

The core idea of attention can be understood through a database analogy. A simple database consists of a collection of **(key, value)** pairs. When you provide a **query**, the system compares it to all the keys to find matches and returns the corresponding values.

The attention mechanism formalizes this concept for deep learning. It provides a **differentiable** method for a neural network to select elements from a set and construct a weighted sum of their representations. This allows the model to selectively focus on the most important parts of an input sequence for a given task. The final output is a linear combination of values, where the weights are derived from the compatibility between a query and the keys.

## 2. Core Components: Queries, Keys, and Values

The attention mechanism operates on three main components, which are all vectors:
*   **Query (`q`)**: Represents the current position or item of interest. It is used to "ask" for relevant information by probing the keys.
*   **Key (`k`)**: Paired with each value, the key is compared against the query to determine the relevance or "attention weight" of its corresponding value.
*   **Value (`v`)**: Contains the actual feature representation of an input element. The final output of the attention mechanism is a weighted sum of all value vectors.

In the context of **self-attention**, a mechanism that relates different positions of a single sequence, the query, key, and value vectors are all derived from the **same input sequence**. Each input token's embedding, **x**, is projected into these three roles by multiplying it with three distinct, learnable weight matrices: $W_q$, $W_k$, and $W_v$.

*   Query sequence: $q(i) = x(i)W_q$
*   Key sequence: $k(i) = x(i)W_k$
*   Value sequence: $v(i) = x(i)W_v$

## 3. The Attention Calculation: A Step-by-Step Process

The process of using a query to retrieve a context vector can be broken down into four steps.

### Step 1: Compute Attention Scores

The first step is to calculate a compatibility score between a query $q$ and every key $k_i$ in the sequence. This is done using an **attention scoring function**, denoted as $a(q, k_i)$. The output is a scalar value called the unnormalized attention weight or attention score, often symbolized as $ω$.

### Step 2: Normalize Scores into Attention Weights

The raw scores are then converted into a set of non-negative weights that sum to 1. This is typically achieved by applying the **softmax function**. The resulting normalized weights, $α$, represent how much attention the model should pay to each input element.

The formula is:
$α(q, k_i) = softmax(a(q, k_i)) = exp(a(q, k_i)) / Σ_j exp(a(q, k_j))$

### Step 3: Compute the Context Vector

Finally, the context vector $z$ is calculated as a **weighted sum of all value vectors $v_i$**, using the attention weights $α$ as the coefficients. This operation is also known as *attention pooling*.

The formula for the final output is:
$Attention(q, D) = Σ_i α(q, k_i) * v_i$

This process enhances the original input embedding with information about its context, weighted by relevance.

## 4. Key Attention Scoring Functions

The choice of the scoring function $a(q, k)$ is a critical design detail. The two most common functions are:

### 4.1. Scaled Dot-Product Attention

This is the most popular and widely used attention mechanism, forming the basis of the Transformer architecture. The score is the dot product of the query and key vectors.
*   **Scaling**: The dot product is scaled by $1/√d$, where $d$ is the dimension of the key vectors. This scaling is crucial to ensure the variance of the dot product remains controlled, which prevents the softmax function from producing overly small gradients and helps stabilize training.

The full scoring function is: $a(q, k_i) = q^T * k_i / √d$.

In matrix form for a batch of queries $Q$, keys $K$, and values $V$, the entire scaled dot-product attention operation is calculated efficiently as:
**$Attention(Q, K, V) = softmax( (Q * K^T) / √d ) * V$**

### 4.2. Additive Attention

When query and key vectors have different dimensions, additive attention can be used. This function uses a small feed-forward neural network with a single hidden layer to compute the compatibility score.

The scoring function is: $a(q, k) = w_v^T * tanh(W_q * q + W_k * k)$.
Here, $W_q$, $W_k$, and $w_v$ are learnable weight parameters.

## 5. Practical Implementation Details

### 5.1. Masking for Variable-Length Sequences

In practice, sequences in a single minibatch often have different lengths and are padded to be the same size. The attention mechanism should ignore these padding tokens. This is achieved with the **masked softmax operation**. Before applying the softmax function, the attention scores corresponding to the padded positions are set to a very large negative number (e.g., -1e6). When the softmax is applied, the exponentiation of this large negative number results in a value of 0, effectively nullifying the contribution of padded tokens.

### 5.2. Batch Matrix Multiplication (BMM)

To efficiently process minibatches of queries, keys, and values, deep learning frameworks use **batch matrix multiplication (BMM)**. This operation performs matrix multiplication for each item in the batch independently and in parallel.

## 6. Important Variants of Attention

### 6.1. Self-Attention

Self-attention is the mechanism used when the queries, keys, and values are all derived from the **same input sequence**. This allows the model to weigh the importance of different words within the same sentence and capture intra-sequence dependencies. The entire process can be encapsulated in a class for use as a model layer.

### 6.2. Multi-Head Attention

Instead of performing a single attention calculation, **multi-head attention** runs the scaled dot-product attention mechanism multiple times in parallel. Each parallel instance is called an "attention head," and each head has its own set of learnable weight matrices ($W_q$, $W_k$, $W_v$). The resulting context vectors from each head are concatenated to produce the final output.

The benefits of this approach include:
*   **Diverse Representations**: Each head can learn to focus on different parts of the input sequence, capturing various types of relationships and features.
*   **Parallel Computation**: The independent nature of each head is well-suited for parallel processing on modern hardware like GPUs.

### 6.3. Causal (Masked) Self-Attention

Causal attention is crucial for auto-regressive or decoder-style language models (like GPT) that generate text one token at a time. It ensures that the prediction for a given token can only depend on the preceding tokens and not on any "future" tokens.

This is implemented by applying a mask to the attention scores before the softmax step.
*   **The Mask**: A mask is created to identify all positions where a query $i$ is attending to a key $j$ where $j > i$ (i.e., keys that are in the future).
*   **Efficient Implementation**: The most efficient way to apply this mask is to replace the attention scores for all future positions with **negative infinity** (`-torch.inf`) before the softmax function is applied. The softmax function then converts these `-inf` scores into attention weights of 0, effectively preventing any information flow from future tokens.

# A Tutorial on Word Embeddings: Efficient and Meaningful Word Representations

This tutorial explains what word embeddings are, why they are a crucial component in modern Natural Language Processing (NLP), and how they provide a more efficient and meaningful way to represent words for machine learning models.

## 1. What is Word Embedding and Why Does It Matter?

In natural language, words are the basic unit of meaning. For a computer to process language, we need to convert these words into a numerical format. **Word embedding** is the technique of mapping words to vectors of real numbers. These vectors, also called *word vectors*, serve as feature representations for words that neural networks can process.

The quality of this representation is critical. A good representation should not only allow a model to process the word but also capture the word's meaning and its relationships with other words. This is where older, simpler methods fall short and why modern word embeddings matter so much.

## 2. The Problem: Why One-Hot Vectors Are Inefficient

A straightforward way to represent words numerically is using **one-hot vectors**. Imagine you have a vocabulary of *N* unique words. To represent a word, you create a vector of length *N* that is all zeros except for a single "1" at the index corresponding to that word.

While easy to construct, this method suffers from two major drawbacks:

1.  **It Fails to Capture Meaning**: One-hot vectors cannot express the similarity between different words. The **cosine similarity**—a common metric to measure how similar two vectors are—between the one-hot vectors of any two different words is always 0. This means the representation for "cat" is just as different from "dog" as it is from "car," which is not useful for understanding language.
2.  **It is Highly Inefficient**: For a large vocabulary (e.g., 50,000 words), each word is represented by a 50,000-dimensional vector. These vectors are extremely **sparse** (mostly filled with zeros), making them very inefficient in terms of both memory storage and computation.

## 3. The Solution: Dense and Efficient Embeddings with `word2vec`

To solve these issues, the **`word2vec`** tool was developed. It maps each word to a **fixed-length, dense vector**. This approach serves as a form of **dimensional reduction**, transforming the high-dimensional, sparse one-hot representations into lower-dimensional, dense vectors that are far more efficient.

The most important benefit is that these vectors are **semantically meaningful**. Words with similar meanings will have similar vector representations, allowing the model to capture similarity and analogy relationships.

## 4. How Word Embeddings Are Learned

`word2vec` models learn these meaningful vectors through a **self-supervised** process. This means they learn directly from large amounts of raw text (corpora) without needing human-annotated labels. The core idea is that a word's meaning can be inferred from the words that typically appear around it. The training process involves predicting words from their context.

There are two primary `word2vec` models:

### 4.1. The Skip-Gram Model

The skip-gram model assumes that a word can be used to generate its surrounding words.
*   **Task**: Given a *center word*, the model's goal is to predict the *context words* that appear within a certain window.
*   **Example**: In the phrase "the man loves his son," if "loves" is the center word and the window size is 2, the model tries to predict "the," "man," "his," and "son" based on "loves".
*   **Training**: During training, the model's parameters (the word vectors) are adjusted to maximize the probability of correctly generating the context words for each center word in the text. After training, the model's **center word vectors** are typically used as the final word representations.

### 4.2. The Continuous Bag of Words (CBOW) Model

The CBOW model does the opposite of skip-gram; it assumes a center word is generated from its surrounding context.
*   **Task**: Given a set of *context words*, the model's goal is to predict the *center word*.
*   **Example**: In "the man loves his son," the model would use the context words "the," "man," "his," and "son" to predict the center word "loves". The vectors of the context words are averaged to form the input.
*   **Training**: The model is trained by adjusting the word vectors to maximize the probability of correctly predicting the center word from its context. Unlike skip-gram, CBOW typically uses the **context word vectors** as the final word representations.

## 5. The Efficiency of Representation

Word embeddings provide a far more efficient representation of words compared to one-hot vectors in two key ways:

*   **Memory and Storage Efficiency**: By transforming words into dense vectors of a much lower dimension (e.g., 300 dimensions instead of 50,000), word embeddings are significantly more memory-efficient.
*   **Computational Efficiency**: While the training process itself can be computationally intensive—especially the softmax calculation, which requires summing over the entire vocabulary for every prediction—the resulting dense vectors are computationally cheaper to use in downstream models. The challenge of large vocabularies during training has led to optimization techniques (like negative sampling, which is not detailed in the sources) that make `word2vec` practical even for huge dictionaries.

In summary, word embeddings are a foundational technique in NLP because they create **dense, efficient, and semantically rich representations** of words, overcoming the critical limitations of sparse methods like one-hot encoding.

# References


*   Ahead of AI. (2024, January 14). *Understanding and Coding Self-Attention, Multi-Head Attention, Causal-Attention, and Cross-Attention in LLMs*. Ahead of AI. https://magazine.sebastianraschka.com/p/understanding-and-coding-self-attention

*   Hugging Face. (n.d.). *How do Transformers work?* Hugging Face LLM Course. https://huggingface.co/learn/llm-course/chapter1/4

*   Hugging Face. (n.d.). *How 🤗 Transformers solve tasks*. Hugging Face LLM Course. https://huggingface.co/learn/llm-course/chapter1/5


*   Zhang, A., Lipton, Z. C., Li, M., & Smola, A. J. (n.d.). *11.1. Queries, Keys, and Values*. In Dive into Deep Learning. https://www.d2l.ai/chapter_attention-mechanisms-and-transformers/queries-keys-values.html

*   Zhang, A., Lipton, Z. C., Li, M., & Smola, A. J. (n.d.). *11.3. Attention Scoring Functions*. In Dive into Deep Learning. https://www.d2l.ai/chapter_attention-mechanisms-and-transformers/attention-scoring-functions.html

*   Zhang, A., Lipton, Z. C., Li, M., & Smola, A. J. (n.d.). *15.1. Word Embedding (word2vec)*. In Dive into Deep Learning. https://www.d2l.ai/chapter_natural-language-processing-pretraining/word2vec.html
