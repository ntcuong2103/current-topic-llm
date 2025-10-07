## Tutorial: From Chain-of-Thought to Tree-of-Thought

**Understanding How Large Language Models “Think” During Inference**

### 1. The Basic Engine: Autoregressive Inference

A large language model (LLM) doesn’t “reason” in a human sense.
It predicts the next token, over and over, based on all previous tokens.
Each prediction is a **forward pass** through the network.

When we encourage reasoning (e.g., *“Let’s reason step by step”*), the model simply **writes its intermediate steps as text** before concluding.
These intermediate tokens become part of the prompt for later tokens, guiding the continuation.

---

### 2. Chain-of-Thought (CoT)

**Definition:**
CoT is the practice of letting an LLM generate intermediate reasoning steps before the final answer.

**How it works:**

1. You give a question and a reasoning cue (“think step by step”).
2. The model generates a *chain* of text tokens: Step 1 → Step 2 → … → Conclusion.
3. Each new token conditions on the prior text—this running text *is* the model’s state.

**No special mode:**
There is no internal “stop-and-think” pause. The model’s latent activations carry information continuously.
The visible text acts as the **externalized state**.

**Benefits:**

* Encourages structured reasoning instead of shallow pattern matching.
* Reduces “shortcut” answers.
* Easy to implement—just a prompt.

**Limits:**

* The chain can still go wrong early and never recover.
* Long chains cost more tokens and can amplify hallucination if unchecked.

---

### 3. Adding Branching: From Single Chain to Multiple Paths

To increase reliability, researchers introduced **branching methods** that explore *several possible reasoning paths* instead of just one.

All of these wrap multiple CoT runs inside a **controller loop**.

---

### 4. Self-Consistency

**Idea:**
Run CoT several times stochastically and take a majority vote on the final answer.

**Process:**

1. Same prompt → random sampling (temperature > 0).
2. Collect N independent full rationales.
3. Choose the most frequent or best-scoring conclusion.

**Advantages:**
Simple, parallelizable, improves accuracy on math and logic tasks.

**Drawback:**
Wastes compute on entirely separate runs; no sharing of progress.

---

### 5. Mini-Tree-of-Thought (mini-ToT)

**Idea:**
Branch only at key steps.

**Process:**

1. Generate reasoning up to a pivot step.
2. Sample k candidate continuations.
3. Score them (model or rule-based).
4. Keep the best branch and continue.

**Advantages:**
Captures “what-if” flexibility without full search explosion.
**Use-case:** Error-prone mid-chain decisions (e.g., formula choice, hypothesis test).

---

### 6. Full Tree-of-Thought (ToT)

**Idea:**
Represent reasoning as a **search tree** of partial thoughts.

Each node = partial reasoning trace.
Each edge = a possible next thought.

**Controller functions:**

* **Expand:** sample candidate next steps.
* **Evaluate:** score promise of each branch.
* **Select:** keep best nodes (beam search, BFS, DFS).
* **Repeat:** until solution or depth limit.

**Advantages:**
Can backtrack, compare branches, plan ahead.
**Cost:** many forward passes, exponential if unpruned.

---

### 7. Monte-Carlo Tree Search (MCTS)

**Idea:**
Borrow from game AI to explore reasoning trees *strategically*.

**Four phases per cycle:**

1. **Selection** – pick a node balancing exploration vs. exploitation.
2. **Expansion** – sample continuations.
3. **Simulation** – quickly roll out each branch to a provisional answer.
4. **Backpropagation** – update node values.

**Roles in LLM context:**

* *Policy model* → generates next steps.
* *Value model* → estimates correctness or utility.
* *Controller* → orchestrates search.

**Advantages:**
Efficiently focuses compute on promising branches; excels in planning and creative reasoning.

---

### 8. Comparing the Methods

| Method               | Branch Interaction | Controller Complexity | Compute Cost               | Typical Use                    |
| -------------------- | ------------------ | --------------------- | -------------------------- | ------------------------------ |
| **Chain-of-Thought** | None               | None                  | 1×                         | Simple reasoning, explanations |
| **Self-Consistency** | None               | Low                   | N×                         | Math, logic tasks              |
| **Mini-ToT**         | Local              | Moderate              | 2–3×                       | Correcting mid-chain errors    |
| **Full ToT**         | Global             | High                  | Exponential (beam-bounded) | Planning, proofs               |
| **MCTS**             | Guided             | High                  | Variable                   | Strategy, creative search      |

---

### 9. “Stop-and-Think” Behavior

LLMs themselves never truly *pause* mid-inference.
However, outer controllers can simulate reflection:

* **Stepwise prompting:** ask for one step, verify, continue.
* **Verifier loops:** critique or score reasoning before resuming.
* **Latent CoT:** hidden reasoning steps used internally but not shown.

These make reasoning explicit and controllable, enabling inspection and debugging.

---

### 10. Takeaway

* **CoT** is sequential self-explanation: cheap, simple, powerful.
* **Self-Consistency** adds statistical stability.
* **Mini-ToT** and **ToT** introduce structural exploration.
* **MCTS** adds informed search efficiency.

All reuse the same underlying LLM forward passes—only the *orchestration* changes.
What we’re really learning is not how models think, but how to **manage their thinking space**.

---

**Further exploration:**
Look up *ReAct* (reason + act with tools), *Reflexion* (self-critique loops), and *verifier-guided decoding*—all descendants of the same idea: turning token prediction into a controlled process of deliberation.
