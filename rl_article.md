# Fine-tuning LLMs using Reinforcement Learning

JANU VERMA

OCT 12, 2025

## Introduction

Reinforcement learning (RL) has emerged to take a central role in the modern developments of LLMs-powered AI. Not only, RL is being used for LLM alignment (RLHF), it is being used as a recipe to enhance both training and inference. Most notably the emergence of large reasoning models — DeepSeek-R1 and OpenAI o-series — has sparked interest in using RL for fine-tuning pre-trained LLMs to cultivate ‘thinking’ capabilities in models so they can tackle complex tasks e.g., code generation and mathematical reasoning.

One of the simplest RL algorithm is REINFORCE - which is a type of policy gradient method. I decided to implement REINFORCE from scratch in PyTorch and explore its efficacy for fine-tuning LLMs. This exercise provides a deeper understanding of the algorithm, its deficiencies, and improvements, particularly for training LLMs. Another motivating factor is that many of the modern RL algorithms (e.g. PPO, GRPO, REINFORCE++) are built on top of REINFORCE, so one can easily extend the system to other methods. In this post, we will implement a simple recipe for RL-training of a LLM using REINFORCE algorithm. We will cover basic theory which is needed to better understand the setup.

## LLM Fine-tuning

At its core, fine-tuning is the process of specializing a general-purpose tool. A pre-trained language model is excellent at predicting text but indifferent to your goals (helpfulness, correctness under tools, safety, reasoning discipline). Fine-tuning means changing the model’s parameters so its behavior better matches a new objective while preserving the useful prior learned during pre-training. For example, adapting a LLM to acts as a Python code generator by training it on a vast library of code and its descriptions. The most common method for this is Supervised Fine-Tuning (SFT), where you provide the model with a dataset of high-quality (prompt, ideal_response) pairs. The model learns to imitate these examples. This is highly effective for teaching the model a specific format and style.

While SFT is powerful, it has fundamental limitations that become clear when we want to teach the model more abstract concepts than just imitation. This is where the motivation for Reinforcement Learning (RL) comes from.

The core problem is that “a good response” is often complex, subjective, and hard to define with a single static example.

- There isn’t always one “correct” answer. For a prompt like “Explain black holes to a 5-year-old,” there are countless creative, correct, and helpful responses. SFT forces you to choose just one “ideal” response for your dataset, which limits the model’s creativity and flexibility.
- It’s easier to judge than to create. Just like a film critic can easily tell you if a movie is good or bad without being able to direct a masterpiece themselves, humans are much better at comparing two responses and judging which is better than writing a perfect response from scratch. SFT requires the latter, which is expensive and difficult to scale.
- You can’t easily capture abstract goals in examples. How do you create a dataset of examples to teach a model to be “harmless,” “ethically aligned,” “cautious when discussing medical topics,” or “more engaging”? These are behavioral traits, not knowledge points. You want to guide the model’s behavior based on principles, not just mimic a fixed set of answers.

This is precisely the kind of problem RL is designed to solve. RL is a framework for teaching an agent to achieve a goal through trial and error, guided by a reward signal. Instead of telling the model “here is the perfect answer,” we tell it “that was a good answer, do more like that” or “that was a bad answer, avoid doing that.”

This allows us to move from imitation to alignment — shaping the model’s behaviour to align with complex, abstract human values.

Before we delve into RL-based fine-tuning of LLMs, let’s quickly review basics of RL.

## RL Basics

Markov Decision Process (MDP) provides a formal mathematical framework used to describe the environment in an RL problem. For a situation to be modeled as an MDP, it must satisfy the Markov Property which states that “the future is independent of the past given the present.” In simpler terms, the current state s_t contains all the information needed to decide the future, we don’t need to know the entire history of all the states and actions that led us to s_t.

For example, to decide the next best move in chess, all you need is the current configuration of the pieces on the board. But for Poker, the history of how opponents have bet in past is crucial for guessing what’s in their hand.

A MDP is defined by (S, A, P, R, \gamma)

- State Space S is the set of all possible states the agent can be in. e.g. the agent’s location on a grid.
- Action Space A is the set of all possible actions the agent can take. e.g. {up, down, left, right}
- Transition Probability P defines the dynamics of the environment that is P(s_t+1 | a_t, s_t) is the probability of transitioning to the state s_t+1 after taking action a_t in state s_t.
- Reward r_t defines the immediate reward the agent receives after taking action a_t in state s_t and landing in the state s_t+1.
- Discount Factor gamma is a number between 0 and 1 that determines the importance of future rewards. A reward received k steps in the future is worth only $\gamma^{k-1}$ times what it would be worth if received immediately. We need it for mathematical convenience as it prevents the total reward from becoming infinite and also it models the preference for immediate rewards.

In MDP, the goal of an agent is to find a policy π which is a strategy that tells the agent which action to take in any given state. A policy is the agent’s brain. Formally, the policy is a mapping from the state space to the action space.

$$\pi : S \rightarrow A$$

We can then model the probability of an action from a given state as π_θ(a_t | s_t) where θ are the parameters of the policy. In DeepRL, we parameterize the policy as a neural network aka policy network, with weights θ. The goal is to find the optimal parameters θ* that produce the best behaviour. First, we define our objective J(θ) which is the expected total reward over all possible trajectories that could be generated by our policy π_θ.

A trajectory τ is a sequence of states, actions, and rewards τ = (s_0, a_0, r_0, s_1, a_1, r_1, ...). The total reward of a trajectory is simply:

$$R(\tau) = \sum_{t=0}^{\infty} r_t$$

The objective function is the expectation value of this reward:

$$J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}[R(\tau)]$$

Can write this expectation as an integral (or sum) over all possible trajectories, weighted by their probability of occurring under our policy.

$$J(\theta) = \int_\tau P(\tau; \theta) R(\tau) d\tau$$

where $P(\tau; \theta)$ is the probability of trajectory given our policy parameters $\theta$.

We can maximize this loss using gradient ascent as:

$$\theta = \theta + \alpha \nabla_\theta J(\theta)$$

The Policy Gradient theorem simplifies this (using simple calculus) as

$$\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \left( \sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t | s_t) \right) R(\tau) \right]$$

After running an episode, multiply the total reward of that episode by the sum of the log-policy gradients for each action taken. Then, average this result over many episodes and take a step in that direction. The intuition is simple: **“Actions that lead to good outcomes should be made more likely, and actions that lead to bad outcomes should be made less likely.”**

In summary, the MDP provides the formal description of the problem (rules of the game), and the policy gradient method provides a mechanism find the optimal policy to solve it.

## RL fine-tuning of LLMs

Equipped with an understanding of the fine-tuning goal and a knowledge of the reinforcement learning, we can now try to map LLM fine-tuning setup to a legit RL system. As we are training the LLM, it becomes the policy that we are trying to optimise. The state of the environment is the LLM response (i.e. completion generated by the model) with initial state being the input prompt. The following is taken from Cameron Wolfe’s post.

| General RL | RL for LLMs |
| --- | --- |
| Policy | LLM |
| Initial State | Prompt |
| Action | Output (Tokens or Full Completion) |
| State | Prompt + Output |

If the input prompt is x_0, and the next token generated is x_1, then we can understand this as the agent chosing an action a_0 = x_1 at state s_0 = x_0 and the new state becomes s_1 = (x, x_1). The weights of the LLM are the parameters of the policy.

A missing piece here is the reward assignment. How do we assign rewards to the actions that the agent takes? This is the crux of the problem. This is done using a reward model which produces one or more numerical rewards for a given action in a state. It is imperative to come up with rewards that reflect the task and are less amenable to hacking (i.e. model finding a shortcut rather than actual learning).

My goal here is to fine-tune a pre-trained LLM for a specific task that allows us to explore the efficacy of RL methods for training LLMs.

## Problem Description

The task we consider is Code Generation with Test-Based Rewards. Which means train an LLM to generate python functions that pass unit tests. I chose this problem due to its simplicity and usefulness.

- Deterministic rewards: No ambiguity, pass/fail is clear
- Fast feedback: Can run tests in milliseconds
- Clear improvement metric: Pass@k rate is standard
- RL’s sweet spot: Learning from failures is exactly what RL does well

Data: We will use HumanEval (164 problems, simple) dataset. For quicker experiments, I further selected 50 problems as training and 30 problems as evaluation.

```python
from datasets import load_dataset
humaneval_dataset = load_dataset("openai_humaneval", split="test")
```

## Base Model Performance

Model: We will use Qwen/Qwen2.5-Coder-1.5B-Instruct as our base model.

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
MODEL_NAME = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
```

Before we try to fine-tune, let’s see how well the model performs on the test set. This will give us baseline metrics. To be able to test a model on this data, we need:

1. A routine to extract code from the completion.
2. Execution environment for running the extracted code on test examples.
3. Reward functions to produce rewards for test answers by comparing them with the actual answers.

We will use python regex to extract the code from prompt + completion. Let’s call the code extractor function extract_code_with_prompt. And we create a sandboxed Python executor with timeout using exec() with restricted globals. Denote the executor as HumanEvalExecutor. Let’s define the rewards:

```python
import torch
class HumanEvalReward:
    """Reward function for HumanEval"""
    def __init__(self, executor):
        self.executor = executor
        self.current_problem = None

    def set_problem(self, problem):
        self.current_problem = problem

    def __call__(self, prompts: List[str], completions: List[str]):
        if self.current_problem is None:
            raise ValueError("Must set problem first")
        rewards = []
        for prompt, completion in zip(prompts, completions):
            code = extract_code_with_prompt(
                prompt,
                completion,
                self.current_problem['entry_point']
            )
            if code is None:
                rewards.append(0.0)
                continue
            # Execute test
            passed, error = self.executor.execute_test(
                code,
                self.current_problem['test'],
                self.current_problem['entry_point']
            )
            reward = 1.0 if passed else 0.0
            rewards.append(reward)
        return torch.tensor(rewards, dtype=torch.float32).unsqueeze(-1)
```

For evaluation, we use pass@1 and pass@k accuracy metrics.

```python
def evaluate_model(model, tokenizer, problems, reward_fn,
                   num_samples=5, max_new_tokens=256,
                   temperature=0.7, top_p=0.9):
    """
    Evaluate model on problems
    """
    results = []
    model.eval()
    for problem in tqdm(problems, desc="Evaluating"):
        reward_fn.set_problem(problem)
        completions = []
        rewards = []
        with torch.no_grad():
            inputs = tokenizer(
                problem['prompt'],
                return_tensors="pt",
                padding=False # No padding for single sequence
            ).to(device)
            for _ in range(num_samples):
                outputs = model.generate(
                    inputs.input_ids,
                    attention_mask=inputs.attention_mask, # FIX: Pass attention mask explicitly
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                    pad_token_id=tokenizer.eos_token_id
                )
                completion = tokenizer.decode(
                    outputs[0][inputs.input_ids.shape[1]:],
                    skip_special_tokens=True
                )
                completions.append(completion)
        # Compute rewards
        reward_values = reward_fn([problem['prompt']] * num_samples, completions)
        rewards = reward_values.squeeze().tolist()
        if not isinstance(rewards, list):
            rewards = [rewards]
        # Compute metrics
        pass_at_1 = 1.0 if rewards[0] == 1.0 else 0.0
        pass_at_k = 1.0 if any(r == 1.0 for r in rewards) else 0.0
        avg_reward = np.mean(rewards)
        results.append({
            'task_id': problem['task_id'],
            'pass_at_1': pass_at_1,
            'pass_at_k': pass_at_k,
            'avg_reward': avg_reward,
            'rewards': rewards
        })
    # Summary statistics
    summary = {
        'pass_at_1': np.mean([r['pass_at_1'] for r in results]),
        'pass_at_k': np.mean([r['pass_at_k'] for r in results]),
        'avg_reward': np.mean([r['avg_reward'] for r in results]),
        'results': results
    }
    return summary
```

Evaluation on the base model resulted:

```python
baseline_results = evaluate_model(
    model, tokenizer, eval_problems, reward_fn,
    num_samples=SAMPLES_PER_PROBLEM,
    max_new_tokens=256,
    temperature=0.7,
    top_p=0.9
)
```

```
======================================================================
BASELINE RESULTS
======================================================================
Pass@1: 43.3%
Pass@5: 63.3%
Avg Reward: 0.427
```

## REINFORCE

Back to RL - we saw above that we can optimise a policy by gradient ascent in the direction of the policy gradient.

$$\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \left( \sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t | s_t) \right) R(\tau) \right]$$

In order to apply this machinery, we need:

1. Reward for the trajectory (completion)
2. Log probabilities of the actions (tokens)

The reward is based on whether the code actually solves the provided test cases. This is an example of the **Reinforcement Learning with Verifiable Rewards (RLVR)** framework where we use verifiable rewards instead of a trained reward model. The log probabilities are given by the logits of the tokens generated by the LLM.

## Policy Gradient Optimisation

First we implement the vanilla policy gradient optimisation:

- Generate N completions for each prompt using the current policy (model).
- Assign a reward to each completion based on the test-based reward functions.
- Compute the log probabilities for the tokens in each completion.
- Calculate loss as -log_prob * reward
- Backpropagate

Below is a simple implementation of policy gradient optimisation in plain PyTorch.

```python
@dataclass
class TrainingConfig:
    # Generation settings
    max_new_tokens: int = 40
    temperature: float = 0.8
    top_p: float = 0.9
    # Training settings
    num_epochs: int = 3
    prompts_per_epoch: int = 20 # Small for fast experimentation
    generations_per_prompt: int = 2 # Sample G completions per prompt
    learning_rate: float = 1e-6
    # Logging
    log_every: int = 5

config = TrainingConfig()

# ============================================
# Training loop
# ============================================
# Storage for logging
training_stats = {
    'step': [],
    'loss': [],
    'avg_reward': [],
    'avg_completion_length': [],
}
global_step = 0
for epoch in range(config.num_epochs):
    print(f"EPOCH {epoch + 1}/{config.num_epochs}")
    # Shuffle prompts each epoch
    epoch_prompts = np.random.choice(train_prompts,
        size=config.prompts_per_epoch, replace=False)
    for prompt_idx, prompt in enumerate(epoch_prompts):
        # ====================
        # Step 1: Generate completions (no gradients)
        # ====================
        all_completions = []
        all_sequences = []
        all_prompt_lengths = []
        with torch.no_grad():
            prompt_tokens = tokenizer(prompt, return_tensors="pt", padding=False)
            prompt_ids = prompt_tokens.input_ids.to(device)
            prompt_length = prompt_ids.shape[1]
            # Generate G completions for this prompt
            for g in range(config.generations_per_prompt):
                outputs = model.generate(
                    prompt_ids,
                    max_new_tokens=config.max_new_tokens,
                    do_sample=True,
                    temperature=config.temperature,
                    top_p=config.top_p,
                    pad_token_id=tokenizer.eos_token_id
                )
                full_sequence = outputs[0]
                completion_only = full_sequence[prompt_length:]
                completion_text = tokenizer.decode(completion_only,
                    skip_special_tokens=True)
                all_completions.append(completion_text)
                all_sequences.append(full_sequence)
                all_prompt_lengths.append(prompt_length)
        # ====================
        # Step 2: Compute rewards
        # ====================
        prompts_list = [prompt] * config.generations_per_prompt
        rewards = reward_fn(prompts_list, all_completions).to(device)
        # ====================
        # Step 3: Compute log probabilities (with gradients!)
        # ====================
        # Pad sequences to same length
        max_len = max(seq.shape[0] for seq in all_sequences)
        padded_ids = []
        completion_masks = []
        for seq, plen in zip(all_sequences, all_prompt_lengths):
            # Pad
            padding_length = max_len - seq.shape[0]
            padded = F.pad(seq, (0, padding_length), value=tokenizer.pad_token_id)
            padded_ids.append(padded)
            # Mask: 1 for completion tokens, 0 for prompt/padding
            mask = torch.zeros(max_len, dtype=torch.float32, device=device)
            mask[plen:seq.shape[0]] = 1.0
            completion_masks.append(mask)
        # Stack into batch
        input_ids = torch.stack(padded_ids)
        completion_mask = torch.stack(completion_masks)
        attention_mask = (input_ids != tokenizer.pad_token_id).long()
        # Forward pass WITH gradients
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        # Compute log probabilities
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()
        shift_mask = completion_mask[:, 1:].contiguous()
        log_probs = F.log_softmax(shift_logits, dim=-1)
        token_log_probs = log_probs.gather(
            dim=-1,
            index=shift_labels.unsqueeze(-1)
        ).squeeze(-1)
        # ====================
        # Step 4: Compute REINFORCE loss
        # ====================
        # Policy gradient: -log_prob * reward
        # Negative because PyTorch does gradient descent, we want ascent
        loss_per_token = (-token_log_probs * rewards) * shift_mask
        # Average over all tokens
        loss = loss_per_token.sum() / shift_mask.sum()
        # ====================
        # Step 5: Backprop and update
        # ====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # ====================
        # Logging
        # ====================
        global_step += 1
        avg_reward = rewards.mean().item()
        avg_length = np.mean([len(c.split()) for c in all_completions])
        training_stats['step'].append(global_step)
        training_stats['loss'].append(loss.item())
        training_stats['avg_reward'].append(avg_reward)
        training_stats['avg_completion_length'].append(avg_length)
        if global_step % config.log_every == 0:
            print(f"Step {global_step:3d} | Loss: {loss.item():7.4f} |"
                f"Avg Reward: {avg_reward:6.3f} |"
                f"Avg Length: {avg_length:5.1f} tokens")
            # Show one example
            print(f" Example completion: '{all_completions[0][:80]}...'")
```

This loop is fairly straightforward, except the log-probabilities calculation stumped me a bit. Let me explain this a bit.

### Step 1: Shifting for Next-Token Prediction

```
Original sequence: [tok0, tok1, tok2, tok3]
Model logits: [logit0, logit1, logit2, logit3]
Shift alignment:
shift_logits = [logit0, logit1, logit2] (predicting next token)
shift_labels = [tok1, tok2, tok3] (actual next tokens)
```

We remove the last logit and first label to align predictions with targets.

### Step 2: Log Softmax

```python
log_probs = F.log_softmax(shift_logits, dim=-1)
# Shape: (batch, sequence_length, vocab_size)
```

For each position, this gives log probabilities over ALL 100k+ vocab tokens.

### Step 3: Gather - Extract Log Prob of Actual Token

```python
token_logp = log_probs.gather(dim=-1, index=shift_labels.unsqueeze(-1))
```

What gather does:

- log_probs has probabilities for every possible token at each position
- shift_labels tells us which token was ACTUALLY generated
- gather picks out just those specific log probabilities

Visual example:

```
Position 0: log_probs = [0.1, 2.3, 0.5, ..., 5.2] (50k values)
shift_labels[0] = 42
token_logp[0] = log_probs[0, 42] = 0.5

Position 1: log_probs = [1.2, 0.3, 3.1, ..., 2.8]
shift_labels[1] = 7
token_logp[1] = log_probs[1, 7] = 0.3
```

token_logp = log probability the model assigned to each token that was actually generated. This is what we need for the policy gradient: “How likely was the model to generate these exact tokens?”

After training for 3 epochs with this setup, we evaluate on the test set.

```
======================================================================
VANILLA POLICY GRADIENT RESULTS
======================================================================
Pass@1: 43.3%
Pass@5: 73.3%
Avg Reward: 0.507
```

Comparing this with the base model, there is no change in Pass@1, but the Pass@5 improved by 10% and the average reward went from 0.427 to 0.507.

This is good, but the plain policy gradient suffers from training instability and has high variance in gradient updates. The policy gradient update that we use multiplies the gradient of the action probabilities (token log probabilities) by the total reward from a trajectory. An episode might have a high total reward just by pure luck, and the optimisation would blindly increase the probability of every action taken in that lucky episode, even the bad ones. This high variance in the gradient estimate makes learning very slow and unstable.

## REINFORCE

To mitigate the high variance problem of policy gradient methods, often some normalisation is applied to the trajectory rewards. A baseline is a function that is subtracted from the return in a policy gradient update to reduce the variance of the gradient estimate, leading to faster and more stable learning. The key is that this baseline must only depend on the state (s), not the action (a). The simplest baselines are averages over the batch of rewards or a moving average. This is the REINFORCE algorithm in its simplest form. The difference between the raw rewards and the baseline reward is called advantage of the trajectory - which essentially measures how much better taking action a in state s was compared to the average action from that state. The training loop becomes:

- Generate N completions for each prompt using the current policy (model).
- Assign a reward to each completion based on the test-based reward functions.
- Calculate the baseline and the advantage.
- Compute the log probabilities for the tokens in each completion.
- Calculate loss as -log_prob * advantage
- Backpropagate

The code for the training loop is modified to reflect this:

```python
# Step 2: Compute rewards (same as before)
prompts_list = [prompt] * baseline_config.generations_per_problem
rewards = reward_fn(prompts_list, all_completions).to(device)
rewards = rewards.squeeze(-1) # Shape: (G,)
# Step 3: NEW - Compute advantages with baseline subtraction
if baseline_config.use_baseline:
    baseline_value = rewards.mean()
    advantages = rewards - baseline_value
else:
    advantages = rewards
    baseline_value = 0.0
advantages = advantages.unsqueeze(-1) # Shape: (G, 1)
```

This change results in better training which shows in evaluation on test set:

```
======================================================================
REINFORCE RESULTS
======================================================================
Pass@1: 53.3%
Pass@5: 76.7%
Avg Reward: 0.507
```

This is a big improvement over the vanilla policy gradient approach.

## KL-divergence

When we fine-tune an LLM with RL, unconstrained reward maximisation tends to “run away” from the model’s pre-trained distribution: the policy exploits imperfections in the reward (or preference) model, over-optimises superficial signals, and rapidly forgets general linguistic competence - classic reward hacking and mode collapse. A KL penalty to a strong reference policy acts as a trust-region: it caps the step size in function space, stabilises learning, and preserves the valuable priors encoded during pre-training. Practically, this curbs distribution shift (keeping generations fluent and calibrated), reduces gradient variance, and forces the optimizer to earn reward improvements that are robust rather than brittle. The result is a principled bias–variance trade-off: with a tunable hyperparameter, we interpolate between safe imitation and genuine improvement which raises probability only where advantages are credible while anchoring everything else.

The KL divergence is a measure of how one probability distribution differs from a second, reference distribution. In this context, it measures the “distance” between the action probabilities of the fine-tuned policy and the original base model. In practice, we keep a frozen base model as the reference policy and add a KL cost against it while training the RL policy. For each prompt, we sample a completion from the current policy, score it with the reward model as before, calculate loss, and apply KL penalty to the loss.

- Generate N completions for each prompt using the current policy (model).
- Assign a reward to each completion based on the test-based reward functions.
- For each completion, compute its KL divergence from the reference policy.
- Adjust rewards with KL penalty - Compute adjusted rewards by subtracting KL penalty:
- Calculate the baseline and the advantage using adjusted rewards.
- Compute the log probabilities for the tokens in each completion using the policy,
- Calculate policy loss as -log_prob * advantage
- Backpropagate

```
======================================================================
REINFORCE + KL RESULTS
======================================================================
Pass@1: 56.7%
Pass@5: 66.7%
Avg Reward: 0.480
```

This shows a 3% improvement in Pass@1 over the REINFORCE version.

## REINFORCE Leave-One-Out (RLOO)

RLOO is a variance reduction technique which slightly alters the calculation of baseline in REINFORCE.

Standard REINFORCE with baseline:

```python
# Generate G completions for a prompt
completions = [comp_1, comp_2, ..., comp_G]
rewards = [r_1, r_2, ..., r_G]
# Compute baseline (mean of ALL rewards)
baseline = mean(r_1, r_2, ..., r_G)
# Compute advantage for EACH completion
advantage_1 = r_1 - baseline
advantage_2 = r_2 - baseline
...
```

Problem is that the baseline includes the current completion’s reward, which creates correlation and doesn’t reduce variance optimally.

REINFORCE Leave-One-Out (RLOO)

```python
# Generate G completions for a prompt
completions = [comp_1, comp_2, ..., comp_G]
rewards = [r_1, r_2, ..., r_G]
# For EACH completion, compute baseline EXCLUDING that completion
baseline_1 = mean(r_2, r_3, ..., r_G) # Leave out r_1
baseline_2 = mean(r_1, r_3, ..., r_G) # Leave out r_2
baseline_3 = mean(r_1, r_2, r_4, ..., r_G) # Leave out r_3
...
# Compute advantage for each
advantage_1 = r_1 - baseline_1
advantage_2 = r_2 - baseline_2
...
```

Key insight is that when computing the advantage for completion i, we use a baseline that’s independent of that completion’s reward. This reduces variance better!

Implementation is straightforward:

```python
def compute_rloo_advantages(rewards, normalize=True):
    """
    Compute REINFORCE Leave-One-Out advantages
    Args:
        rewards: Tensor of shape (G,) - rewards for G completions of same prompt
        normalize: Whether to normalize advantages
    Returns:
        advantages: Tensor of shape (G,)
    """
    G = len(rewards)
    if G == 1:
        # Can't do leave-one-out with single sample, return zero advantage
        return torch.zeros_like(rewards)
    advantages = []
    for i in range(G):
        # Compute baseline excluding i-th reward
        other_rewards = torch.cat([rewards[:i], rewards[i+1:]])
        baseline_i = other_rewards.mean()
        # Advantage for i-th completion
        advantage_i = rewards[i] - baseline_i
        advantages.append(advantage_i)
    advantages = torch.stack(advantages)
    # Optional: normalize advantages
    if normalize and len(advantages) > 1:
        if advantages.std() > 1e-8:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    return advantages
```

## Conclusion

Stepping back, this little expedition shows that even the most minimal RL recipe can move a pre-trained coder in useful ways. Scrappy, from the scratch implementation provides a deeper understanding of the working of the algorithm - where they fail, and what are mitigations. We explored three practical regimes, keeping things simple:

- Reward design matters: verifiable, test-based rewards are wonderfully crisp, but brittle extraction/execution can silently inject noise; getting the sandbox, timeouts, and code parsing right is part of the learning signal.
- Variance has been pitched as the enemy for training RL systems. It has been seen that this is less of a problem for fine-tuning LLMs. Nevertheless, simple ideas like batch average baselines stabilise training far more than extra epochs.
- Trust regions pay for themselves: the KL term is not decoration; it is the difference between a model that remembers how to write and one that chases reward hacks.

This study is intentionally small-budget (1.5B model, modest sampling, 50/30 split), so treat absolute numbers with care. The point isn’t SOTA; it’s mechanistic clarity: how to wire up REINFORCE for LLMs, where it breaks, and why simple fixes (baselines, KL) change the learning dynamics.

The broader takeaway is simple: SFT teaches what to say; RL teaches when and how to change it. With a good reward channel and a measured KL leash, even a bare-bones REINFORCE loop can translate verifiable feedback into better behavior at the pointy end of the ranking, without unlearning the language model you started with.

## Acknowledgement

In the middle of working on this post, I came across this excellent post by Cameron Wolfe on REINFORCE as an easy RL for LLMs. I have drawn a lot from this work. In fact, I would cast the current post as my personal notes and exercises after my reading of Cameron Wolfe’s article.