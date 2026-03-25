# AI Agent Assignment 2 – Single General-Purpose ReAct Agent

This project implements a **single Python-based ReAct agent** for Assignment 2: **Reasoning & Action Taking**.  
The agent follows a standard **Thought → Action → Observation** loop and is designed to demonstrate not only tool use, but also **planning** and partial **self-correction** behavior, as required by the assignment. 

## Project Goal

The assignment requires building a **single general-purpose ReAct agent** that can answer complex questions using:
- a real **Search API**
- a **few-shot system prompt**
- **stop logic** to prevent hallucinated observations
- a **loop limit**
- the **same ReActAgent instance** for all three benchmark questions 

This implementation uses:
- **Groq** as the LLM backend
- a search wrapper in `tools.py`
- one shared `ReActAgent` instance in `main.py` to answer all three benchmark questions in sequence 

---

## Features

- Single general-purpose `ReActAgent`
- Few-shot system prompt with multiple ReAct examples
- Standard **Thought → Action → Observation** loop
- Observation history fed back into the next LLM call
- `stop=["\nObservation:"]` to halt generation before hallucinated observations
- Hard loop limit (`max_steps=5`)
- Duplicate-search guard
- Arithmetic-search guard
- Invalid-format recovery
- Incomplete-final-answer feedback
- Same agent instance used to run all three benchmark tasks 

---

## Repository Structure

```text
.
├── agent.py              # ReAct agent class
├── tools.py              # Search tool wrapper
├── main.py               # Execution script
├── .env.example          # Environment variable template
├── requirements.txt      # Python dependencies
├── report.pdf            # Assignment report
├── RE18.txt              # Main benchmark trace used in report
├── RE14.txt              # Supplementary benchmark trace
└── README.md
```

The assignment specifically requires agent.py, tools.py, main.py, .env.example, requirements.txt, and report.pdf in the GitHub repository, and also requires .env to be excluded from version control.

## How It Works

The agent keeps a `history` list of previous **Thought / Action / Observation** steps.
After each search result is returned, the result is wrapped as:

```text
Observation:
...
```
and appended back into history, so the next LLM call can continue reasoning from previous steps instead of starting over.

The LLM is called with:
* model: `llama-3.1-8b-instant`
* temperature: `0`
* stop sequence: `["\nObservation:"]`

The benchmark in `main.py` uses the same `ReActAgent` instance to answer these three questions in order:

1. What fraction of Japan's population is Taiwan's population as of 2025?
2. Compare the main display specs of iPhone 15 and Samsung S24.
3. Who is the CEO of the startup 'Morphic' AI search?

## Environment Setup

Create a `.env` file in the project root.

## Minimum required

`agent.py` requires:
```txt
GROQ_API_KEY=your_groq_api_key_here
```

because the agent loads `GROQ_API_KEY` from `.env` before creating the Groq client.

## Search API key

You also need the search API key expected by your `tools.py` implementation.

If your `tools.py` uses Tavily (the recommended search API in the assignment), your `.env` may look like this:
```txt
GROQ_API_KEY=your_groq_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here
```
The assignment requires a real search API such as Tavily / Serper / DuckDuckGo.

## How to Run
1. Clone the repository
```txt
git clone <your-repo-url>
cd <your-repo-folder>
```
2. Create a virtual environment
Windows
```txt
python -m venv venv
venv\Scripts\activate
```
macOS / Linux
```txt
python3 -m venv venv
source venv/bin/activate
```
3. Install dependencies
```txt
pip install -r requirements.txt
```
4. Create `.env`

Copy `.env.example` to `.env`, then fill in your real API keys.

Windows PowerShell
```txt
copy .env.example .env
```

macOS / Linux
```txt
cp .env.example .env
```

Then edit `.env` and add your keys.

5. Run the benchmark script
```txt
python main.py
```

This will run one complete benchmark session using the same `ReActAgent` instance for all three tasks.

## Expected Output

Running python `main.py` prints a console trace like:

* `Task 1`
* `Question: ...`
* `--- Step 1 ---`
* `Thought: ...`
* `Action: Search["..."]`
* `Observation: ...`

and continues until the agent outputs a `Final Answer` or reaches the loop limit. `RE18.txt` is an example of such a full run.

## Benchmark Trace Files
`RE18.txt`

This is the main benchmark trace used in the report.
It records one complete run in which the same ReActAgent instance answers all three benchmark questions in sequence. The report uses it as the main console trace evidence.

`RE14.txt`

This is a supplementary trace kept for comparison and failure analysis.
It is not the primary trace used in Section 2 of the report, but it is useful for discussing instability and failure modes across reruns.

## Notes on Robustness

The agent includes several safeguards in agent.py:

* Stop logic: prevents the model from continuing into a hallucinated Observation
* Duplicate-search guard: blocks repeated or near-duplicate queries
* Arithmetic guard: prevents searching for pure arithmetic instead of computing directly
* Invalid-format recovery: handles malformed LLM output
* Incomplete-answer feedback: tells the model to reflect and try one genuinely new search if the answer does not actually answer the question

## Known Limitations

This project depends on a real external search API.
During later reruns, the search API quota was exhausted, so repeated benchmark runs could no longer be reproduced fairly under the same external conditions. Because of this, the report uses `RE18.txt` as the main benchmark evidence and keeps `RE14.txt` as supplementary evidence for failure analysis and stability comparison.

Also, benchmark quality can still be affected by:

* noisy multi-result search snippets
* mixed-year or mixed-statistic observations
* entity ambiguity
* source contamination (for example, iPhone 15 vs iPhone 15 Plus snippets)

## Assignment Deliverables

This repository is intended to contain the files required by the assignment:

* `agent.py`
* `tools.py`
* `main.py`
* `.env.example`
* `requirements.txt`
* `report.pdf`

Make sure:

`.env` is not committed
API keys are not leaked
`report.pdf` is included in the repository

## Reference

Assignment: Reasoning & Action Taking – Assignment 2
Focus areas:

* ReAct loop
* few-shot prompting
* stop sequences
* loop limits
* planning
* reflection / self-correction
* real search API integration
