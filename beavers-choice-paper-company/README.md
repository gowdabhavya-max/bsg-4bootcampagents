# Beaver's Choice Paper Company: Multi-Agent Quoting, Inventory and Sales System

A text-in, text-out multi-agent system built with [smolagents](https://huggingface.co/docs/smolagents/index).
It answers inventory questions, quotes orders with bulk discounts informed by past quotes, buys from the
supplier when stock is short, and completes the sale in a SQLite database, or declines it with a clear reason
when it cannot be done in time.

## Agents (4)

| Agent | Owns |
|---|---|
| Orchestrator | Understands the request, orders the work, writes the customer reply |
| Inventory agent | Stock knowledge and all supplier purchasing |
| Quoting agent | Prices and discounts (uses historical quotes) |
| Ordering agent | Final checks and the sale record |

Every tool belongs to exactly one agent. Business rules (pricing, stock arithmetic, delivery deadlines) live in
deterministic Python tools, so the language model only routes work and explains results.
See `agent_workflow_diagram.png` for the tools and the starter helper functions each one uses.

## Files

| File | Purpose |
|---|---|
| `project_starter.py` | The complete implementation (starter helpers plus the multi-agent system) |
| `agent_workflow_diagram.png` | Workflow diagram: agents, tools, helper functions, data flow |
| `Beavers_Choice_Multi_Agent_Report.docx` | Design, decisions, evaluation and improvement suggestions |
| `test_results.csv` | Output of the evaluation run on the 20 sample requests |

## Results (final run on `quote_requests_sample.csv`)

- 20 of 20 requests processed
- 9 fulfilled, 11 not fulfilled, each with a stated reason (mostly supplier delivery after the customer's date, or an item that is not sold)
- Cash balance changed on 13 of 20 rows

## Running it

1. Put `project_starter.py` and the three course data files (`quote_requests.csv`, `quotes.csv`,
   `quote_requests_sample.csv`) in one folder. The data files are course material and are not included here.
2. Install the packages: `pip install smolagents pandas numpy sqlalchemy python-dotenv openai`
3. Provide your API key through the environment or a `.env` file, never in the code:
   `UDACITY_OPENAI_API_KEY=<your key>`
4. Run `python project_starter.py`. It creates `munder_difflin.db`, processes every request in date order
   and writes `test_results.csv`.
