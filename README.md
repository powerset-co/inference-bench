<p align="center"><img src="assets/banner.svg" alt="Providers: Sail, Wafer, Modal, Baseten, Fireworks, Together AI, Vultr, OpenCode Zen, OpenAI, Anthropic"></p>

# inference-bench

> [!NOTE]
> This repo is in beta. We're still iterating on the methodology, so the approach and results may change.
> These results show speed (time to complete a research task) and cost for a single agent, not performance
> under heavy load, and providers differ in ways these tests won't capture.

How fast and how cheap is each inference provider at serving an **AI research agent** using open-source LLMs?

We record each model doing a set of research tasks once, using questions from Google's
[FRAMES](https://huggingface.co/datasets/google/frames-benchmark) benchmark, then replay those
conversations on every provider. We measure **how long the model calls take** and **what they cost at
list price**. Every model provider gets the same set of requests for a given model and task, so we're testing how each
provider serves the model, not the model's ability to complete the task well.

**This is not a benchmark of model quality.** Every model here completes every task, and we don't check
whether its answers are right. The purpose of this benchmark is to compare timing and cost across
providers, not model intelligence.

## Results

<!-- results:start -->
<img src="results/charts/time-and-cost.svg" alt="time-and-cost" width="100%">

<img src="results/charts/latency.svg" alt="latency" width="100%">

<img src="results/charts/caching.svg" alt="caching" width="100%">

<sub>Last 30 days of runs: 2026-11-09 to 2026-11-11. Data: [summary](results/summary.csv), [raw results](results/).</sub>
<!-- results:end -->

## Methodology

Each model answers a set of multi-hop questions from [FRAMES](https://huggingface.co/datasets/google/frames-benchmark) as a research agent, with tools to search and
read Wikipedia. We only use tasks that every model is able to finish, and save each model's conversations
along with the Wikipedia responses.

The benchmark replays those conversations. For each turn, we send the conversation up to that point to the
provider, time the response, and move on to the next recorded turn. The Wikipedia results come from the recording,
so the timings only include model calls. Each model works through the tasks differently, so results are only comparable
between providers serving the same model.

Every provider gets the same settings for a given model (minimal reasoning, and temperature 0). After an untimed warm-up request, each provider handles one request at a time. Time spent waiting
on rate limits isn't counted. Caching works normally within a task, and to see how much it helps, we also
resend each task's last turn without the conversation cached.

Each run goes through every task several times. A task that fails gets retried, and we only publish a run if
every task finished. We run the benchmark monthly and on request, and the charts average every run from the last
30 days. Some runs only cover one provider, so providers may be measured on different days.

Cost is the list price for the tokens each provider reports. Credits and discounts aren't included.

## Usage

```sh
npm install
cp .env.example .env              # add the keys you have; providers without keys are skipped
npm run bench                     # replay every recorded task on every provider
npm run report                    # update the charts above and results/summary.csv
```

## Notes and limitations

- OpenAI and Anthropic models are included for reference.
- Sail has three tiers (`asap`, `balanced`, `flex`) at different prices. `flex` only works through the
  Responses API, while everything else here uses Chat Completions. It doesn't stream, so its time isn't
  split into waiting and generating. On it, Kimi K3.1 Pro often ends a turn without making the tool call it
  describes, which would stop a real agent, so its numbers may be slightly optimistic.
- OpenCode Zen is a gateway rather than an inference provider. It forwards each request to providers it
  doesn't name, so its numbers include that routing.
- Spot tiers, like Fireworks', are left out. Their capacity varies too much to measure fairly in a single run.
- Providers may run the same model differently, for example at a different precision. That's part of what's
  compared.
- Each provider gets one request at a time, like a single agent, so this doesn't measure performance under
  heavy load.
- Research agents have moderate context lengths. Coding agents, with much longer contexts, would need their
  own workload.
- We use self-serve, pay-as-you-go accounts. Enterprise plans may get dedicated capacity or higher limits.
- The benchmark runs on a GitHub-hosted runner, so network distance to each provider varies.
- Some providers don't stream reasoning, so it shows up as waiting for the first token. Total time is still
  comparable.

## Contributing

Contributions are welcome. Feel free to open an issue or PR. To ask for updated results for a provider,
[open a request](https://github.com/powerset-co/inference-bench/issues/new?template=updated-results.yml).

Code is MIT licensed. Wikipedia text in `traces/` is from [Wikipedia](https://en.wikipedia.org)
([CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/)), and the questions are from Google's FRAMES
dataset (Apache 2.0). Provider logos are trademarks of their owners, used here only to identify each provider.
