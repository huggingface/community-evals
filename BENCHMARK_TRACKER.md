# Benchmark Status Tracker

Last updated: 2026-03-16

## Active Benchmarks (on Hub)

| Benchmark | Hub Dataset ID | Status | Owner/Contact | Scores | Notes |
|-----------|---------------|--------|---------------|--------|-------|
| HLE | [cais/hle](https://huggingface.co/datasets/cais/hle) | ✅ Active | CAIS | High | Core benchmark, appears on most new releases |
| GPQA | [Idavidrein/gpqa](https://huggingface.co/datasets/Idavidrein/gpqa) | ✅ Active | Rein et al. | High | Core benchmark |
| MMLU-Pro | [TIGER-Lab/MMLU-Pro](https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro) | ✅ Active | TIGER-Lab | High | Core benchmark |
| Terminal-Bench 2.0 | [terminal-bench/terminal-bench-2.0](https://huggingface.co/datasets/terminal-bench/terminal-bench-2.0) | ✅ Active | Terminal-Bench | Growing | Added scores recently |
| OlmOCRBench | [allenai/olmOCR-bench](https://huggingface.co/datasets/allenai/olmOCR-bench) | ✅ Active | AllenAI | Growing | Rednote/dots.ocr SOTA drama — community org created |
| mteb/arguana | [mteb/arguana](https://huggingface.co/datasets/mteb/arguana) | ✅ Active | MTEB/Tom Aarsen | 10+ | Recently activated, MTEB adding results. Task dropdown was missing (now fixed?) |
| SWE-bench Verified | [SWE-bench/SWE-bench_Verified](https://huggingface.co/datasets/SWE-bench/SWE-bench_Verified) | ⚠️ Careful | SWE-bench | Some | Only use official harness results. SWE-bench author pushed back on model card sourcing |
| GSM8K | [openai/gsm8k](https://huggingface.co/datasets/openai/gsm8k) | ✅ Active | OpenAI | High | Legacy but still tracked |

## In Progress

| Benchmark | Hub Dataset ID | Status | Owner/Contact | Blocker | Notes |
|-----------|---------------|--------|---------------|---------|-------|
| compute-eval | [nvidia/compute-eval](https://huggingface.co/datasets/nvidia/compute-eval) | 🔄 Ready | Nvidia | Bertrand to flip switch | eval-framework field validation issue resolved? |
| LiveCodeBench | livecodebench/??? | 🔄 Blocked | Nvidia/Quentin | Loading script needs removal | Quentin's PR didn't convince author |
| AIME 2026 | ??? | 🔄 Pending | — | Need to ask providers to report | aime26 on hub (not 25) |

## Outreach / Prospective

| Benchmark | Source | Status | Contact | Notes |
|-----------|--------|--------|---------|-------|
| TaxCalcBench | [x.com post](https://x.com/michaelrbock/status/2029931536636858694) | 📧 In touch | Niels → author DM | Cool benchmark: can agent calculate your taxes? Get data on hub + eval open models |
| EyeBench v3 | [x.com post](https://x.com/adonis_singh/status/2031761832390672487) | 📧 Reached out | Niels | v2 saturated, v3 just launched. Want scripts + data on hub |
| PostTrainBench | [posttrainbench.com](https://posttrainbench.com/) | 👀 Watching | — | CLI agent improves post-training. Interesting but niche |
| BullshitBench | [x.com post](https://x.com/petergostev/status/2026396163637731794) | 👀 Watching | — | Fun vibes, community appeal |
| SWE-Rebench | Nebius | ❌ Low engagement | Niels pinged | Not responding |
| Scale AI | Scale | ❌ Low engagement | Niels pinged | Not responding |
| KernelBench | Stanford | 👀 Exploring | Ben (via #kernels) | Want on hub. Mixed feelings in GPU Mode working group about "yet another standard" |
| FINAL-Bench | [Space](https://huggingface.co/spaces/FINAL-Bench/all-bench-leaderboard) | 👀 Interesting | — | Aggregates many benchmarks. Goal: have Spaces like this aggregating hub leaderboards |

## Key Decisions & Strategy

- **Focus:** Grow leaderboard *consumers* and *model reporters* first, before benchmark creators
- **Playbook:** Model release + leaderboard image on X (Qwen3.5 got 80k impressions) — repeat monthly
- **Reporting:** Always report as "author" for consistency. Be prepared to change.
- **Verification:** Exploring "verified" tag for PRs opened by benchmark owners
- **Aggregation:** Per-benchmark data first. Users can aggregate via API. (Tom Aarsen's position)
- **Limitation noted:** Lower scores for a model not shown on leaderboard (by design?)
- **MTEB:** 135 datasets too many — focus on impactful ones, not all (Bertrand)

## Recent Activity (Last 2 Weeks)

- **Mar 14-16:** Rednote/dots.ocr situation — model on ModelScope not Hub, team unresponsive. Created `rednote-dots-ocr-community` org. Niels shared OlmOCRBench post + local run video.
- **Mar 12:** Niels reaching out to Nemotron-3 benchmarks, AIME26, SWE-bench harness questions
- **Mar 10:** Nathan update — used Ben's skills to fetch from papers, MTEB arguana results added, Nvidia compute-eval PR validation issue
- **Mar 7:** Hackathon-adjacent discussion, embedding-atlas MCP support for benchmark analysis
- **Mar 1:** Nightly polling / webhook idea (Ben → Avijit, not yet delivered)
- **Feb 27:** Qwen3.5 post — 80k impressions. Strategy: repeat monthly.

## Open Actions

| Owner | Action | Since | Status |
|-------|--------|-------|--------|
| Ben/Dave | Nightly polling job example for Avijit | Feb 20 | 🔨 Building now |
| Ben | Explore LCB coding benchmark | Feb 20 | Open |
| Ben | Repeat model release + leaderboard pattern | Feb 27 | Need next big model release |
| Bertrand | compute-eval → benchmark activation | Mar 7 | Ready to flip |
| Niels | Scale AI follow-up | Mar 10 | Low engagement |
| Niels | SWE-Rebench (Nebius) follow-up | Mar 7 | Low engagement |
| Niels | TaxCalcBench author DM | Mar 7 | In progress |
| Nathan | Remove SWE-bench model card results | Mar 5 | Pending |
| Quentin | LiveCodeBench loading script | Mar 7 | Blocked on author |
