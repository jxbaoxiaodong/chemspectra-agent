# ChemSpectra Agent — Voiceover Script (v3)

Total duration: 180 seconds (~380 words, ~2.1 words/second)

---

## Scene 1 — Title (0:00–0:07)
*[No voiceover. Music/ambient only.]*

---

## Scene 2 — Identity + Origin (0:07–0:22)
> I'm not a software engineer. I'm a materials science graduate student.
>
> FTIR spectral analysis takes 30 to 60 minutes per sample — reading peaks, searching libraries, cross-checking literature. In 2024, that pain pushed me to teach myself Python. The need came from the lab. AI came second.

---

## Scene 3 — Architecture (0:22–0:47)
> ChemSpectra Agent is a multi-round self-verifying ReAct agent powered by Qwen-Max through the Alibaba Cloud dashscope SDK. Five specialized tools are exposed via Function Calling. But unlike a typical API wrapper, this agent doesn't just call tools and merge results. After the initial reasoning loop — think, act, observe — it runs automated cross-validation to detect evidence conflicts, then triggers a self-verification round when confidence is low. If any LLM output is malformed, self-repair kicks in automatically. The platform also exposes an MCP endpoint.

---

## Scene 4 — Live Demo (0:47–1:35)
> Let me show you. A user submits peaks at 2920, 1720, and 1230 wavenumbers, asking to identify the material.
>
> The agent calls dashscope with our tool definitions. Qwen-Max returns three tool calls: identify material, explain peaks, and assign functional groups. Each tool hits the FTIR.fun API, backed by 130,000 reference spectra.
>
> The initial synthesis identifies PET with confidence 0.72 — below the 0.75 threshold. Cross-validation flags this automatically. The agent launches a verification round, calling explain_peaks again with focused parameters to confirm the ester linkage.
>
> After verification, confidence rises from 0.72 to 0.94 — a 30% improvement through autonomous investigation, not human intervention. Then the system pauses for human confirmation.

---

## Scene 5 — Innovation (1:35–2:00)
> Here's what makes this different. A typical API wrapper calls tools once and returns results — one to two LLM calls, no verification. ChemSpectra Agent runs a full reasoning loop with evidence cross-validation, automatic self-verification when confidence drops, self-repair on output failures, and a human-in-the-loop gate. Three to six adaptive LLM calls per analysis — the agent decides how deep to go.

---

## Scene 6 — Impact (2:00–2:35)
> This isn't a demo built last week. FTIR.fun is a production platform — 130,000 reference spectra, 28 file formats, users in 52 countries.
>
> I started self-teaching Python in 2024. The platform launched in 2025. Today it has paying users worldwide. The best AI applications come from people who understand the problem deeply.

---

## Scene 7 — Closing (2:35–3:00)
> ChemSpectra Agent: Qwen-Max via Alibaba Cloud dashscope, multi-round reasoning with self-verification and self-repair, five autonomous tools with evidence cross-validation, MCP integration, backed by 130,000 production spectra.
>
> Built by a domain expert. Powered by AI. The code is open source on GitHub.
>
> Thank you.

---

## Recording Notes
- Pace: moderate, ~2.1 words/second. Pause at scene transitions.
- Tone: confident and measured. Slightly personal in Scene 2, authoritative in Scenes 3-4.
- Language: English (competition requirement).
- Record in a quiet room, close to mic. Export WAV 48kHz.
- Scene 4 (demo) is the longest — emphasize the confidence improvement (0.72 → 0.94).
- Scene 5 — contrast should be clear: "typical wrapper" vs "ChemSpectra's multi-round approach."
