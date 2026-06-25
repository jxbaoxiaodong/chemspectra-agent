# ChemSpectra Agent — 演示视频制作方案

> 时长: ≤3 分钟 | 上传平台: YouTube（公开） | 语言: 英文旁白 + 屏幕录制

---

## 一、核心策略

评审四个维度的得分权重决定了视频每一秒该展示什么：

| 维度 | 权重 | 视频中如何体现 | 分配时长 |
|------|------|----------------|----------|
| Technical Depth | 30% | 展示 Agent 自主选择不同工具组合 + 代码闪现 | ~40s |
| Innovation | 30% | 对比「固定流水线 vs 自主路由」，强调"Agent 自己决定调哪些工具" | ~25s |
| Impact | 25% | 开场讲真实痛点 + 结尾讲已有收入 | ~25s |
| Presentation | 15% | 全程 UI 录屏流畅、过渡自然 | 贯穿全片 |

**一句话定位**: "不是 API 套壳，是一个会思考该用哪把工具的化学分析专家。"

**叙事主线**: 一个材料学研究者，不是程序员，因为每天重复 30-60 分钟的手工光谱分析而自学编程、借助 AI 构建了这个工具——需求从实验室里长出来，不是从 hackathon 题目里编出来。这条线在开头建立，在结尾收束（已有付费用户），贯穿全片的可信度。

---

## 二、分镜脚本

### Scene 1 — 开场: 身份 + 痛点 + 动机 (0:00 – 0:30)

**画面**: 屏幕先显示一张 FTIR 光谱图（从 FTIR.fun 截图），3 秒后切到标题卡。

**旁白**:
> I'm not a software engineer — I'm a materials scientist. Every day in the lab, identifying a sample by its infrared spectrum means 30 to 60 minutes of manual work: reading peaks, searching libraries, cross-checking literature, writing a report.
>
> That pain point pushed me to teach myself programming and build a tool to automate the whole process. This is ChemSpectra Agent.

**要点**:
- **第一句话就讲"我不是程序员"**——立刻和所有其他参赛者区分开。评审会想："那你是怎么做出来的？"产生好奇心
- "That pain point pushed me to teach myself" 一句话建立了完整叙事：痛点 → 自学 → 产品。不需要展开细节
- 不要说"我用 AI 辅助编程"——这个信息在结尾讲更有力。开场只建立"领域专家 + 真实痛点"
- 标题卡上写: **ChemSpectra Agent — AI Autopilot for FTIR Spectral Analysis**，下面一行: Track 4: Autopilot Agent

---

### Scene 2 — 演示: 文件上传 + Agent 启动 (0:30 – 0:50)

**画面**: 浏览器打开 `localhost:8080`，展示 Web UI。

**操作步骤**:
1. 鼠标点击「Spectrum File」，选择一个 `.csv` 光谱文件
2. 在「Sample Description」输入: `Unknown white powder, need full identification`
3. 「Analysis Type」选 `Identify unknown material`
4. 点击「Analyze Spectrum」

**旁白**:
> Here's the web UI. I upload a real spectrum file — we support 28 formats — describe the sample, and hit Analyze.
>
> Now watch what the Agent decides to do.

**要点**:
- 操作要慢、鼠标动作清晰，让评审看到每一步
- "28 formats" 一句带过即可，不用展开

---

### Scene 3 — Agent 工作过程: 多工具选择（核心卖点）(0:50 – 1:40)

**画面**: UI 左侧 Pipeline 进度条从 Step 1 开始动画推进。

**旁白 (Step 1 动画时)**:
> Step one: the Agent reasons about my request using Qwen-Max. It has five specialized tools available — material identification, peak explanation, functional group assignment, library matching, and public result search.
>
> Instead of always calling the same API, it decides WHICH tools to use based on what I asked.

**画面**: 等待结果返回（约 15-30 秒），这段时间旁白持续。如果等待太长，可在剪辑时加速中间空白。

**旁白 (等待中)**:
> Under the hood, this is a ReAct loop — Reason, then Act. Qwen reads my request, picks the right tools via Function Calling, executes them against a production library of 130,000 reference spectra, then synthesizes all results.

**画面**: 结果页出现。重点指向:
1. **"Tools Used by Agent" 标签区** — 显示 Agent 选了哪些工具（如 `Material ID` + `Peak Explain` + `Func Groups`）
2. **最佳匹配结果** — 物质名 + CAS 号 + 分数
3. **Agent Synthesis** — 综合分析文本

**旁白**:
> For this sample, the Agent chose three tools: material identification, peak explanation, and functional group assignment. It identified the sample as 4-nitrobenzaldehyde with 71% confidence, and flagged it for human review because the score is below 80%.

**要点**:
- **"Tools Used by Agent" 标签区是本视频最关键的画面** — 这是和其他参赛者拉开差距的地方。请放慢速度，让评审看清
- 如果结果加载太慢，可以预先录好一次成功的结果页面，剪辑时无缝衔接
- "130,000 reference spectra" 要说出来——这不是 toy data

---

### Scene 4 — 对比: 不同问法 → 不同工具组合 (1:40 – 2:05)

**画面**: 快速切到第二次分析。这次不上传文件，直接输入峰位。

**操作步骤**:
1. 清空表单（点 New Analysis）
2. 在「Peak Positions」输入: `1730, 1250, 1100`
3. 在「Sample Description」输入: `What functional groups correspond to these peaks?`
4. 点击 Analyze

**画面**: 结果返回后，指向 Tools Used 标签区——这次显示的是 `Func Groups` + `Peak Explain`（不同于上一次的三工具组合）。

**旁白**:
> Now I ask a different question — just "what functional groups are these peaks?" — and the Agent picks completely different tools. This time: functional group assignment and peak explanation. No material identification, because I didn't ask for it.
>
> That's the key innovation: the Agent reasons about your intent, not just follows a script.

**要点**:
- 这个对比场景直接打 Innovation 30% 的分——评审一眼看出 Agent 在做决策
- 两次分析的工具标签不同，视觉冲击力很强

---

### Scene 5 — Human-in-the-Loop + 追问 (2:05 – 2:25)

**画面**: 回到第一次分析的结果页（或用当前页面），在聊天框中追问。

**操作步骤**:
1. 在 Chat 输入框输入: `Why is the confidence score low?`
2. 点击 Send
3. 等 Agent 回复

**旁白**:
> Before I accept the results, I can ask follow-up questions. The Agent answers with chemical context — it's not a separate chatbot, it remembers the full analysis.
>
> In regulated industries like pharma or forensics, this human-in-the-loop step is not optional — AI cannot make the final call alone.

**要点**:
- "regulated industries" 这个点很重要——说明 HITL 不是噱头，是行业合规要求
- 追问内容选化学相关的，展示 Agent 有化学推理能力

---

### Scene 6 — 报告生成 + 下载 (2:25 – 2:40)

**画面**: 点击「Accept & Generate Report」，等报告出现，再点「Download Report」。

**旁白**:
> Once I confirm, the Agent generates a structured report — including which tools were used, the chemical reasoning, and confidence assessment. One click to download as Markdown.

**操作步骤**:
1. 点击「Accept & Generate Report」
2. 等报告渲染在页面上（展示 2-3 秒让评审扫一眼内容）
3. 点击「Download Report」

**要点**:
- 报告内容里有「Tools Used in Analysis」这一节，呼应多工具主题
- 展示速度要快，不在这里停留太久

---

### Scene 7 — 收尾: 回扣主线 + 影响力 (2:40 – 3:00)

**画面**: 切到一张预先准备的总结画面（可以是 terminal 全屏显示几行关键信息，或一张简洁的 slide）。

总结画面内容建议:
```
ChemSpectra Agent
━━━━━━━━━━━━━━━━━━━━━━━
Qwen-Max (dashscope SDK) — reasoning + tool selection
FTIR.fun API            — 130,000+ reference spectra
5 tools                 — autonomous selection via Function Calling
Human-in-the-loop       — regulated industry compliance

github.com/jxbaoxiaodong/chemspectra-agent
```

**旁白**:
> ChemSpectra Agent runs on Qwen-Max through Alibaba Cloud's dashscope SDK. The spectral library behind it is real — 130,000 spectra powering a platform that already has paying users.
>
> I started this project because I was tired of doing the same manual analysis every day in the lab. AI made it possible for a domain expert — not a developer — to build a production tool that solves a real problem. Thank you.

**要点**:
- 结尾回扣开头的叙事线：从"我不是程序员，但痛点逼我自学"到"AI 让领域专家能自己造工具"——这是一个完整的故事弧
- "dashscope SDK" 和 "Alibaba Cloud" 必须说出来——阿里云使用证明的口头确认
- "already has paying users" 是 Impact 维度的杀手锏——大多数 hackathon 项目做完比赛就扔了，你的不是
- "domain expert, not a developer" 最后再强调一次，让评审带着这个印象结束观看
- GitHub 链接在画面上停留到视频结束

---

## 三、录制前准备清单

### 素材准备

| 项目 | 说明 | 状态 |
|------|------|------|
| 测试光谱文件 | 用一个能稳定出结果的 `.csv` 文件，提前跑通确认 | 需准备 |
| 标题卡画面 | 项目名 + Track 4 + 你的身份，简洁背景 | 需制作 |
| 总结画面 | Scene 7 的技术栈摘要 + GitHub 链接 | 需制作 |
| 旁白录音 | 提前朗读几遍，确保 3 分钟内说完 | 需录制 |

### 环境准备

1. **启动 FTIR.fun 后端** — 确认 `http://127.0.0.1:18080/health` 返回正常
2. **启动 ChemSpectra Agent** — 设好环境变量后运行 `python server.py`
3. **浏览器** — 用 Chrome，字体放大到 125%（方便评审看清），关掉书签栏和多余标签页
4. **录屏软件** — macOS 自带 QuickTime 或 OBS，分辨率 1920×1080，帧率 30fps
5. **预跑一遍** — 确认从上传到报告生成全流程不出错，记录实际耗时
6. **清空 terminal** — 录制前 `clear`，不要暴露 API key 或私人路径

### 时间控制

| 场景 | 内容 | 目标时长 | 累计 |
|------|------|----------|------|
| Scene 1 | 身份 + 痛点 + 动机 | 30s | 0:30 |
| Scene 2 | 文件上传 + 启动 | 20s | 0:50 |
| Scene 3 | Agent 多工具选择（核心卖点） | 50s | 1:40 |
| Scene 4 | 对比: 不同问法 → 不同工具 | 25s | 2:05 |
| Scene 5 | HITL 追问 | 20s | 2:25 |
| Scene 6 | 报告生成 | 15s | 2:40 |
| Scene 7 | 回扣主线 + 影响力 | 20s | 3:00 |

如果某段超时，**优先压缩 Scene 3 的等待时间**（加速处理）和 **Scene 6**（报告展示可以更快带过）。Scene 1 和 Scene 7 是叙事骨架，不要压缩。

---

## 四、剪辑要点

1. **加速 Agent 等待时间** — API 调用可能需要 15-30 秒，视频中可以剪辑加速到 3-5 秒，画面右下角加一个 "⏩ 2x speed" 角标
2. **不加背景音乐** — 规则禁止第三方版权音乐，纯旁白更专业
3. **关键画面加标注** — 当 "Tools Used by Agent" 标签出现时，用红色箭头或高亮框指向它
4. **代码闪现（可选）** — 如果 Scene 3 等待时间有富余，可以切 0.5 秒的 `agent.py` 代码画面（`AGENT_TOOLS` 定义或 `run_tool_loop` 函数），然后立刻切回 UI。目的是让评审知道你写了真代码
5. **字幕** — 全程加英文字幕。旁白说到技术术语时，字幕用醒目颜色

---

## 五、旁白完整稿（英文）

以下是可以直接照读的完整旁白，控制在 2 分 50 秒以内。斜体为画面提示，方括号为语气/节奏标注。

---

*[FTIR 光谱图画面，2 秒后切到标题卡]*

I'm not a software engineer — I'm a materials scientist. [停顿 1 秒] Every day in the lab, identifying a sample by its infrared spectrum means 30 to 60 minutes of manual work: reading peaks, searching libraries, cross-checking literature, writing a report.

That pain point pushed me to teach myself programming and build a tool to automate the whole process. This is ChemSpectra Agent.

*[切到浏览器 UI]*

Here's the web UI. I upload a real spectrum file — we support over 28 formats — describe the sample, and hit Analyze.

Now watch what the Agent decides to do.

*[Pipeline 动画推进]*

The Agent has five specialized tools: material identification, peak explanation, functional group assignment, library matching, and public result search.

Instead of always calling the same API, it decides which tools to use based on what I asked. Under the hood, this is a ReAct loop — Reason, then Act. Qwen-Max reads my request, picks the right tools via Function Calling, executes them against a production library of 130,000 reference spectra, then synthesizes all the results.

*[结果页出现，指向 Tools Used 标签]*

For this sample, the Agent chose three tools: material identification, peak explanation, and functional group assignment. It identified the sample as 4-nitrobenzaldehyde with 71% confidence, and flagged it for human review.

*[点击 New Analysis，输入峰位]*

Now I ask a different question — just peak positions and "what functional groups are these?" The Agent picks completely different tools this time: functional group assignment and peak explanation. No material identification, because I didn't ask for it.

That's the key: the Agent reasons about your intent, not just follows a script.

*[回到结果页，在 Chat 输入追问]*

Before I accept, I can ask follow-up questions. The Agent answers with chemical context from the full analysis. In regulated industries like pharma or forensics, this human-in-the-loop step is required — AI cannot make the final call alone.

*[点击 Accept，报告出现]*

Once confirmed, a structured report is generated with multi-tool evidence. One click to download.

*[切到总结画面]*

ChemSpectra Agent runs on Qwen-Max through Alibaba Cloud's dashscope SDK. The spectral library behind it is real — 130,000 spectra powering a platform that already has paying users.

[放慢语速] I started this project because I was tired of doing the same manual analysis every day. AI made it possible for a domain expert — not a developer — to build a production tool that solves a real problem. Thank you.

---

## 六、配音方案

**采用方案: 本人英文旁白录制**

理由：视频的叙事主线是"我不是程序员，我是材料学研究者"——这句话由你本人说出来，可信度和感染力远超 AI 合成语音。口音完全不影响，Devpost 评审听过各种英语，清晰即可。

### 录音要点

1. **设备** — 手机录音即可，找安静房间，离麦克风一拳距离
2. **节奏** — 按稿子读，语速偏慢。宁可说慢点剪辑加速，也不要赶着说含糊
3. **分段录** — 不需要一口气读完。按 Scene 分段录，后期剪辑对齐画面。说错了重来该段就行
4. **重音** — 以下词句需要稍微加重或放慢，是给评审的记忆锚点：
   - "I'm **not** a software engineer"
   - "**130,000** reference spectra"
   - "the Agent **chose three tools**"
   - "picks **completely different** tools"
   - "a **domain expert** — not a developer"
5. **导出格式** — MP3 或 M4A，后期用剪辑软件和录屏合成

### 备选方案（仅当英文录制确实困难时）

| 方案 | 做法 | 代价 |
|------|------|------|
| 中文旁白 + 英文字幕 | 用中文讲，剪辑时烧录英文字幕 | 字幕制作有工作量，且评审需要边看边读，注意力分散 |
| AI 语音合成 | 用 Edge TTS 或通义语音把英文稿转音频 | 发音标准但缺少真人感，和"我是领域专家"的叙事矛盾 |
