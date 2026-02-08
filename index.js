import "dotenv/config";
import express from "express";
import cors from "cors";
import OpenAI from "openai";

const app = express();
app.use(cors());
app.use(express.json({ limit: "2mb" }));

const PORT = process.env.PORT || 8787;

const client = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

// -------- helpers --------
function safeText(x) {
  return (x ?? "").toString().trim();
}

// 从模型输出里尽量提取 JSON（兼容 ```json ...``` 或前后夹杂文本）
function extractJsonObject(text) {
  const raw = safeText(text);
  if (!raw) return null;

  // 1) 尝试直接 parse
  try {
    return JSON.parse(raw);
  } catch {}

  // 2) 尝试提取 ```json ... ```
  const fenced = raw.match(/```json\s*([\s\S]*?)```/i) || raw.match(/```\s*([\s\S]*?)```/);
  if (fenced?.[1]) {
    try {
      return JSON.parse(fenced[1].trim());
    } catch {}
  }

  // 3) 从第一个 { 到最后一个 } 之间截取
  const start = raw.indexOf("{");
  const end = raw.lastIndexOf("}");
  if (start !== -1 && end !== -1 && end > start) {
    const slice = raw.slice(start, end + 1);
    try {
      return JSON.parse(slice);
    } catch {}
  }

  return null;
}

function normalizeAnalyzeResult(obj) {
  const o = obj && typeof obj === "object" ? obj : {};

  // 兼容旧字段
  const conclusion = safeText(o.conclusion || o.core_intent || o.analysis);
  const intentRadar = Array.isArray(o.intent_radar) ? o.intent_radar : Array.isArray(o.intent_hypotheses) ? o.intent_hypotheses : [];
  const landmines = Array.isArray(o.landmines) ? o.landmines : Array.isArray(o.risks) ? o.risks : [];

  const replies = o.replies && typeof o.replies === "object" ? o.replies : null;

  // 兼容旧 replies 数组 → 组装成 A/B/C
  let normalizedReplies = replies;
  if (!normalizedReplies && Array.isArray(o.replies)) {
    const arr = o.replies;
    normalizedReplies = {
      A_safe: { label: "稳妥保分", text: safeText(arr?.[0]?.text), effect: safeText(arr?.[0]?.effect), risk: safeText(arr?.[0]?.risk), why: "" },
      B_push: { label: "轻推进", text: safeText(arr?.[1]?.text), effect: safeText(arr?.[1]?.effect), risk: safeText(arr?.[1]?.risk), why: "" },
      C_avoid: { label: "不推荐", text: safeText(arr?.[2]?.text), effect: safeText(arr?.[2]?.effect), risk: safeText(arr?.[2]?.risk), why: "" },
    };
  }

  // 兜底：如果模型没给 replies，就至少给空结构避免前端炸
  if (!normalizedReplies) {
    normalizedReplies = {
      A_safe: { label: "稳妥保分", text: "", effect: "", risk: "", why: "" },
      B_push: { label: "轻推进", text: "", effect: "", risk: "", why: "" },
      C_avoid: { label: "不推荐", text: "", effect: "", risk: "", why: "" },
    };
  }

  return {
    conclusion,
    intent_radar: intentRadar.map((x) => ({
      name: safeText(x?.name || x?.title),
      confidence: Number.isFinite(Number(x?.confidence)) ? Number(x.confidence) : undefined,
      evidence: Array.isArray(x?.evidence) ? x.evidence.map(safeText).filter(Boolean) : [],
      disconfirm: Array.isArray(x?.disconfirm) ? x.disconfirm.map(safeText).filter(Boolean) : Array.isArray(x?.counter_evidence) ? x.counter_evidence.map(safeText).filter(Boolean) : [],
      verify_next: safeText(x?.verify_next || x?.verify || x?.next),
    })),
    landmines: landmines.map((r) => ({
      landmine: safeText(r?.landmine || r?.trigger || r?.title),
      consequence: safeText(r?.consequence),
      why: safeText(r?.why),
      fix: safeText(r?.fix || r?.suggestion),
    })),
    replies: {
      A_safe: {
        label: safeText(normalizedReplies?.A_safe?.label) || "稳妥保分",
        text: safeText(normalizedReplies?.A_safe?.text),
        effect: safeText(normalizedReplies?.A_safe?.effect),
        risk: safeText(normalizedReplies?.A_safe?.risk),
        why: safeText(normalizedReplies?.A_safe?.why),
      },
      B_push: {
        label: safeText(normalizedReplies?.B_push?.label) || "轻推进",
        text: safeText(normalizedReplies?.B_push?.text),
        effect: safeText(normalizedReplies?.B_push?.effect),
        risk: safeText(normalizedReplies?.B_push?.risk),
        why: safeText(normalizedReplies?.B_push?.why),
      },
      C_avoid: {
        label: safeText(normalizedReplies?.C_avoid?.label) || "不推荐",
        text: safeText(normalizedReplies?.C_avoid?.text),
        effect: safeText(normalizedReplies?.C_avoid?.effect),
        risk: safeText(normalizedReplies?.C_avoid?.risk),
        why: safeText(normalizedReplies?.C_avoid?.why),
      },
    },
    followup_question: safeText(o.followup_question),
  };
}

// -------- routes --------

app.post("/analyze", async (req, res) => {
  try {
    const body = req.body || {};

    const message = safeText(body.message_text) || safeText(body.message);
    if (!message) return res.status(400).json({ error: "message_text is required" });

    const scenarioType = safeText(body.scenario_type);
    const counterpartyRole = safeText(body.counterparty_role);
    const counterpartyRoleExtra = safeText(body.counterparty_role_extra);
    const relationshipStage = safeText(body.relationship_stage);
    const relationshipStageExtra = safeText(body.relationship_stage_extra);
    const userFeeling = safeText(body.user_feeling);
    const goal = safeText(body.goal);

    const prompt = `
你是一个成熟，知情达理，高情商，对人与人之间的关系情感非常敏感的分析师。你既可以理解人性，也能给出实用建议。
你可以像好闺蜜一样给女孩情绪价值，也可以像职场导师一样给出理性建议。你可以像情感主播一样判断出一个男孩是不是被他喜欢的对象PUA/渣了，或者吊着。也可以像一个知心的成熟的长辈/朋友，总给出最理性的建议。
你要输出【短、自然】的结果，让用户一眼能用。
禁止写长段解释；禁止“作文”；禁止道德说教。不要贴标签，不要PUA。

输入：
- 场景：${scenarioType || "未说明"}
- 对方身份：${counterpartyRole || "未说明"}${counterpartyRoleExtra ? "（补充：" + counterpartyRoleExtra + "）" : ""}
- 关系阶段：${relationshipStage || "未说明"}${relationshipStageExtra ? "（补充：" + relationshipStageExtra + "）" : ""}
- 用户感受：${userFeeling || "（无）"}
- 用户目标：${goal || "（无）"}
- 原文："""${message}"""

你只输出 JSON（无 markdown，无多余文字），结构如下（字段必须齐全）：
{
  "intent": {
    "primary": "对方更可能…（<=18字）",
    "confidence": 0,
    "alt": "也可能…（<=18字）",
    "verify_next": "一句可复制的验证话术（<=26字）"
  },
  "replies": {
    "A": { "text": "可直接发送（<=2句）", "watch_out": "一句提醒（<=12字）" },
    "B": { "text": "可直接发送（<=2句）", "watch_out": "一句提醒（<=12字）" },
    "C": { "label": "不推荐", "text": "常见翻车回法（<=2句）", "watch_out": "翻车点（<=12字）" }
  },
  "followup": "（可选）如果缺信息，问1个关键问题；否则空字符串"
}

补充规则：
- confidence 给 40~85 之间，不要 100。
- 回复的文字要像真人一样，不要那么“人机感”。要让用户觉得正常，感觉在和真人交互，理解用户的情绪和处境。
- watch_out 只写一句，像朋友提醒（例如：别解释太多/别太热/别硬刚）。
`.trim();

    const completion = await client.chat.completions.create({
      model: "gpt-4o",
      messages: [{ role: "user", content: prompt }],
      temperature: 0.6,
    });

    const content = completion.choices?.[0]?.message?.content || "";
    const parsed = extractJsonObject(content);
    if (!parsed) {
      return res.status(500).json({ error: "模型输出无法解析为JSON", raw: content.slice(0, 400) });
    }

    // 轻量规范化（避免缺字段导致前端炸）
    const out = {
      intent: {
        primary: safeText(parsed?.intent?.primary),
        confidence: Number.isFinite(Number(parsed?.intent?.confidence)) ? Number(parsed.intent.confidence) : 60,
        alt: safeText(parsed?.intent?.alt),
        verify_next: safeText(parsed?.intent?.verify_next),
      },
      replies: {
        A: {
          label: safeText(parsed?.replies?.A?.label) || "稳妥保分",
          text: safeText(parsed?.replies?.A?.text),
          watch_out: safeText(parsed?.replies?.A?.watch_out),
        },
        B: {
          label: safeText(parsed?.replies?.B?.label) || "轻推进",
          text: safeText(parsed?.replies?.B?.text),
          watch_out: safeText(parsed?.replies?.B?.watch_out),
        },
        C: {
          label: safeText(parsed?.replies?.C?.label) || "不推荐",
          text: safeText(parsed?.replies?.C?.text),
          watch_out: safeText(parsed?.replies?.C?.watch_out),
        },
      },
      followup: safeText(parsed?.followup),
    };

    res.json(out);
  } catch (err) {
    res.status(500).json({ error: String(err?.message || err) });
  }
});


// 续聊：基于初次结果 + 历史对话继续给“下一句怎么接/这句会不会太热”等
app.post("/chat", async (req, res) => {
  try {
    const body = req.body || {};
    const userInput = safeText(body.user_input);
    if (!userInput) return res.status(400).json({ error: "user_input is required" });

    const scenarioType = safeText(body.scenario_type);
    const counterpartyRole = safeText(body.counterparty_role);
    const relationshipStage = safeText(body.relationship_stage);
    const counterpartyRoleExtra = safeText(body.counterparty_role_extra);
    const relationshipStageExtra = safeText(body.relationship_stage_extra);
    const userFeeling = safeText(body.user_feeling);
    const goal = safeText(body.goal);


    const messageText = safeText(body.message_text);
    if (!messageText && !initialResult) {
  return res.status(400).json({ error: "missing conversation context" });
}

    const initialResult = body.initial_result ?? null;
    const chatHistory = Array.isArray(body.chat_history) ? body.chat_history : [];

    const trimmedHistory = chatHistory.slice(-12).map((m) => ({
      role: m?.role === "user" ? "user" : "assistant",
      content: safeText(m?.content),
    }));

    const system = `
你是一个成熟，知情达理，高情商，对人与人之间的关系情感非常敏感的分析师。你既可以理解人性，也能给出实用建议。
你可以像好闺蜜一样给女孩情绪价值，也可以像职场导师一样给出理性建议。你可以像情感主播一样判断出一个男孩是不是被他喜欢的对象PUA/渣了，或者吊着。也可以像一个知心的成熟的长辈/朋友，总给出最理性的建议。
你要输出【短、自然】的结果，让用户一眼能用。
禁止写长段解释；禁止“作文”；禁止道德说教。不要贴标签，不要PUA.
你要结合用户给的所有信息，客观理性回答用户最新追问。如果感受到用户有情绪，适当安抚。
注意语气要像真人聊天，不要“AI腔”。
`.trim();


    const user = `
【情境】
场景：${scenarioType || "未说明"}
对方身份：${counterpartyRole || "未说明"}${counterpartyRoleExtra ? `（补充：${counterpartyRoleExtra}）` : ""}
关系阶段：${relationshipStage || "未说明"}${relationshipStageExtra ? `（补充：${relationshipStageExtra}）` : ""}
${userFeeling ? `用户感受：${userFeeling}` : ""}
${goal ? `用户目标：${goal}` : ""}

【最初原文】
${messageText}

【初次分析结果（可用作一致性参考）】
${initialResult ? JSON.stringify(initialResult).slice(0, 2000) : "（无）"}

【续聊历史（最近）】
${trimmedHistory.map((m) => `${m.role === "user" ? "用户" : "军师"}：${m.content}`).join("\n")}

【用户最新追问】
${userInput}
`.trim();


    const completion = await client.chat.completions.create({
      model: "gpt-4o",
      messages: [
        { role: "system", content: system },
        { role: "user", content: user },
      ],
      temperature: 0.6,
    });

    const reply = safeText(completion.choices?.[0]?.message?.content);
    if (!reply) return res.status(500).json({ error: "empty reply from model" });

    res.json({ reply });
  } catch (err) {
    res.status(500).json({ error: String(err?.message || err) });
  }
});

app.get("/", (req, res) => {
  res.send("Reply server OK");
});

app.listen(PORT, () => {
  console.log(`Reply server running at http://localhost:${PORT}`);
});
