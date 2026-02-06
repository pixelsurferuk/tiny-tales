// server/index.js
import "dotenv/config";
import express from "express";
import cors from "cors";
import OpenAI from "openai";
import { createClient } from "@supabase/supabase-js";

const app = express();
app.use(cors());
app.use(express.json({ limit: "12mb" }));

// ======================
// 🔧 CONFIG
// ======================
const CONFIG = {
    // Bank system
    BANK_DAILY_SIZE: 100,
    BANK_LEARN_BATCH_SIZE: 25,
    BANK_LEARN_MIN_ACCEPT: 30,

    // Thought length rules (WORDS)
    FREE_THOUGHT_MIN_WORDS: 5,
    FREE_THOUGHT_MAX_WORDS: 15,

    PRO_THOUGHT_MIN_WORDS: 10,
    PRO_THOUGHT_MAX_WORDS: 35,

    // Auto-relax behaviour
    RELAX_1_DELTA_MIN_WORDS: 2,
    RELAX_1_DELTA_MAX_WORDS: 4,
    RELAX_2_DELTA_MIN_WORDS: 3,
    RELAX_2_DELTA_MAX_WORDS: 8,

    // Models
    CLASSIFY_MODEL: "gpt-4o-mini",
    THOUGHT_MODEL: "gpt-4o-mini",

    // Prewarm (hot labels)
    PREWARM_ENABLED: true,
    PREWARM_LABELS: ["man", "woman", "dog", "cat", "rabbit", "hamster", "fish", "bird", "horse"],
    PREWARM_ON_START: true,
    PREWARM_CHECK_INTERVAL_MINUTES: 60,

    // ✅ New users start with this many Smart Thoughts
    DEFAULT_PRO_BALANCE: Number(process.env.DEFAULT_PRO_BALANCE || 5),
};

const client = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

// ======================
// Supabase
// ======================
const SUPABASE_URL = process.env.SUPABASE_URL;
const SUPABASE_SERVICE_ROLE_KEY = process.env.SUPABASE_SERVICE_ROLE_KEY;

if (!SUPABASE_URL || !SUPABASE_SERVICE_ROLE_KEY) {
    console.warn("⚠️ Missing SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY. Endpoints will fail.");
}

const supabase = createClient(
    SUPABASE_URL || "http://invalid",
    SUPABASE_SERVICE_ROLE_KEY || "invalid",
    { auth: { persistSession: false } }
);

function requireDeviceId(req) {
    const id = req.body?.deviceId;
    if (typeof id !== "string") return null;
    const trimmed = id.trim();
    if (trimmed.length < 3) return null;
    return trimmed;
}

// ======================
// Label helpers
// ======================
const utcDayKey = () => new Date().toISOString().slice(0, 10);
const pick = (arr) => arr[Math.floor(Math.random() * arr.length)];
const cleanLabel = (l) => String(l || "").trim().toLowerCase();
const isValidLabel = (l) => /^[a-z]{2,24}$/.test(l);

const LABEL_ALIASES = {
    budgie: "parrot",
    parakeet: "parrot",
    canary: "bird",
    kitten: "cat",
    puppy: "dog",
    guineapig: "guinea",
    guinea_pig: "guinea",
};

function normalizeLabel(label) {
    const l = cleanLabel(label);
    return LABEL_ALIASES[l] || l;
}

function stripLinePrefix(t) {
    return String(t || "").replace(/^[-•\d.)\s]+/, "").trim();
}

function wordCount(str) {
    const s = String(str || "").trim();
    if (!s) return 0;
    return s.split(/\s+/).filter(Boolean).length;
}

function normalizeThoughtLines(text, { minWords, maxWords }) {
    return String(text)
        .split("\n")
        .map((t) => stripLinePrefix(t))
        .filter(Boolean)
        .filter((t) => {
            const wc = wordCount(t);
            return wc >= minWords && wc <= maxWords;
        });
}

function ensureSingleEndingEmoji(text) {
    const t = String(text || "").trim();
    if (!t) return t;
    const emojiAtEnd = t.match(/([\p{Emoji_Presentation}\p{Extended_Pictographic}])$/u);
    if (emojiAtEnd) return t;
    return `${t} 🙂`;
}

function getRelaxedRanges(minWords, maxWords) {
    const r1 = {
        minWords: Math.max(1, minWords - CONFIG.RELAX_1_DELTA_MIN_WORDS),
        maxWords: Math.max(minWords, maxWords + CONFIG.RELAX_1_DELTA_MAX_WORDS),
    };
    const r2 = {
        minWords: Math.max(1, minWords - CONFIG.RELAX_2_DELTA_MIN_WORDS),
        maxWords: Math.max(minWords, maxWords + CONFIG.RELAX_2_DELTA_MAX_WORDS),
    };
    return { r1, r2 };
}

function filterWithAutoRelax(text, { minWords, maxWords, labelForLogs = "unknown" }) {
    let lines = normalizeThoughtLines(text, { minWords, maxWords });
    if (lines.length >= 8) return { lines, mode: "strict" };

    const { r1, r2 } = getRelaxedRanges(minWords, maxWords);

    const relaxed1 = normalizeThoughtLines(text, r1);
    if (relaxed1.length >= 8) {
        console.warn(
            `⚠️ Auto-relax (level 1) used for '${labelForLogs}' batch: ${minWords}-${maxWords} → ${r1.minWords}-${r1.maxWords}`
        );
        return { lines: relaxed1, mode: "relax1" };
    }

    const relaxed2 = normalizeThoughtLines(text, r2);
    if (relaxed2.length) {
        console.warn(
            `⚠️ Auto-relax (level 2) used for '${labelForLogs}' batch: ${minWords}-${maxWords} → ${r2.minWords}-${r2.maxWords}`
        );
        return { lines: relaxed2, mode: "relax2" };
    }

    return { lines, mode: "strict-empty" };
}

// ======================
// Generation (FREE bank)
// ======================
async function generateThoughtBatch(label, count) {
    const minW = CONFIG.FREE_THOUGHT_MIN_WORDS;
    const maxW = CONFIG.FREE_THOUGHT_MAX_WORDS;

    const resp = await client.responses.create({
        model: CONFIG.THOUGHT_MODEL,
        input: [
            {
                role: "system",
                content:
                    "You write funny, family-friendly inner thoughts for a camera app. " +
                    "IMPORTANT: Thoughts MUST be in first-person as the subject in the image (the character). " +
                    "Use 'I/me/my' language. Never write as a narrator or a human observing a photo. " +
                    "Do NOT mention 'photo', 'picture', 'camera', 'app', 'user', 'viewer'. " +
                    "Use UK humour and UK wording (mates, cheeky, faff, sorted, etc) but keep it widely understandable. " +
                    "No hate, no sexual content, no slurs, no profanity. " +
                    "One thought per line. Avoid numbering/bullets. " +
                    "IMPORTANT: End the thought with exactly ONE fitting emoji at the very end. Do not add emojis elsewhere.",
            },
            {
                role: "user",
                content:
                    `Character: a ${label}\n` +
                    `Write ${count} funny inner thoughts as that character.\n` +
                    `Length: ${minW}-${maxW} words each.\n` +
                    `One per line.`,
            },
        ],
        max_output_tokens: 700,
    });

    const text =
        resp.output_text ||
        resp.output?.[0]?.content?.find((c) => c.type === "output_text")?.text ||
        "";

    const { lines } = filterWithAutoRelax(text, {
        minWords: minW,
        maxWords: maxW,
        labelForLogs: label,
    });

    return lines.map(ensureSingleEndingEmoji);
}

// ======================
// ✅ Supabase THOUGHT BANK helpers
// ======================

async function sbGetTodaysBank(label) {
    const { data, error } = await supabase.rpc("get_thought_bank_today", {
        p_label: label,
    });

    if (error) throw error;

    const row = Array.isArray(data) ? data[0] : data;
    const thoughts = row?.thoughts;

    if (!thoughts) return null;

    // thoughts is jsonb array → turn into string[]
    if (Array.isArray(thoughts)) return thoughts;
    if (typeof thoughts === "string") {
        // just in case
        try {
            const parsed = JSON.parse(thoughts);
            return Array.isArray(parsed) ? parsed : null;
        } catch {
            return null;
        }
    }

    return null;
}

async function sbUpsertTodaysBank(label, thoughtsArray) {
    const { data, error } = await supabase.rpc("upsert_thought_bank_today", {
        p_label: label,
        p_thoughts: thoughtsArray, // supabase-js will send as json
    });

    if (error) throw error;

    const row = Array.isArray(data) ? data[0] : data;
    const thoughts = row?.thoughts;
    return Array.isArray(thoughts) ? thoughts : thoughtsArray;
}

// Prevent multiple concurrent bank builds per label (in-process lock)
const bankBuildLocks = new Set();

async function buildBankInBackground(label) {
    if (!label || !isValidLabel(label)) return;

    const today = utcDayKey();
    const lockKey = `${label}:${today}`;

    if (bankBuildLocks.has(lockKey)) return;
    bankBuildLocks.add(lockKey);

    try {
        // If already exists in DB, bail
        const existing = await sbGetTodaysBank(label);
        if (existing?.length) return;

        console.log(`📚 Building daily bank for '${label}'...`);

        let thoughts = [];
        while (thoughts.length < CONFIG.BANK_DAILY_SIZE) {
            const batch = await generateThoughtBatch(label, CONFIG.BANK_LEARN_BATCH_SIZE);
            thoughts.push(...batch);
            thoughts = [...new Set(thoughts)];
            if (batch.length < 8) break;
        }

        thoughts = thoughts.slice(0, CONFIG.BANK_DAILY_SIZE);

        if (thoughts.length < CONFIG.BANK_LEARN_MIN_ACCEPT) {
            console.error(`❌ Bank generation too small for '${label}':`, thoughts.length);
            return;
        }

        await sbUpsertTodaysBank(label, thoughts);
        console.log(`✅ Bank ready for '${label}' (${thoughts.length})`);
    } catch (e) {
        console.error("Bank build error:", e);
    } finally {
        bankBuildLocks.delete(lockKey);
    }
}

async function ensureDailyBank(label) {
    const existing = await sbGetTodaysBank(label);
    if (existing?.length) return existing;

    await buildBankInBackground(label);
    return (await sbGetTodaysBank(label)) || [];
}

async function prewarmHotLabels() {
    if (!CONFIG.PREWARM_ENABLED) return;

    const labels = (CONFIG.PREWARM_LABELS || [])
        .map(normalizeLabel)
        .map(cleanLabel)
        .filter(isValidLabel);

    if (!labels.length) return;

    console.log("🔥 Prewarming labels:", labels.join(", "));
    for (const label of labels) {
        // fire-and-forget
        buildBankInBackground(label);
    }
}

// ======================
// 🧠 ENRICHMENT (PRO ONLY)
// ======================
async function enrichImage(imageDataUrl) {
    const r = await client.responses.create({
        model: CONFIG.CLASSIFY_MODEL,
        input: [
            { role: "system", content: "Return short visual tags only. No sentences." },
            {
                role: "user",
                content: [
                    { type: "input_text", text: "Return 5-10 comma-separated tags (e.g. sunny, indoors, laptop, sofa)." },
                    { type: "input_image", image_url: imageDataUrl, detail: "low" },
                ],
            },
        ],
        max_output_tokens: 120,
    });

    const text = (r.output_text || "").trim();
    return text
        .split(",")
        .map((t) => t.trim().toLowerCase())
        .filter(Boolean)
        .slice(0, 10);
}

async function generateProThought(label, tags) {
    const minW = CONFIG.PRO_THOUGHT_MIN_WORDS;
    const maxW = CONFIG.PRO_THOUGHT_MAX_WORDS;

    const r = await client.responses.create({
        model: CONFIG.THOUGHT_MODEL,
        input: [
            {
                role: "system",
                content:
                    "Write ONE funny, personality-rich inner thought for a family app. " +
                    "IMPORTANT: It MUST be in first-person as the subject in the image (the character). " +
                    "Use 'I/me/my' language. Never write as an observer or narrator. " +
                    "Do NOT mention 'photo', 'picture', 'camera', 'app', 'user', 'viewer'. " +
                    "Use UK humour/wording. No profanity, hate, sexual content. " +
                    `Length: ${minW}-${maxW} words. ` +
                    "IMPORTANT: End the thought with exactly ONE fitting emoji at the very end. Do not add emojis elsewhere.",
            },
            {
                role: "user",
                content:
                    `You are a ${label}.\n` +
                    `Scene tags (your surroundings): ${tags.join(", ")}\n` +
                    `Write the thought as your private inner monologue.`,
            },
        ],
        max_output_tokens: 160,
    });

    const out = stripLinePrefix((r.output_text || "").trim());
    return ensureSingleEndingEmoji(out);
}

// ======================
// Supabase usage helpers (YOUR EXISTING)
// ======================
async function sbGetStatus(deviceId) {
    const { data: existing, error: selErr } = await supabase
        .from("device_usage")
        .select("device_id, pro_tokens, pro_used")
        .eq("device_id", deviceId)
        .maybeSingle();

    if (selErr) throw selErr;

    if (!existing) {
        const { data: ins, error: insErr } = await supabase
            .from("device_usage")
            .insert({
                device_id: deviceId,
                pro_tokens: CONFIG.DEFAULT_PRO_BALANCE,
                pro_used: 0,
            })
            .select("device_id, pro_tokens, pro_used")
            .single();

        if (insErr) throw insErr;

        return {
            proTokens: ins.pro_tokens,
            proUsed: ins.pro_used,
            remainingPro: Math.max(0, ins.pro_tokens - ins.pro_used),
        };
    }

    return {
        proTokens: existing.pro_tokens,
        proUsed: existing.pro_used,
        remainingPro: Math.max(0, existing.pro_tokens - existing.pro_used),
    };
}

async function sbGrantCredits(deviceId, amount) {
    const { data, error } = await supabase.rpc("grant_pro_credits", {
        p_device_id: deviceId,
        p_amount: amount,
        p_default_seed: CONFIG.DEFAULT_PRO_BALANCE,
    });

    if (error) throw error;
    const row = Array.isArray(data) ? data[0] : data;
    return {
        proTokens: row.pro_tokens,
        proUsed: row.pro_used,
        remainingPro: row.remaining_pro,
    };
}

async function sbSpendCredits(deviceId, cost) {
    const { data, error } = await supabase.rpc("spend_pro_credits", {
        p_device_id: deviceId,
        p_cost: cost,
        p_default_seed: CONFIG.DEFAULT_PRO_BALANCE,
    });

    if (error) throw error;
    const row = Array.isArray(data) ? data[0] : data;
    return {
        ok: !!row.ok,
        proTokens: row.pro_tokens,
        proUsed: row.pro_used,
        remainingPro: row.remaining_pro,
    };
}

// ======================
// Routes
// ======================
app.get("/health", (req, res) => res.json({ ok: true }));

app.post("/status", async (req, res) => {
    try {
        const deviceId = requireDeviceId(req);
        if (!deviceId) return res.status(400).json({ ok: false, error: "MISSING_DEVICE_ID" });

        const s = await sbGetStatus(deviceId);
        return res.json({
            ok: true,
            remainingPro: s.remainingPro,
            proTokens: s.proTokens,
            proUsed: s.proUsed,
            source: "supabase",
        });
    } catch (e) {
        console.error("Server error in /status:", e);
        res.status(500).json({ ok: false, error: "SERVER_ERROR" });
    }
});

app.post("/classify", async (req, res) => {
    try {
        const { imageDataUrl } = req.body;
        if (!imageDataUrl || typeof imageDataUrl !== "string") return res.json({ ok: false });

        const r = await client.responses.create({
            model: CONFIG.CLASSIFY_MODEL,
            input: [
                { role: "system", content: "Classify the main subject in the image." },
                {
                    role: "user",
                    content: [
                        {
                            type: "input_text",
                            text:
                                "Return JSON only with:\n" +
                                "- ok: boolean\n" +
                                "- category: 'human' or 'animal'\n" +
                                "- label:\n" +
                                "   • if human -> 'man' or 'woman'\n" +
                                "   • if animal -> a simple lowercase species word (e.g. dog, cat, rabbit, horse, bird)\n" +
                                "Rules:\n" +
                                "• If unsure, ok=false.\n" +
                                "• label must be lowercase letters only.\n" +
                                "• Avoid generic labels like 'animal' or 'pet'.",
                        },
                        { type: "input_image", image_url: imageDataUrl, detail: "low" },
                    ],
                },
            ],
            text: {
                format: {
                    type: "json_schema",
                    strict: true,
                    name: "classify",
                    schema: {
                        type: "object",
                        additionalProperties: false,
                        properties: {
                            ok: { type: "boolean" },
                            category: { type: "string", enum: ["human", "animal"] },
                            label: { type: "string", pattern: "^[a-z]+$", minLength: 2, maxLength: 24 },
                        },
                        required: ["ok", "category", "label"],
                    },
                },
            },
        });

        const parsed = JSON.parse(r.output_text || "{}");
        if (!parsed?.ok || !parsed?.label) return res.json({ ok: false, reason: "not_detected" });

        parsed.label = normalizeLabel(parsed.label);

        const blocked = new Set(["animal", "pet", "mammal", "person", "human"]);
        if (!isValidLabel(parsed.label) || blocked.has(parsed.label)) {
            return res.json({ ok: false, reason: "invalid_label" });
        }

        return res.json(parsed);
    } catch (e) {
        console.error("Server error in /classify:", e);
        res.status(500).json({ ok: false });
    }
});

app.post("/thought-free", async (req, res) => {
    try {
        const label = normalizeLabel(req.body.label);
        if (!isValidLabel(label)) return res.json({ ok: false });

        // ✅ Supabase daily bank now
        let thoughts = await sbGetTodaysBank(label);
        if (!thoughts) {
            buildBankInBackground(label); // start now
            thoughts = await ensureDailyBank(label); // try to satisfy request
        }

        if (!thoughts?.length) return res.json({ ok: false, error: "NO_BANK_YET" });

        return res.json({
            ok: true,
            thought: pick(thoughts),
            tier: "free",
            source: "supabase-daily-bank",
        });
    } catch (e) {
        console.error("Server error in /thought-free:", e);
        res.status(500).json({ ok: false });
    }
});

app.post("/thought-pro", async (req, res) => {
    try {
        const deviceId = requireDeviceId(req);
        if (!deviceId) return res.status(400).json({ ok: false, error: "MISSING_DEVICE_ID" });

        const { label, imageDataUrl } = req.body;
        const clean = normalizeLabel(label);

        if (!isValidLabel(clean) || !imageDataUrl) {
            return res.json({ ok: false, error: "BAD_REQUEST" });
        }

        // ✅ Atomic spend first
        const spend = await sbSpendCredits(deviceId, 1);
        if (!spend.ok) {
            return res.json({ ok: false, error: "PRO_LIMIT_REACHED", remainingPro: spend.remainingPro ?? 0 });
        }

        const tags = await enrichImage(imageDataUrl);
        const thought = await generateProThought(clean, tags);

        if (!thought || typeof thought !== "string") {
            // Refund if generation fails (nice touch)
            await sbGrantCredits(deviceId, 1).catch(() => {});
            return res.json({ ok: false, error: "PRO_FAILED" });
        }

        return res.json({
            ok: true,
            thought,
            tags,
            tier: "pro",
            source: "supabase",
            remainingPro: spend.remainingPro,
            proTokens: spend.proTokens,
            proUsed: spend.proUsed,
        });
    } catch (e) {
        console.error("Server error in /thought-pro:", e);
        res.status(500).json({ ok: false, error: "SERVER_ERROR" });
    }
});

// DEV purchase simulator
app.post("/dev/add-credits", async (req, res) => {
    try {
        const { deviceId, amount } = req.body || {};
        const id = typeof deviceId === "string" ? deviceId.trim() : null;
        const n = Math.max(0, Number(amount || 0));

        if (!id) return res.status(400).json({ ok: false, error: "missing_deviceId" });
        if (!n) return res.status(400).json({ ok: false, error: "missing_amount" });

        const r = await sbGrantCredits(id, n);

        return res.json({
            ok: true,
            remainingPro: r.remainingPro,
            proTokens: r.proTokens,
            proUsed: r.proUsed,
            source: "dev+supabase",
        });
    } catch (e) {
        console.error("dev add credits error", e);
        return res.status(500).json({ ok: false, error: "server_error" });
    }
});

const PORT = process.env.PORT || 8787;

app.listen(PORT, () => {
    console.log(`Server running on port ${PORT}`);

    if (CONFIG.PREWARM_ON_START) prewarmHotLabels();

    const mins = Number(CONFIG.PREWARM_CHECK_INTERVAL_MINUTES || 0);
    if (mins > 0) setInterval(() => prewarmHotLabels(), mins * 60 * 1000);
});
