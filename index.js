// server/index.js
import "dotenv/config";
import express from "express";
import cors from "cors";
import OpenAI from "openai";
import { createClient } from "@supabase/supabase-js";
import crypto from "crypto";

const app = express();
app.use(cors());
app.use(express.json({ limit: "20mb" }));

// ======================
// 🔧 CONFIG
// ======================
const CONFIG = {
    BANK_DAILY_SIZE: 100,
    BANK_LEARN_BATCH_SIZE: 25,
    BANK_LEARN_MIN_ACCEPT: 30,

    FREE_THOUGHT_MIN_WORDS: 5,
    FREE_THOUGHT_MAX_WORDS: 15,

    PRO_THOUGHT_MIN_WORDS: 10,
    PRO_THOUGHT_MAX_WORDS: 35,

    RELAX_1_DELTA_MIN_WORDS: 2,
    RELAX_1_DELTA_MAX_WORDS: 4,
    RELAX_2_DELTA_MIN_WORDS: 3,
    RELAX_2_DELTA_MAX_WORDS: 8,

    CLASSIFY_MODEL: "gpt-4o-mini",
    THOUGHT_MODEL: "gpt-4o-mini",

    PREWARM_ENABLED: true,
    PREWARM_LABELS: ["man", "woman", "dog", "cat", "rabbit", "hamster", "fish", "bird", "horse"],
    PREWARM_ON_START: true,
    PREWARM_CHECK_INTERVAL_MINUTES: 60,

    DEFAULT_PRO_BALANCE: Number(process.env.DEFAULT_PRO_BALANCE || 5),

    // ✅ Subject-only cache (free)
    SUBJECT_CACHE_TTL_MS: 10 * 60 * 1000, // 10 minutes
    SUBJECT_CACHE_MAX: 2000,
};

const client = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

// ======================
// Supabase
// ======================
const SUPABASE_URL = process.env.SUPABASE_URL;
const SUPABASE_SERVICE_ROLE_KEY = process.env.SUPABASE_SERVICE_ROLE_KEY;

if (!SUPABASE_URL || !SUPABASE_SERVICE_ROLE_KEY) {
    console.warn("⚠️ Missing SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY. Supabase endpoints will fail.");
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

function jwtRole(key) {
    try {
        const payload = key.split(".")[1];
        const json = Buffer.from(payload, "base64").toString("utf8");
        return JSON.parse(json).role || JSON.parse(json).aud;
    } catch {
        return "unreadable";
    }
}

console.log("SUPABASE key role:", jwtRole(process.env.SUPABASE_SERVICE_ROLE_KEY || ""));

// ======================
// Helpers
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
// Subject-only cache (FREE)
// ======================
const subjectCache = new Map(); // key -> { label, subject, expiresAt }

function subjectCacheKey(imageDataUrl) {
    // sha1 of the full string is fine at ~150KB
    return crypto.createHash("sha1").update(String(imageDataUrl || "")).digest("hex");
}

function subjectCacheGet(key) {
    const v = subjectCache.get(key);
    if (!v) return null;
    if (Date.now() > v.expiresAt) {
        subjectCache.delete(key);
        return null;
    }
    return v;
}

function subjectCacheSet(key, value) {
    if (subjectCache.size >= CONFIG.SUBJECT_CACHE_MAX) {
        // delete one (oldest-ish: Map preserves insertion order)
        const firstKey = subjectCache.keys().next().value;
        if (firstKey) subjectCache.delete(firstKey);
    }
    subjectCache.set(key, value);
}

// ======================
// Supabase bank helpers
// Table: thought_banks(label text, bank_date date, thoughts text[])
// ======================

// Returns bank for specific date key
async function sbGetBankByDate(label, bankDate) {
    const { data, error } = await supabase
        .from("thought_banks")
        .select("thoughts, bank_date")
        .eq("label", label)
        .eq("bank_date", bankDate)
        .maybeSingle();

    if (error) throw error;

    const thoughts = data?.thoughts;
    if (!Array.isArray(thoughts) || thoughts.length === 0) return null;

    return thoughts
        .filter((t) => typeof t === "string")
        .map((t) => t.trim())
        .filter(Boolean);
}

// ✅ wrapper so existing code still works
async function sbGetTodaysBank(label) {
    return sbGetBankByDate(label, utcDayKey());
}

// ✅ Returns most recent bank (any date)
async function sbGetLatestBank(label) {
    const { data, error } = await supabase
        .from("thought_banks")
        .select("thoughts, bank_date")
        .eq("label", label)
        .order("bank_date", { ascending: false })
        .limit(1)
        .maybeSingle();

    if (error) throw error;

    const thoughts = data?.thoughts;
    if (!Array.isArray(thoughts) || thoughts.length === 0) return null;

    return thoughts
        .filter((t) => typeof t === "string")
        .map((t) => t.trim())
        .filter(Boolean);
}

async function sbUpsertBank(label, thoughts) {
    const bankDate = utcDayKey();

    const cleanThoughts = (thoughts || [])
        .filter((t) => typeof t === "string")
        .map((t) => t.trim())
        .filter(Boolean);

    const { error } = await supabase
        .from("thought_banks")
        .upsert({ label, bank_date: bankDate, thoughts: cleanThoughts }, { onConflict: "label,bank_date" });

    if (error) throw error;
}

// ======================
// Generation (Free bank)
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

    const { lines } = filterWithAutoRelax(text, { minWords: minW, maxWords: maxW, labelForLogs: label });
    return lines.map(ensureSingleEndingEmoji);
}

// ======================
// Daily bank builder
// ======================
const bankBuildLocks = new Set();

async function buildBankInBackground(label) {
    if (!label || !isValidLabel(label)) return;
    if (bankBuildLocks.has(label)) return;

    bankBuildLocks.add(label);
    try {
        const existing = await sbGetTodaysBank(label).catch(() => null);
        if (existing?.length) return;

        console.log(`📚 Building daily bank for '${label}' (Supabase)...`);

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

        await sbUpsertBank(label, thoughts);
        console.log(`✅ Bank saved for '${label}' (${thoughts.length})`);
    } catch (e) {
        console.error("Bank build error:", e);
    } finally {
        bankBuildLocks.delete(label);
    }
}

async function ensureDailyBank(label) {
    const today = utcDayKey();

    // 1) Try today's bank
    const todays = await sbGetBankByDate(label, today).catch(() => null);
    if (todays?.length) return { thoughts: todays, source: "today" };

    // 2) Fallback to latest available (yesterday etc.)
    const latest = await sbGetLatestBank(label).catch(() => null);
    if (latest?.length) {
        // Build today's in background but don't block user
        buildBankInBackground(label);
        return { thoughts: latest, source: "latest-fallback" };
    }

    // 3) Nothing exists anywhere -> kick off build and return empty
    buildBankInBackground(label);
    return { thoughts: [], source: "none" };
}

async function prewarmHotLabels() {
    if (!CONFIG.PREWARM_ENABLED) return;

    const labels = (CONFIG.PREWARM_LABELS || [])
        .map(normalizeLabel)
        .map(cleanLabel)
        .filter(isValidLabel);

    if (!labels.length) return;

    for (const label of labels) {
        const existing = await sbGetTodaysBank(label).catch(() => null);
        if (!existing?.length) buildBankInBackground(label);
    }
}

// ======================
// 🧠 ENRICHMENT (used for pro)
// ======================
async function enrichImage(imageDataUrl) {
    const r = await client.responses.create({
        model: CONFIG.CLASSIFY_MODEL,
        input: [
            {
                role: "system",
                content:
                    "You are a visual analyst for a family-friendly humour app. " +
                    "Return JSON only. Be concise. If unsure, pick the best guess.",
            },
            {
                role: "user",
                content: [
                    {
                        type: "input_text",
                        text:
                            "Analyse the subject and their behaviour. Return JSON with:\n" +
                            "{\n" +
                            '  "subject": "dog|cat|man|woman|other",\n' +
                            '  "action": "sleeping|yawning|chewing|staring|playing|posing|walking|eating|begging|other",\n' +
                            '  "expression": "happy|sleepy|suspicious|annoyed|excited|guilty|confused|neutral|other",\n' +
                            '  "gaze": "at_camera|away|side_eye|up|down|unknown",\n' +
                            '  "pose": "lying|sitting|standing|curled|sprawled|unknown",\n' +
                            '  "setting": ["indoors|outdoors", "sofa|bed|car|garden|office|street|other"],\n' +
                            '  "props": ["toy","food","leash","phone","laptop","bowl","blanket","shoe","none"],\n' +
                            '  "extra_tags": ["short", "descriptive", "words"],\n' +
                            '  "vibe": "2-5 words"\n' +
                            "}\n" +
                            "Rules: keep arrays short (max 6). No sentences.",
                    },
                    { type: "input_image", image_url: imageDataUrl, detail: "high" },
                ],
            },
        ],
        text: {
            format: {
                type: "json_schema",
                strict: true,
                name: "enrichment",
                schema: {
                    type: "object",
                    additionalProperties: false,
                    properties: {
                        subject: { type: "string" },
                        action: { type: "string" },
                        expression: { type: "string" },
                        gaze: { type: "string" },
                        pose: { type: "string" },
                        setting: { type: "array", items: { type: "string" }, maxItems: 2 },
                        props: { type: "array", items: { type: "string" }, maxItems: 6 },
                        extra_tags: { type: "array", items: { type: "string" }, maxItems: 6 },
                        vibe: { type: "string" },
                    },
                    required: ["subject", "action", "expression", "gaze", "pose", "setting", "props", "extra_tags", "vibe"],
                },
            },
        },
        max_output_tokens: 220,
    });

    return JSON.parse(r.output_text || "{}");
}

function classifyFromEnrich(enrich) {
    if (!enrich || typeof enrich.subject !== "string") {
        return { ok: false, reason: "no_subject" };
    }

    const label = normalizeLabel(enrich.subject);

    if (!label || label === "other") return { ok: false, reason: "other" };

    const category = label === "man" || label === "woman" ? "human" : "animal";

    return { ok: true, category, label };
}

// ======================
// Subject-only classify (used for free) + cache
// ======================
async function classifySubjectOnly(imageDataUrl, timings) {
    const key = subjectCacheKey(imageDataUrl);
    const cached = subjectCacheGet(key);
    if (cached) {
        timings.subject_only_cache_hit = true;
        return { subject: cached.subject, label: cached.label, cached: true };
    }

    const r = await client.responses.create({
        model: CONFIG.CLASSIFY_MODEL,
        input: [
            {
                role: "system",
                content:
                    "Return JSON only. Identify the single main subject in the image as either man, woman, or a pet; if it matches one of the following pets return that exact value: dog, cat, horse, bird, rabbit, hamster, fish, guinea pig, turtle, tortoise, parrot, ferret, hedgehog, chinchilla, gecko, snake, lizard, pig, goat, sheep, cow, chicken, duck, goose, donkey, pony, alpaca, llama, deer, fox, wolf, raccoon, squirrel, rat, mouse, gerbil, frog, toad, axolotl, crab, shrimp, goldfish, budgie, canary, cockatiel, pigeon, swan, peacock; if it is clearly a pet but not in this list return the most specific single-word animal name; if none clearly match return other; use lowercase only and return exactly one value with no additional text.",
            },
            {
                role: "user",
                content: [
                    { type: "input_text", text: 'Return JSON: {"subject":"..."}' },
                    { type: "input_image", image_url: imageDataUrl, detail: "low" },
                ],
            },
        ],
        text: {
            format: {
                type: "json_schema",
                strict: true,
                name: "subject_only",
                schema: {
                    type: "object",
                    additionalProperties: false,
                    properties: { subject: { type: "string" } },
                    required: ["subject"],
                },
            },
        },
        max_output_tokens: 60,
    });

    const out = JSON.parse(r.output_text || "{}");
    const label = normalizeLabel(out.subject);

    subjectCacheSet(key, {
        subject: out.subject,
        label,
        expiresAt: Date.now() + CONFIG.SUBJECT_CACHE_TTL_MS,
    });

    return { subject: out.subject, label, cached: false };
}

// ======================
// Pro thought generator
// ======================
async function generateProThought(label, enrich) {
    const minW = CONFIG.PRO_THOUGHT_MIN_WORDS;
    const maxW = CONFIG.PRO_THOUGHT_MAX_WORDS;

    const r = await client.responses.create({
        model: CONFIG.THOUGHT_MODEL,
        input: [
            {
                role: "system",
                content:
                    "Write ONE funny, personality-rich inner thought for a family app. " +
                    "Must be first-person as the subject in the image. Use I/me/my. " +
                    "No mention of photo/camera/app/user/viewer. " +
                    "Use UK humour/wording. No profanity/hate/sexual content. " +
                    `Length: ${minW}-${maxW} words. ` +
                    "End with exactly ONE fitting emoji at the very end.",
            },
            {
                role: "user",
                content:
                    `You are a ${label}.\n` +
                    `Behaviour:\n` +
                    `- action: ${enrich.action}\n` +
                    `- expression: ${enrich.expression}\n` +
                    `- gaze: ${enrich.gaze}\n` +
                    `- pose: ${enrich.pose}\n` +
                    `Scene:\n` +
                    `- setting: ${(enrich.setting || []).join(", ")}\n` +
                    `- props: ${(enrich.props || []).join(", ")}\n` +
                    `- extra: ${(enrich.extra_tags || []).join(", ")}\n` +
                    `- vibe: ${enrich.vibe}\n` +
                    `Write an inner thought that clearly reflects the action + expression + gaze (not just the room).`,
            },
        ],
        max_output_tokens: 160,
    });

    const out = stripLinePrefix((r.output_text || "").trim());
    return ensureSingleEndingEmoji(out);
}

// ======================
// Supabase usage helpers
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

// ✅ Unified thought endpoint
// POST /thought
// body: { tier: "free"|"pro", imageDataUrl: "data:...base64", deviceId?: string }
app.post("/thought", async (req, res) => {
    const t0 = Date.now();
    const rid = `srv_${crypto.randomBytes(6).toString("hex")}`;

    const timings = { start: 0 };

    try {
        const { tier, imageDataUrl } = req.body || {};
        const isPro = tier === "pro";

        const contentLength = String(imageDataUrl || "").length;

        console.log("[THOUGHT] start", { rid, tier, contentLength });

        if (!imageDataUrl || typeof imageDataUrl !== "string") {
            return res.status(400).json({ ok: false, error: "BAD_REQUEST" });
        }

        let label = "other";
        let enrich = null;

        if (isPro) {
            const tE = Date.now();
            enrich = await enrichImage(imageDataUrl);
            timings.enrich_done = Date.now() - t0;

            const out = classifyFromEnrich(enrich);
            label = out?.ok ? out.label : "other";

            console.log("[THOUGHT] enrich", { rid, ms: Date.now() - tE, label });
        } else {
            const tS = Date.now();
            const subj = await classifySubjectOnly(imageDataUrl, timings);
            timings.subject_only_done = Date.now() - t0;
            label = subj?.label || "other";

            console.log("[THOUGHT] subject_only", {
                rid,
                ms: Date.now() - tS,
                label,
                cached: !!subj?.cached,
                cacheHit: !!timings.subject_only_cache_hit,
            });
        }

        timings.label_set = Date.now() - t0;

        const blocked = new Set(["animal", "pet", "mammal", "person", "human"]);
        if (!isValidLabel(label) || blocked.has(label) || label === "other") {
            timings.done_free = Date.now() - t0;
            console.log("[THOUGHT] unknown-label", { rid, label, totalMs: Date.now() - t0 });

            return res.json({
                ok: true,
                thought: "I can’t tell what I’m looking at… but I’m judging it anyway. 👀",
                tier: isPro ? "pro" : "free",
                label: "unknown",
                ms: Date.now() - t0,
                timings,
                enrich: isPro ? enrich : undefined,
            });
        }

        // FREE path: bank pick (single return, logs always fire)
        if (!isPro) {
            const tB = Date.now();

            const bank = await ensureDailyBank(label);
            const thoughts = bank.thoughts;

            timings.bank_fetch_done = Date.now() - t0;

            if (!thoughts?.length) {
                timings.done_free = Date.now() - t0;

                console.log("[THOUGHT] no_bank_anywhere", {
                    rid,
                    label,
                    bankSource: bank.source,
                    bankMs: Date.now() - tB,
                    totalMs: Date.now() - t0,
                });

                return res.json({
                    ok: true,
                    thought: "I can’t tell what I’m looking at… but I’m judging it anyway. 👀",
                    tier: "free",
                    label: "unknown",
                    ms: Date.now() - t0,
                    timings,
                    source: "fallback-no-bank",
                });
            }

            timings.done_free = Date.now() - t0;

            console.log("[THOUGHT] free_done", {
                rid,
                label,
                bankSource: bank.source, // today | latest-fallback | none
                bankMs: Date.now() - tB,
                totalMs: Date.now() - t0,
                bankCount: thoughts.length,
            });

            return res.json({
                ok: true,
                thought: pick(thoughts),
                tier: "free",
                label,
                ms: Date.now() - t0,
                timings,
                source: `supabase-bank:${bank.source}`,
            });
        }

        // PRO path: spend credits then generate
        const deviceId = requireDeviceId(req);
        if (!deviceId) return res.status(400).json({ ok: false, error: "MISSING_DEVICE_ID" });

        const tSpend = Date.now();
        const spend = await sbSpendCredits(deviceId, 1);
        timings.credits_spend_done = Date.now() - t0;

        console.log("[THOUGHT] spend", {
            rid,
            ok: spend.ok,
            ms: Date.now() - tSpend,
            remainingPro: spend.remainingPro,
        });

        if (!spend.ok) {
            return res.json({
                ok: false,
                error: "PRO_LIMIT_REACHED",
                remainingPro: spend.remainingPro ?? 0,
                ms: Date.now() - t0,
                timings,
            });
        }

        const tGen = Date.now();
        const thought = await generateProThought(label, enrich);
        timings.pro_generate_done = Date.now() - t0;

        console.log("[THOUGHT] pro_done", { rid, label, genMs: Date.now() - tGen, totalMs: Date.now() - t0 });

        return res.json({
            ok: true,
            thought,
            tier: "pro",
            label,
            enrich,
            remainingPro: spend.remainingPro,
            ms: Date.now() - t0,
            timings,
        });
    } catch (e) {
        console.error("Server error in /thought:", e);
        return res.status(500).json({ ok: false, error: "SERVER_ERROR", ms: Date.now() - t0, timings });
    }
});

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

// POST /classify
// body: { imageDataUrl: "data:...base64" }
// returns: { ok: true, label: "dog", subject: "dog", cached: true/false }
app.post("/classify", async (req, res) => {
    const t0 = Date.now();
    const timings = {};

    try {
        const { imageDataUrl } = req.body || {};
        const contentLength = String(imageDataUrl || "").length;

        if (!imageDataUrl || typeof imageDataUrl !== "string") {
            return res.status(400).json({ ok: false, error: "BAD_REQUEST" });
        }

        console.log("[CLASSIFY] start", { contentLength });

        // Use your existing fast subject-only classifier (+ cache)
        const subj = await classifySubjectOnly(imageDataUrl, timings);
        const label = subj?.label || "other";
        const subject = subj?.subject || "other";

        // Normalize “other” and blocked values to unknown for pet profiles
        const blocked = new Set(["animal", "pet", "mammal", "person", "human"]);
        const finalLabel =
            !isValidLabel(label) || blocked.has(label) || label === "other"
                ? "unknown"
                : label;

        return res.json({
            ok: true,
            label: finalLabel,
            subject,
            cached: !!subj?.cached,
            ms: Date.now() - t0,
            timings,
        });
    } catch (e) {
        console.error("Server error in /classify:", e);
        return res.status(500).json({ ok: false, error: "SERVER_ERROR", ms: Date.now() - t0 });
    }
});

const PORT = process.env.PORT || 8787;

app.listen(PORT, () => {
    console.log(`Server running on port ${PORT}`);

    if (CONFIG.PREWARM_ON_START) prewarmHotLabels();

    const mins = Number(CONFIG.PREWARM_CHECK_INTERVAL_MINUTES || 0);
    if (mins > 0) setInterval(() => prewarmHotLabels(), mins * 60 * 1000);
});
