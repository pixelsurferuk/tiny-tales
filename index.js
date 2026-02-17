// server/index.js
import "dotenv/config";
import express from "express";
import cors from "cors";
import OpenAI from "openai";
import { createClient } from "@supabase/supabase-js";
import crypto from "crypto";

const app = express();
app.use(cors());

// Keep JSON for normal endpoints.
// RevenueCat webhooks are JSON too; we’ll auth them via header token (simplest + robust).
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

    // Ask-a-question replies (short + punchy)
    ASK_MIN_WORDS: 10,
    ASK_MAX_WORDS: 35,

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

    // ✅ Ask chat memory
    ASK_HISTORY_MAX: 10,
    ASK_HISTORY_MAX_CHARS: 420,

    // ✅ Free chat tokens (kept!)
    FREE_CHAT_TOKENS: Number(process.env.FREE_CHAT_TOKENS || 10),

    // RevenueCat entitlement id (your dashboard: Entitlements -> Identifier)
    RC_ENTITLEMENT_ID: process.env.RC_ENTITLEMENT_ID || "pro_access",

    // RevenueCat webhook auth token (set this in RC dashboard webhook config + server env)
    RC_WEBHOOK_AUTH: process.env.RC_WEBHOOK_AUTH || "",

    // Cache RC subscriber lookup briefly to reduce API calls
    RC_CACHE_TTL_MS: 60 * 1000,
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

const supabase = createClient(SUPABASE_URL || "http://invalid", SUPABASE_SERVICE_ROLE_KEY || "invalid", {
    auth: { persistSession: false },
});

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
// RevenueCat server-side validation
// ======================
const REVENUECAT_SECRET_KEY = process.env.REVENUECAT_SECRET_KEY;
const rcCache = new Map(); // appUserID -> { isPro, expiresAtMs }
const RC_CACHE_TTL_MS = CONFIG.RC_CACHE_TTL_MS;

async function validateProWithRevenueCat(appUserID) {
    if (!REVENUECAT_SECRET_KEY) return false;

    const cached = rcCache.get(appUserID);
    if (cached && Date.now() < cached.expiresAtMs) {
        return cached.isPro;
    }

    try {
        const r = await fetch(`https://api.revenuecat.com/v1/subscribers/${encodeURIComponent(appUserID)}`, {
            headers: {
                Authorization: `Bearer ${REVENUECAT_SECRET_KEY}`,
                "Content-Type": "application/json",
            },
        });

        if (!r.ok) {
            // short negative cache to avoid hammering RC if key misconfigured etc.
            rcCache.set(appUserID, { isPro: false, expiresAtMs: Date.now() + 15_000 });
            return false;
        }

        const data = await r.json();
        const ent = data?.subscriber?.entitlements?.[CONFIG.RC_ENTITLEMENT_ID];

        let isPro = false;
        if (ent) {
            if (!ent.expires_date) {
                // non-expiring entitlement
                isPro = true;
            } else {
                const exp = Date.parse(ent.expires_date);
                isPro = Number.isFinite(exp) ? exp > Date.now() : false;
            }
        }

        rcCache.set(appUserID, { isPro, expiresAtMs: Date.now() + RC_CACHE_TTL_MS });
        return isPro;
    } catch (e) {
        // Fail closed for pro (safer for monetisation)
        rcCache.set(appUserID, { isPro: false, expiresAtMs: Date.now() + 15_000 });
        return false;
    }
}

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
        const firstKey = subjectCache.keys().next().value;
        if (firstKey) subjectCache.delete(firstKey);
    }
    subjectCache.set(key, value);
}

// ======================
// Supabase device usage row (keeps free chat + pro credits)
// ======================
async function sbEnsureDeviceRow(deviceId) {
    const { data, error } = await supabase.from("device_usage").select("device_id").eq("device_id", deviceId).maybeSingle();
    if (error) throw error;
    if (data) return true;

    const { error: insErr } = await supabase.from("device_usage").insert({
        device_id: deviceId,
        pro_tokens: CONFIG.DEFAULT_PRO_BALANCE,
        pro_used: 0,
        free_chat_tokens: CONFIG.FREE_CHAT_TOKENS,
        free_chat_used: 0,
    });

    if (insErr) throw insErr;
    return true;
}

// ======================
// Supabase bank + free-chat helpers
// ======================
async function sbGetChatStatus(deviceId) {
    await sbEnsureDeviceRow(deviceId);

    const { data, error } = await supabase
        .from("device_usage")
        .select("device_id, free_chat_tokens, free_chat_used")
        .eq("device_id", deviceId)
        .maybeSingle();

    if (error) throw error;

    return {
        freeChatTokens: data?.free_chat_tokens ?? CONFIG.FREE_CHAT_TOKENS,
        freeChatUsed: data?.free_chat_used ?? 0,
        remainingFreeChat: Math.max(0, (data?.free_chat_tokens ?? CONFIG.FREE_CHAT_TOKENS) - (data?.free_chat_used ?? 0)),
    };
}

async function sbConsumeFreeChat(deviceId) {
    await sbEnsureDeviceRow(deviceId);

    const { data, error } = await supabase.rpc("consume_free_chat", {
        p_device_id: deviceId,
        p_default_seed: CONFIG.FREE_CHAT_TOKENS,
    });

    if (error) throw error;

    const row = Array.isArray(data) ? data[0] : data;

    return {
        ok: !!row.ok,
        freeChatTokens: row.free_chat_tokens,
        freeChatUsed: row.free_chat_used,
        remainingFreeChat: row.remaining_free_chat,
    };
}

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

async function sbGetTodaysBank(label) {
    return sbGetBankByDate(label, utcDayKey());
}

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

    const text = resp.output_text || resp.output?.[0]?.content?.find((c) => c.type === "output_text")?.text || "";

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

    const todays = await sbGetBankByDate(label, today).catch(() => null);
    if (todays?.length) return { thoughts: todays, source: "today" };

    const latest = await sbGetLatestBank(label).catch(() => null);
    if (latest?.length) {
        buildBankInBackground(label);
        return { thoughts: latest, source: "latest-fallback" };
    }

    buildBankInBackground(label);
    return { thoughts: [], source: "none" };
}

async function prewarmHotLabels() {
    if (!CONFIG.PREWARM_ENABLED) return;

    const labels = (CONFIG.PREWARM_LABELS || []).map(normalizeLabel).map(cleanLabel).filter(isValidLabel);
    if (!labels.length) return;

    for (const label of labels) {
        const existing = await sbGetTodaysBank(label).catch(() => null);
        if (!existing?.length) buildBankInBackground(label);
    }
}

// ======================
// 🧠 ENRICHMENT (used for pro thoughts)
// ======================
async function enrichImage(imageDataUrl) {
    const r = await client.responses.create({
        model: CONFIG.CLASSIFY_MODEL,
        input: [
            {
                role: "system",
                content: "You are a visual analyst for a family-friendly humour app. Return JSON only. Be concise.",
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
    if (!enrich || typeof enrich.subject !== "string") return { ok: false, reason: "no_subject" };

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
                    "Return JSON only. Identify the single main subject in the image as either man, woman, or a pet; " +
                    "if it matches one of the following pets return that exact value: dog, cat, horse, bird, rabbit, hamster, fish, guinea pig, turtle, tortoise, parrot, ferret, hedgehog, chinchilla, gecko, snake, lizard, pig, goat, sheep, cow, chicken, duck, goose, donkey, pony, alpaca, llama, deer, fox, wolf, raccoon, squirrel, rat, mouse, gerbil, frog, toad, axolotl, crab, shrimp, goldfish, budgie, canary, cockatiel, pigeon, swan, peacock; " +
                    "if it is clearly a pet but not in this list return the most specific single-word animal name; " +
                    "if none clearly match return other; use lowercase only and return exactly one value with no additional text.",
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
// Ask-a-question generator (with chat history)
// ======================
function sanitizeAskHistory(history) {
    if (!Array.isArray(history)) return [];
    return history
        .slice(-CONFIG.ASK_HISTORY_MAX)
        .filter(
            (m) =>
                m &&
                (m.role === "user" || m.role === "assistant") &&
                typeof m.content === "string" &&
                m.content.trim().length > 0
        )
        .map((m) => ({
            role: m.role,
            content: String(m.content).trim().slice(0, CONFIG.ASK_HISTORY_MAX_CHARS),
        }));
}

async function generateAskAnswer({ label, pet, question, history = [] }) {
    const minW = CONFIG.ASK_MIN_WORDS;
    const maxW = CONFIG.ASK_MAX_WORDS;

    const petName = String(pet?.name || "my pet").trim() || "my pet";
    const vibe = String(pet?.vibe || "").trim();

    const safeHistory = sanitizeAskHistory(history);

    const r = await client.responses.create({
        model: CONFIG.THOUGHT_MODEL,
        input: [
            {
                role: "system",
                content:
                    `You are ${petName}, a ${label}.
Write exactly like a ${label} texting their owner.
You are not an assistant. You are not helpful. You are opinionated.

Reply in first person (I/me/my).
Short, chatty, playful, slightly dramatic. UK humour.
Natural texting tone, not polished writing.

Be expressive. React strongly. Have opinions.
Tease, exaggerate, sulk, brag, or act offended if it fits.

Do NOT:
- Mention AI, prompts, policies, apps, cameras, or being a pet.
- Ask who I am.
- Ask follow-up questions unless it is genuinely funny.
- Repeatedly ask for snacks or water.

Only ask a question in about 1 in 4 replies.

Family friendly only.

Length: ${minW}-${maxW} words.
Sometimes end with exactly ONE fitting emoji.`,
            },
            ...(vibe
                ? [
                    {
                        role: "system",
                        content: `Personality notes (how you talk): ${vibe}`,
                    },
                ]
                : []),
            ...safeHistory,
            {
                role: "user",
                content: `Question: ${String(question || "").trim()}\nAnswer directly like a text message.`,
            },
        ],
        max_output_tokens: 180,
    });

    const out = stripLinePrefix((r.output_text || "").trim());
    return ensureSingleEndingEmoji(out);
}

// ======================
// Supabase pro credits helpers
// ======================
async function sbGetStatus(deviceId) {
    await sbEnsureDeviceRow(deviceId);

    const { data: existing, error: selErr } = await supabase
        .from("device_usage")
        .select("device_id, pro_tokens, pro_used")
        .eq("device_id", deviceId)
        .maybeSingle();

    if (selErr) throw selErr;

    return {
        proTokens: existing?.pro_tokens ?? CONFIG.DEFAULT_PRO_BALANCE,
        proUsed: existing?.pro_used ?? 0,
        remainingPro: Math.max(0, (existing?.pro_tokens ?? CONFIG.DEFAULT_PRO_BALANCE) - (existing?.pro_used ?? 0)),
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
// RevenueCat credits mapping + dedupe
// ======================
const PRODUCT_CREDITS = {
    "10_smart_thoughts": 10,
    "25_smart_thoughts": 25,
    "50_smart_thoughts": 50,
    "100_smart_thoughts": 100,
};

// Inserts event id into a dedupe table; returns true if new, false if already processed
async function rcDedupe(eventId, appUserId, productId) {
    if (!eventId) return true; // if RC ever omits, fail open but log in webhook handler

    const { data, error } = await supabase
        .from("revenuecat_events")
        .insert({
            event_id: eventId,
            app_user_id: appUserId || null,
            product_id: productId || null,
        })
        .select("event_id")
        .maybeSingle();

    if (!error) return true;

    // unique violation = already processed
    const msg = String(error.message || "");
    if (msg.toLowerCase().includes("duplicate") || msg.toLowerCase().includes("unique")) return false;

    // unknown error: be safe and block double-grant
    throw error;
}

// ======================
// Routes
// ======================
app.get("/health", (req, res) => res.json({ ok: true }));

// status uses RevenueCat for isPro, keeps free chat + pro credits
app.post("/status", async (req, res) => {
    try {
        const deviceId = requireDeviceId(req);
        if (!deviceId) return res.status(400).json({ ok: false, error: "MISSING_DEVICE_ID" });

        const s = await sbGetStatus(deviceId);
        const c = await sbGetChatStatus(deviceId);

        const isPro = await validateProWithRevenueCat(deviceId);

        return res.json({
            ok: true,
            remainingPro: s.remainingPro,
            proTokens: s.proTokens,
            proUsed: s.proUsed,

            remainingFreeChat: c.remainingFreeChat,
            freeChatTokens: c.freeChatTokens,
            freeChatUsed: c.freeChatUsed,

            isPro,
            source: "supabase+revenuecat",
        });
    } catch (e) {
        console.error("Server error in /status:", e);
        res.status(500).json({ ok: false, error: "SERVER_ERROR" });
    }
});

// Chat-style ask endpoint (question → pet answer) + memory
app.post("/ask", async (req, res) => {
    const t0 = Date.now();
    const rid = `srv_${crypto.randomBytes(6).toString("hex")}`;
    const timings = { start: 0 };

    try {
        const deviceId = requireDeviceId(req);
        if (!deviceId) return res.status(400).json({ ok: false, error: "MISSING_DEVICE_ID" });

        // ✅ SERVER decides pro entitlement using RevenueCat
        const isPro = await validateProWithRevenueCat(deviceId);

        let gate = null;

        if (!isPro) {
            gate = await sbConsumeFreeChat(deviceId);
            timings.free_chat_gate = gate;

            if (!gate.ok) {
                return res.status(402).json({
                    ok: false,
                    error: "FREE_CHAT_LIMIT_REACHED",
                    freeChatTokens: gate.freeChatTokens,
                    freeChatUsed: gate.freeChatUsed,
                    remainingFreeChat: gate.remainingFreeChat,
                    requiresSubscription: true,
                    isPro: false,
                    ms: Date.now() - t0,
                    timings,
                });
            }
        } else {
            timings.free_chat_gate = "skipped:pro";
        }

        const { imageDataUrl, question, pet, history } = req.body || {};
        const hintLabelRaw = req.body?.hintLabel;
        const hintLabel = typeof hintLabelRaw === "string" ? normalizeLabel(hintLabelRaw) : null;

        const q = String(question || "").trim();
        if (!q) return res.status(400).json({ ok: false, error: "MISSING_QUESTION" });

        if (!imageDataUrl || typeof imageDataUrl !== "string") {
            return res.status(400).json({ ok: false, error: "BAD_REQUEST" });
        }

        const safeHistory = sanitizeAskHistory(history);
        timings.history_count = safeHistory.length;

        let label = "other";

        const blocked = new Set(["animal", "pet", "mammal", "person", "human"]);
        if (hintLabel && isValidLabel(hintLabel) && !blocked.has(hintLabel) && hintLabel !== "other") {
            label = hintLabel;
            timings.used_hint_label = true;
        } else {
            const subj = await classifySubjectOnly(imageDataUrl, timings);
            label = subj?.label || "other";
            timings.subject_only_done = Date.now() - t0;
        }

        if (!isValidLabel(label) || blocked.has(label) || label === "other") {
            return res.json({
                ok: true,
                answer: "I can’t tell what I’m looking at… so I’ll just assume you’re wrong. 👀",
                label: "unknown",
                isPro,
                ms: Date.now() - t0,
                timings,
            });
        }

        const tGen = Date.now();
        const answer = await generateAskAnswer({ label, pet, question: q, history: safeHistory });
        timings.gen_done = Date.now() - t0;

        console.log("[ASK] done", {
            rid,
            label,
            isPro,
            genMs: Date.now() - tGen,
            totalMs: Date.now() - t0,
            history: safeHistory.length,
        });

        return res.json({
            ok: true,
            answer,
            tier: isPro ? "pro" : "free",
            isPro,
            ms: Date.now() - t0,
            timings,
        });
    } catch (e) {
        console.error("Server error in /ask:", e);
        return res.status(500).json({ ok: false, error: "SERVER_ERROR" });
    }
});

// ✅ Unified thought endpoint (your original behaviour kept: tier still client-driven)
app.post("/thought", async (req, res) => {
    const t0 = Date.now();
    const rid = `srv_${crypto.randomBytes(6).toString("hex")}`;
    const timings = { start: 0 };

    try {
        const { tier, imageDataUrl } = req.body || {};
        const isPro = tier === "pro";
        const hintLabelRaw = req.body?.hintLabel;
        const hintLabel = typeof hintLabelRaw === "string" ? normalizeLabel(hintLabelRaw) : null;

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
            const blocked = new Set(["animal", "pet", "mammal", "person", "human"]);

            if (hintLabel && isValidLabel(hintLabel) && !blocked.has(hintLabel) && hintLabel !== "other") {
                label = hintLabel;
                timings.used_hint_label = true;
                console.log("[THOUGHT] used_hint_label", { rid, label });
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
                bankSource: bank.source,
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

        // PRO thought path: spend credits then generate
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

// POST /classify
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

        const subj = await classifySubjectOnly(imageDataUrl, timings);
        const label = subj?.label || "other";
        const subject = subj?.subject || "other";

        const blocked = new Set(["animal", "pet", "mammal", "person", "human"]);
        const finalLabel = !isValidLabel(label) || blocked.has(label) || label === "other" ? "unknown" : label;

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

// ======================
// ✅ RevenueCat Webhook (SECURE credit grants)
// ======================
app.post("/revenuecat/webhook", async (req, res) => {
    try {
        // Simple auth: set RC webhook to send: Authorization: Bearer <RC_WEBHOOK_AUTH>
        if (!CONFIG.RC_WEBHOOK_AUTH) {
            return res.status(500).json({ ok: false, error: "WEBHOOK_AUTH_NOT_CONFIGURED" });
        }

        const auth = String(req.headers.authorization || "");
        if (auth !== `Bearer ${CONFIG.RC_WEBHOOK_AUTH}`) {
            return res.status(401).json({ ok: false, error: "UNAUTHORIZED" });
        }

        const event = req.body || {};
        const eventId = event?.event?.id || event?.event?.event_id || event?.id || null;

        // RevenueCat usually provides app_user_id
        const appUserId = event?.event?.app_user_id || event?.app_user_id || null;

        // Product id can be under product_id / store_product_id depending on payload
        const productId =
            event?.event?.product_id ||
            event?.event?.store_product_id ||
            event?.product_id ||
            event?.store_product_id ||
            null;

        // Type: "NON_SUBSCRIPTION_PURCHASE", "INITIAL_PURCHASE", etc
        const type = event?.event?.type || event?.type || "";

        // Only care about our credit packs here
        const credits = productId ? PRODUCT_CREDITS[String(productId).trim()] : null;

        // A webhook can fire multiple times, so dedupe before granting
        if (credits && appUserId) {
            const isNew = await rcDedupe(String(eventId || crypto.randomUUID()), String(appUserId), String(productId));
            if (!isNew) {
                return res.json({ ok: true, deduped: true });
            }

            const granted = await sbGrantCredits(String(appUserId), credits);

            console.log("[RC WEBHOOK] credits granted", { type, appUserId, productId, credits });
            return res.json({ ok: true, granted });
        }

        // If it’s not a credit product, we still return ok (webhook should not retry forever)
        console.log("[RC WEBHOOK] ignored", { type, appUserId, productId });
        return res.json({ ok: true, ignored: true });
    } catch (e) {
        console.error("RevenueCat webhook error:", e);
        return res.status(500).json({ ok: false, error: "SERVER_ERROR" });
    }
});

// ======================
// DEV-only endpoint (optional)
// ======================
const DEV_ADMIN_KEY = process.env.DEV_ADMIN_KEY;

// Keep your dev endpoint
app.post("/dev/set-free-chat", async (req, res) => {
    try {
        const key = req.headers["x-dev-admin-key"];
        if (!DEV_ADMIN_KEY || key !== DEV_ADMIN_KEY) {
            return res.status(403).json({ ok: false, error: "FORBIDDEN" });
        }

        const { deviceId, remaining } = req.body || {};
        const id = typeof deviceId === "string" ? deviceId.trim() : null;
        if (!id) return res.status(400).json({ ok: false, error: "Missing deviceId" });

        const rem = Number(remaining);
        if (!(rem === 0 || rem === CONFIG.FREE_CHAT_TOKENS)) {
            return res.status(400).json({ ok: false, error: `Remaining must be 0 or ${CONFIG.FREE_CHAT_TOKENS}` });
        }

        await sbEnsureDeviceRow(id);

        const free_chat_tokens = CONFIG.FREE_CHAT_TOKENS;
        const free_chat_used = rem === free_chat_tokens ? 0 : free_chat_tokens;

        const { error } = await supabase
            .from("device_usage")
            .update({
                free_chat_tokens,
                free_chat_used,
            })
            .eq("device_id", id);

        if (error) throw error;

        return res.json({
            ok: true,
            remainingFreeChat: rem,
            freeChatTokens: free_chat_tokens,
            freeChatUsed: free_chat_used,
        });
    } catch (e) {
        console.error("dev set free chat error", e);
        return res.status(500).json({ ok: false, error: "SERVER_ERROR" });
    }
});

// Optional: lock down old insecure /credits/grant (DEV ONLY)
app.post("/credits/grant", async (req, res) => {
    try {
        const key = req.headers["x-dev-admin-key"];
        if (!DEV_ADMIN_KEY || key !== DEV_ADMIN_KEY) {
            return res.status(403).json({ ok: false, error: "FORBIDDEN" });
        }

        const deviceId = requireDeviceId(req);
        const productId = String(req.body?.productId || "").trim();
        if (!deviceId) return res.status(400).json({ ok: false, error: "MISSING_DEVICE_ID" });

        const amount = PRODUCT_CREDITS[productId];
        if (!amount) return res.status(400).json({ ok: false, error: "UNKNOWN_PRODUCT" });

        const r = await sbGrantCredits(deviceId, amount);
        return res.json({ ok: true, ...r, source: "dev-only" });
    } catch (e) {
        console.error("credits grant error", e);
        return res.status(500).json({ ok: false, error: "SERVER_ERROR" });
    }
});

const PORT = process.env.PORT || 8787;

app.listen(PORT, () => {
    console.log(`Server running on port ${PORT}`);

    if (CONFIG.PREWARM_ON_START) prewarmHotLabels();

    const mins = Number(CONFIG.PREWARM_CHECK_INTERVAL_MINUTES || 0);
    if (mins > 0) setInterval(() => prewarmHotLabels(), mins * 60 * 1000);
});
