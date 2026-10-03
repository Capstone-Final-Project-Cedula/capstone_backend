// sus-sheets-sync — copies SUS survey responses to the Google Sheet.
//
// Called by the database (trigger on new responses, pg_cron every 10 minutes,
// and the admin "Resync" button through public.sus_request_sync()). The
// request body is ignored: each run sends every row where
// synced_to_sheets = false to the Apps Script web app and marks it synced
// when the script confirms. The script skips Response IDs it already has, so
// a repeated send never creates a duplicate row.
//
// Secrets (Edge Functions > Secrets): SHEETS_WEBAPP_URL, SHEETS_SHARED_SECRET.
// SUPABASE_DB_URL is provided by Supabase. Callers must send the x-sync-key
// header matching the Vault secret sus_sync_key (only the database has it).
import "jsr:@supabase/functions-js/edge-runtime.d.ts";
import postgres from "npm:postgres@3.4.5";

const BATCH_SIZE = 25;
const TIME_BUDGET_MS = 100_000; // stay well inside the function's wall-clock limit

const sql = postgres(Deno.env.get("SUPABASE_DB_URL")!, { max: 1, prepare: false });

function json(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

function safeEqual(a: string, b: string): boolean {
  const x = new TextEncoder().encode(a);
  const y = new TextEncoder().encode(b);
  if (x.length !== y.length) return false;
  let diff = 0;
  for (let i = 0; i < x.length; i++) diff |= x[i] ^ y[i];
  return diff === 0;
}

async function sendToSheet(url: string, secret: string, row: Record<string, unknown>) {
  // Apps Script answers a POST with a redirect to the script's output; fetch
  // follows it and the final body is the script's JSON reply.
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "text/plain;charset=utf-8" },
    body: JSON.stringify({ secret, response: row }),
    redirect: "follow",
  });
  const text = await res.text();
  let reply: { ok?: boolean; status?: string; error?: string } = {};
  try {
    reply = JSON.parse(text);
  } catch {
    throw new Error(`Apps Script returned HTTP ${res.status}, not JSON (check the deployment access is "Anyone")`);
  }
  if (!reply.ok) throw new Error(reply.error ?? `Apps Script returned HTTP ${res.status}`);
  return reply.status ?? "appended";
}

Deno.serve(async (req) => {
  if (req.method !== "POST") return json({ error: "Method not allowed" }, 405);

  const [stored] = await sql`
    select decrypted_secret as key from vault.decrypted_secrets where name = 'sus_sync_key'`;
  if (!stored?.key || !safeEqual(req.headers.get("x-sync-key") ?? "", stored.key)) {
    return json({ error: "Unauthorized" }, 401);
  }

  const url = Deno.env.get("SHEETS_WEBAPP_URL");
  const secret = Deno.env.get("SHEETS_SHARED_SECRET");
  if (!url || !secret) {
    return json({ error: "SHEETS_WEBAPP_URL and SHEETS_SHARED_SECRET are not set" }, 500);
  }

  const started = Date.now();
  const result = { sent: 0, duplicates: 0, failed: 0, errors: [] as string[] };

  // Lock the batch so two overlapping runs never send the same row.
  await sql.begin(async (tx) => {
    const rows = await tx`
      select id, created_at, respondent_name, role, computer_experience, features_used,
             q1, q2, q3, q4, q5, q6, q7, q8, q9, q10, comments, platform
      from public.sus_responses
      where not synced_to_sheets
      order by created_at
      limit ${BATCH_SIZE}
      for update skip locked`;

    for (const row of rows) {
      if (Date.now() - started > TIME_BUDGET_MS) break;
      try {
        const status = await sendToSheet(url, secret, { ...row, created_at: row.created_at.toISOString() });
        await tx`update public.sus_responses set synced_to_sheets = true where id = ${row.id}`;
        if (status === "duplicate") result.duplicates++;
        else result.sent++;
      } catch (err) {
        result.failed++;
        result.errors.push(`${row.id}: ${err instanceof Error ? err.message : String(err)}`);
      }
    }
  });

  if (result.failed) console.error("sus-sheets-sync failures", result.errors);
  return json(result, result.failed && !result.sent && !result.duplicates ? 502 : 200);
});
