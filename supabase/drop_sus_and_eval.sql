-- Plan 012: removes what the in-system SUS survey and Evaluation mode left in the database.
-- Kept OUTSIDE supabase/migrations/ on purpose, so it is never applied by accident (like practice_log_drop.sql).
-- NOT RUN: Ralph chose option A (2026-11-21): the tables and the bucket stay, unused, until he decides.
-- Run it by hand in the Supabase SQL editor only after:
--   1. the plan 012 backend is live (the old backend refuses to start without the eval tables);
--   2. a backup / CSV export of anything the paper still needs (sus_responses, eval_* rows);
--   3. the eval-audio clips are deleted (the "Delete evaluation audio now" button went away with the code,
--      so empty the bucket in the dashboard: Storage -> eval-audio).
-- It drops exactly what these migrations created, newest first; nothing else:
--   20261110090000_eval_cards.sql, 20261028090000_evaluation_mode.sql,
--   20261003151000_sus_sheets_sync.sql, 20261003150000_create_sus_responses.sql
-- Not done by SQL (dashboard, by hand): delete the Edge Function sus-sheets-sync and its secrets
-- SHEETS_WEBAPP_URL and SHEETS_SHARED_SECRET; delete the then-empty private bucket eval-audio.
-- The pg_net and pg_cron extensions stay (other jobs may use them).
-- This deletes every survey answer, evaluation session, hand score and security count. Cannot be undone.

begin;

-- SUS -> Google Sheets sync: the 10-minute cron job, the trigger, its two functions and the Vault key.
select cron.unschedule(jobid) from cron.job where jobname = 'sus-sheets-resync';
drop trigger if exists sus_responses_sync_to_sheets on public.sus_responses;
drop function if exists public.sus_responses_request_sync();
drop function if exists public.sus_request_sync();
delete from vault.secrets where name = 'sus_sync_key';

-- Evaluation mode tables. eval_cards and eval_sessions point at each other, so they go in one statement.
drop table if exists public.eval_answers;
drop table if exists public.eval_cards, public.eval_sessions;
drop table if exists public.security_events;

-- The in-system SUS survey answers.
drop table if exists public.sus_responses;

-- The six evaluation columns on app_settings (each is NOT NULL DEFAULT ..., nothing else reads them).
alter table public.app_settings
  drop column if exists evaluation_mode,
  drop column if exists eval_round_type,
  drop column if exists eval_round_condition,
  drop column if exists eval_round_card_set,
  drop column if exists eval_round_no,
  drop column if exists eval_counts;

commit;
