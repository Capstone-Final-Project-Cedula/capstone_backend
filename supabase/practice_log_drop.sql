-- Plan 001.7: removes the TEMPORARY practice session log.
-- Kept OUTSIDE supabase/migrations/ on purpose, so it is never applied by accident.
-- Run it by hand in the Supabase SQL editor when plan 003 arrives or before the real study.
-- It drops exactly what migrations/20261021090000_practice_log.sql created; nothing else.
-- Afterwards restart the backend with PRACTICE_LOG_ENABLED off (or unset) - with the
-- log on, the backend refuses to start without these tables.
-- This deletes everything the log holds (what was heard, typed names). Cannot be undone.

drop table if exists public.practice_log_answers;
drop table if exists public.practice_log_sessions;
drop table if exists public.practice_log_settings;
