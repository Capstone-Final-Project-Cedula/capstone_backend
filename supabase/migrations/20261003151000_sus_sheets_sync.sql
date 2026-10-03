-- Send SUS survey responses to the "CeduSync SUS Survey Results" Google Sheet.
--
-- Flow: insert into sus_responses -> trigger -> sus_request_sync() queues an
-- HTTP call (pg_net, sent after commit) to the sus-sheets-sync Edge Function,
-- which posts every unsynced row to the sheet's Apps Script web app and sets
-- synced_to_sheets = true on success. A pg_cron job repeats the call every
-- 10 minutes, so rows missed while the internet was down are sent later.
--
-- The Apps Script URL and shared secret are Edge Function secrets
-- (SHEETS_WEBAPP_URL, SHEETS_SHARED_SECRET), never stored here or in the apps.
-- sus_sync_key (Vault) only proves a call to the Edge Function came from this
-- database; it is generated here and read back by the function.

-- pg_net is registered in the extensions schema (its functions live in "net").
create extension if not exists pg_net with schema extensions;
create extension if not exists pg_cron;

select vault.create_secret(
  encode(extensions.gen_random_bytes(32), 'hex'),
  'sus_sync_key',
  'Authenticates database calls to the sus-sheets-sync Edge Function'
)
where not exists (select 1 from vault.secrets where name = 'sus_sync_key');

-- Ask the Edge Function to send unsynced responses. Returns how many are waiting.
create or replace function public.sus_request_sync()
returns integer
language plpgsql
security definer
set search_path = ''
as $$
declare
  pending integer;
  sync_key text;
begin
  select count(*) into pending from public.sus_responses where not synced_to_sheets;
  if pending = 0 then
    return 0;
  end if;

  select decrypted_secret into sync_key from vault.decrypted_secrets where name = 'sus_sync_key';
  perform net.http_post(
    url := 'https://emcbkampzotuajfkxmix.supabase.co/functions/v1/sus-sheets-sync',
    headers := jsonb_build_object('Content-Type', 'application/json', 'x-sync-key', sync_key),
    body := '{}'::jsonb,
    timeout_milliseconds := 60000
  );
  return pending;
end;
$$;

-- Never let a sync problem block saving a survey.
create or replace function public.sus_responses_request_sync()
returns trigger
language plpgsql
security definer
set search_path = ''
as $$
begin
  perform public.sus_request_sync();
  return null;
exception when others then
  raise warning 'sus_request_sync failed: %', sqlerrm;
  return null;
end;
$$;

revoke execute on function public.sus_request_sync() from public, anon, authenticated;
revoke execute on function public.sus_responses_request_sync() from public, anon, authenticated;

create trigger sus_responses_sync_to_sheets
  after insert on public.sus_responses
  for each statement
  execute function public.sus_responses_request_sync();

select cron.schedule('sus-sheets-resync', '*/10 * * * *', 'select public.sus_request_sync()');
