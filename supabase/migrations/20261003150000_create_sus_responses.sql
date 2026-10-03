-- System Usability Scale (SUS) survey responses from the web and mobile apps.
-- The apps never connect to Supabase directly: the FastAPI backend (table
-- owner) inserts responses and serves them to admin/staff accounts, so the
-- table is closed to the public Data API like every other CeduSync table.

create table public.sus_responses (
  id                  uuid primary key default gen_random_uuid(),
  created_at          timestamptz not null default now(),
  respondent_name     text check (respondent_name is null or char_length(respondent_name) <= 120),
  role                text not null
                      check (role in ('barangay_secretary', 'cedula_personnel', 'citizen_applicant')),
  computer_experience text not null
                      check (computer_experience in ('none', 'basic', 'comfortable', 'advanced')),
  features_used       text[] not null default '{}'
                      check (features_used <@ array['voice_input', 'qr_prefill', 'form_generation',
                                                    'receipt', 'voice_feedback']::text[]),
  q1  smallint not null check (q1  between 1 and 5),
  q2  smallint not null check (q2  between 1 and 5),
  q3  smallint not null check (q3  between 1 and 5),
  q4  smallint not null check (q4  between 1 and 5),
  q5  smallint not null check (q5  between 1 and 5),
  q6  smallint not null check (q6  between 1 and 5),
  q7  smallint not null check (q7  between 1 and 5),
  q8  smallint not null check (q8  between 1 and 5),
  q9  smallint not null check (q9  between 1 and 5),
  q10 smallint not null check (q10 between 1 and 5),
  -- Standard SUS scoring: odd items (positive) score q-1, even items
  -- (negative) score 5-q; the 0..40 sum times 2.5 gives 0..100.
  sus_score numeric generated always as (
    ((q1 - 1) + (5 - q2) + (q3 - 1) + (5 - q4) + (q5 - 1)
     + (5 - q6) + (q7 - 1) + (5 - q8) + (q9 - 1) + (5 - q10)) * 2.5
  ) stored,
  comments            text check (comments is null or char_length(comments) <= 2000),
  consent_given       boolean not null check (consent_given),
  platform            text not null check (platform in ('web', 'mobile')),
  transaction_id      integer references public.ctc_issuance (id) on delete set null,
  synced_to_sheets    boolean not null default false
);

comment on table public.sus_responses is
  'SUS usability survey answers (capstone research). Copied to the "CeduSync SUS Survey Results" Google Sheet by the sus-sheets-sync Edge Function.';

create index sus_responses_transaction_id_idx on public.sus_responses (transaction_id);
create index sus_responses_unsynced_idx on public.sus_responses (created_at) where not synced_to_sheets;

-- Closed to the Data API: inserts and reads go through the backend, which
-- enforces "anyone may submit, only admin/staff may read".
alter table public.sus_responses enable row level security;
revoke all on table public.sus_responses from anon, authenticated;
