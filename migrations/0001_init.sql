-- Vendi Retail Intelligence Platform — initial schema
-- Tables, indexes, and Row-Level Security policies per TECHNICAL_SPEC.md §6.

create extension if not exists "pgcrypto";

-- ---------------------------------------------------------------------------
-- updated_at trigger helper
-- ---------------------------------------------------------------------------

create or replace function set_updated_at()
returns trigger as $$
begin
  new.updated_at = now();
  return new;
end;
$$ language plpgsql;

-- ---------------------------------------------------------------------------
-- organizations: one per retailer/tenant
-- ---------------------------------------------------------------------------

create table organizations (
    id          uuid primary key default gen_random_uuid(),
    name        text not null,
    created_at  timestamptz not null default now(),
    updated_at  timestamptz not null default now()
);

create trigger organizations_set_updated_at
    before update on organizations
    for each row execute function set_updated_at();

-- ---------------------------------------------------------------------------
-- memberships: which auth user belongs to which org, and with what role
-- ---------------------------------------------------------------------------

create table memberships (
    user_id     uuid not null references auth.users(id) on delete cascade,
    org_id      uuid not null references organizations(id) on delete cascade,
    role        text not null check (role in ('owner', 'admin', 'member')),
    created_at  timestamptz not null default now(),
    updated_at  timestamptz not null default now(),
    primary key (user_id, org_id)
);

create index memberships_org_id_idx on memberships(org_id);

create trigger memberships_set_updated_at
    before update on memberships
    for each row execute function set_updated_at();

-- ---------------------------------------------------------------------------
-- datasets: an uploaded transactions file
-- ---------------------------------------------------------------------------

create table datasets (
    id                  uuid primary key default gen_random_uuid(),
    org_id              uuid not null references organizations(id) on delete cascade,
    name                text not null,
    storage_path        text,                 -- raw CSV in Storage
    parquet_path        text,                 -- processed Parquet
    status              text not null check (status in ('validating', 'ready', 'failed'))
                            default 'validating',
    validation_report   jsonb,
    row_count           integer,
    customer_count      integer,
    product_count       integer,
    date_start          date,
    date_end            date,
    created_at          timestamptz not null default now(),
    updated_at          timestamptz not null default now()
);

create index datasets_org_id_idx on datasets(org_id);

create trigger datasets_set_updated_at
    before update on datasets
    for each row execute function set_updated_at();

-- ---------------------------------------------------------------------------
-- jobs: one async module run
-- ---------------------------------------------------------------------------

create table jobs (
    id                      uuid primary key default gen_random_uuid(),
    org_id                  uuid not null references organizations(id) on delete cascade,
    dataset_id              uuid not null references datasets(id) on delete cascade,
    module                  text not null check (module in ('forecast', 'basket', 'customer')),
    status                  text not null check (status in ('queued', 'running', 'succeeded', 'failed'))
                                default 'queued',
    params                  jsonb not null default '{}'::jsonb,
    result_summary          jsonb,
    result_storage_path     text,
    error                   text,
    started_at              timestamptz,
    finished_at             timestamptz,
    created_at              timestamptz not null default now(),
    updated_at              timestamptz not null default now()
);

create index jobs_org_id_status_idx on jobs(org_id, status);
create index jobs_dataset_id_idx on jobs(dataset_id);

create trigger jobs_set_updated_at
    before update on jobs
    for each row execute function set_updated_at();

-- ---------------------------------------------------------------------------
-- Row-Level Security
--
-- Policy pattern: a row is visible/mutable only if its org_id is among the
-- caller's memberships. The worker uses the service-role key (bypasses RLS)
-- and MUST explicitly filter every query by the job's org_id — RLS here is
-- defense-in-depth for the user-facing API path only.
-- ---------------------------------------------------------------------------

alter table organizations enable row level security;
alter table memberships  enable row level security;
alter table datasets      enable row level security;
alter table jobs          enable row level security;

-- organizations: visible to members; no client-side insert/update/delete
-- (orgs are provisioned by an admin/service process).
create policy organizations_select on organizations
    for select
    using (
        id in (select org_id from memberships where user_id = auth.uid())
    );

-- memberships: a user can see their own membership rows (and thus discover
-- their orgs + role). No client-side writes.
create policy memberships_select on memberships
    for select
    using (user_id = auth.uid());

-- datasets: full CRUD scoped to the caller's orgs.
create policy datasets_select on datasets
    for select
    using (org_id in (select org_id from memberships where user_id = auth.uid()));

create policy datasets_insert on datasets
    for insert
    with check (org_id in (select org_id from memberships where user_id = auth.uid()));

create policy datasets_update on datasets
    for update
    using (org_id in (select org_id from memberships where user_id = auth.uid()))
    with check (org_id in (select org_id from memberships where user_id = auth.uid()));

create policy datasets_delete on datasets
    for delete
    using (org_id in (select org_id from memberships where user_id = auth.uid()));

-- jobs: select + insert scoped to the caller's orgs. Updates (status
-- transitions, results) are performed by the worker via the service-role
-- key, which bypasses RLS entirely.
create policy jobs_select on jobs
    for select
    using (org_id in (select org_id from memberships where user_id = auth.uid()));

create policy jobs_insert on jobs
    for insert
    with check (org_id in (select org_id from memberships where user_id = auth.uid()));
