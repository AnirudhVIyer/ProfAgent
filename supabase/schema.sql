-- ============================================================
-- SAGE — Complete Supabase Schema
-- Run this entire script in Supabase SQL Editor
-- for a fresh setup.
-- ============================================================


-- ── Extensions ──────────────────────────────────────────────
create extension if not exists vector;


-- ── Profiles ────────────────────────────────────────────────
create table public.profiles (
    id uuid references auth.users(id) on delete cascade primary key,
    email text not null,
    display_name text,
    role text not null default 'user',
    is_active boolean not null default true,
    created_at timestamptz default now(),
    updated_at timestamptz default now()
);

create or replace function public.handle_new_user()
returns trigger as $$
begin
    insert into public.profiles (id, email, display_name)
    values (
        new.id,
        new.email,
        coalesce(
            new.raw_user_meta_data->>'display_name',
            split_part(new.email, '@', 1)
        )
    );
    return new;
end;
$$ language plpgsql security definer;

create trigger on_auth_user_created
    after insert on auth.users
    for each row execute procedure public.handle_new_user();


-- ── Knowledge entries ────────────────────────────────────────
create table public.knowledge_entries (
    id uuid default gen_random_uuid() primary key,
    user_id uuid references auth.users(id) on delete cascade not null,
    topic text not null,
    confidence text not null default 'low',
    source text not null default 'seed',
    connected_to text[] default '{}',
    embedding vector(1024),
    date_learned timestamptz default now(),
    updated_at timestamptz default now(),
    unique(user_id, topic)
);

create index knowledge_entries_embedding_idx
    on public.knowledge_entries
    using ivfflat (embedding vector_cosine_ops)
    with (lists = 100);

create index knowledge_entries_user_idx
    on public.knowledge_entries(user_id);


-- ── Sub concepts ─────────────────────────────────────────────
create table public.sub_concepts (
    id uuid default gen_random_uuid() primary key,
    user_id uuid references auth.users(id) on delete cascade not null,
    knowledge_entry_id uuid references public.knowledge_entries(id)
        on delete cascade not null,
    name text not null,
    status text not null default 'gap',
    confidence text default 'low',
    notes text,
    created_at timestamptz default now(),
    updated_at timestamptz default now(),
    unique(knowledge_entry_id, name)
);

create index sub_concepts_entry_idx
    on public.sub_concepts(knowledge_entry_id);
create index sub_concepts_user_idx
    on public.sub_concepts(user_id);


-- ── Sessions ─────────────────────────────────────────────────
create table public.sessions (
    id uuid default gen_random_uuid() primary key,
    user_id uuid references auth.users(id) on delete cascade not null,
    session_type text not null,
    topic text,
    started_at timestamptz default now(),
    ended_at timestamptz,
    message_count integer default 0,
    tokens_used integer default 0,
    compressed boolean default false,
    transcript text,
    summary text
);

create index sessions_user_idx
    on public.sessions(user_id);
create index sessions_date_idx
    on public.sessions(user_id, started_at);


-- ── Rate limits ──────────────────────────────────────────────
create table public.rate_limits (
    id uuid default gen_random_uuid() primary key,
    user_id uuid references auth.users(id) on delete cascade not null,
    date date not null default current_date,
    llm_calls integer not null default 0,
    tokens_used integer not null default 0,
    tavily_calls integer not null default 0,
    max_llm_calls integer not null default 50,
    max_tokens integer not null default 100000,
    max_tavily_calls integer not null default 20,
    unique(user_id, date)
);

create index rate_limits_user_date_idx
    on public.rate_limits(user_id, date);


-- ── Daily briefs ─────────────────────────────────────────────
create table public.daily_briefs (
    id uuid default gen_random_uuid() primary key,
    user_id uuid references auth.users(id) on delete cascade not null,
    date date not null default current_date,
    topic_title text not null,
    explanation text,
    connections text[],
    why_it_matters text,
    discussion_questions text[],
    email_hook text,
    tldr text,
    source_url text,
    difficulty text,
    created_at timestamptz default now(),
    unique(user_id, date)
);

create index daily_briefs_user_idx
    on public.daily_briefs(user_id, date);


-- ── Access requests ──────────────────────────────────────────
create table public.access_requests (
    id uuid default gen_random_uuid() primary key,
    name text not null,
    email text not null,
    note text,
    status text not null default 'pending',
    requested_at timestamptz default now(),
    reviewed_at timestamptz,
    unique(email)
);


-- ── Error logs ───────────────────────────────────────────────
create table public.error_logs (
    id uuid default gen_random_uuid() primary key,
    severity text not null,
    layer text not null,
    function_name text not null,
    error_type text not null,
    message text not null,
    user_id uuid references auth.users(id) on delete set null,
    context jsonb default '{}',
    traceback text,
    created_at timestamptz default now()
);

create index error_logs_severity_idx
    on public.error_logs(severity, created_at);
create index error_logs_user_idx
    on public.error_logs(user_id, created_at);


-- ── Row Level Security ───────────────────────────────────────

alter table public.profiles enable row level security;
alter table public.knowledge_entries enable row level security;
alter table public.sub_concepts enable row level security;
alter table public.sessions enable row level security;
alter table public.rate_limits enable row level security;
alter table public.daily_briefs enable row level security;
alter table public.access_requests enable row level security;
alter table public.error_logs enable row level security;

-- Profiles
create policy "Users see own profile"
    on public.profiles for all
    using (auth.uid() = id);

-- Knowledge entries
create policy "Users see own knowledge"
    on public.knowledge_entries for all
    using (auth.uid() = user_id);

-- Sub concepts
create policy "Users see own sub concepts"
    on public.sub_concepts for all
    using (auth.uid() = user_id);

-- Sessions
create policy "Users see own sessions"
    on public.sessions for all
    using (auth.uid() = user_id);

-- Rate limits
create policy "Users see own rate limits"
    on public.rate_limits for all
    using (auth.uid() = user_id);

-- Daily briefs
create policy "Users see own briefs"
    on public.daily_briefs for all
    using (auth.uid() = user_id);

-- Access requests — anyone can insert, only admins can read
create policy "Anyone can request access"
    on public.access_requests for insert
    with check (true);

create policy "Admins manage access requests"
    on public.access_requests for select
    using (
        exists (
            select 1 from public.profiles
            where id = auth.uid()
            and role = 'admin'
        )
    );

create policy "Admins update access requests"
    on public.access_requests for update
    using (
        exists (
            select 1 from public.profiles
            where id = auth.uid()
            and role = 'admin'
        )
    );

-- Error logs — only admins
create policy "Admins see error logs"
    on public.error_logs for all
    using (
        exists (
            select 1 from public.profiles
            where id = auth.uid()
            and role = 'admin'
        )
    );


-- ── Helper functions ─────────────────────────────────────────

create or replace function public.get_or_create_rate_limit(
    p_user_id uuid
)
returns public.rate_limits as $$
declare
    result public.rate_limits;
begin
    insert into public.rate_limits (user_id, date)
    values (p_user_id, current_date)
    on conflict (user_id, date) do nothing;

    select * into result
    from public.rate_limits
    where user_id = p_user_id
    and date = current_date;

    return result;
end;
$$ language plpgsql security definer;


create or replace function public.increment_usage(
    p_user_id uuid,
    p_llm_calls integer default 0,
    p_tokens integer default 0,
    p_tavily_calls integer default 0
)
returns public.rate_limits as $$
declare
    result public.rate_limits;
begin
    update public.rate_limits
    set
        llm_calls = llm_calls + p_llm_calls,
        tokens_used = tokens_used + p_tokens,
        tavily_calls = tavily_calls + p_tavily_calls
    where user_id = p_user_id
    and date = current_date
    returning * into result;

    return result;
end;
$$ language plpgsql security definer;


-- ── Vector similarity search ─────────────────────────────────

create or replace function match_knowledge_entries(
    query_embedding vector(1024),
    match_threshold float,
    match_count int,
    p_user_id uuid
)
returns table (
    id uuid,
    topic text,
    confidence text,
    similarity float
)
language sql stable
as $$
    select
        id,
        topic,
        confidence,
        1 - (embedding <=> query_embedding) as similarity
    from knowledge_entries
    where
        user_id = p_user_id
        and embedding is not null
        and 1 - (embedding <=> query_embedding) > match_threshold
    order by embedding <=> query_embedding
    limit match_count;
$$;