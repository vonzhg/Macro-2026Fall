-- =============================================================================
-- ECON 282E course forum — database schema for Supabase (Postgres 15+)
--
-- Run once, in the Supabase dashboard: SQL Editor -> New query -> paste -> Run.
-- It is safe to re-run: every object is dropped and re-created, EXCEPT the
-- tables that hold data (members, profiles, categories, threads, posts, votes,
-- thread_aliases, activity), which are created only if missing.
--
-- Security model, in one paragraph.
--   * Clients never touch a table. Every table has row-level security switched
--     on and NO policy, and all privileges are revoked from `anon` and
--     `authenticated`. (The one exception is `activity`, read-only, ids only,
--     so the realtime channel can notify browsers.)
--   * Reads go through two views, `threads_view` and `posts_view`. They run
--     with the owner's rights and filter rows themselves: nothing is returned
--     unless the caller is a forum member, and the author of an anonymous post
--     is masked unless the caller wrote it or is an instructor.
--   * Writes go through SECURITY DEFINER functions (create_thread, create_post,
--     edit_post, delete_post, toggle_vote, moderate_*), which check membership,
--     ownership, locks, the rate limit and the size caps before touching a row.
--     No client can set pinned/answered/locked/hidden or change an author.
--
-- Who is a member: any signed-in user whose e-mail ends in @ucr.edu, plus
-- every e-mail listed in `members`. A `members` row with role 'blocked'
-- removes access even for a @ucr.edu address. Instructors and TAs are
-- `members` rows with role 'instructor' or 'ta'.
-- =============================================================================

-- ---------------------------------------------------------------------------
-- Tables (data-bearing: create only if missing)
-- ---------------------------------------------------------------------------
create table if not exists public.members (
  email      text primary key check (email = lower(email)),
  role       text not null default 'student'
             check (role in ('instructor', 'ta', 'student', 'guest', 'blocked')),
  note       text,
  added_at   timestamptz not null default now()
);

create table if not exists public.profiles (
  id           uuid primary key,              -- = auth.users.id
  email        text not null,
  display_name text not null check (char_length(display_name) between 2 and 60),
  created_at   timestamptz not null default now()
);

create table if not exists public.categories (
  slug              text primary key,
  name              text not null,
  description       text not null default '',
  sort              int  not null default 100,
  force_anonymous   boolean not null default false,  -- students always post anonymously
  private_to_author boolean not null default false   -- readable by instructors + the thread's author only
);

create table if not exists public.threads (
  id            bigint generated always as identity primary key,
  category      text not null references public.categories(slug),
  title         text not null check (char_length(title) between 3 and 200),
  author_id     uuid not null,
  is_anonymous  boolean not null default false,
  is_pinned     boolean not null default false,
  is_answered   boolean not null default false,
  is_locked     boolean not null default false,
  is_hidden     boolean not null default false,
  created_at    timestamptz not null default now(),
  last_activity timestamptz not null default now()
);
create index if not exists threads_cat_idx on public.threads (category, is_pinned desc, last_activity desc);

create table if not exists public.posts (
  id            bigint generated always as identity primary key,
  thread_id     bigint not null references public.threads(id) on delete cascade,
  author_id     uuid not null,
  is_anonymous  boolean not null default false,
  is_opening    boolean not null default false,   -- the thread's first post
  quote_post_id bigint references public.posts(id) on delete set null,
  body          text not null check (char_length(body) between 1 and 20000),
  is_endorsed   boolean not null default false,
  is_hidden     boolean not null default false,
  is_deleted    boolean not null default false,
  created_at    timestamptz not null default now(),
  edited_at     timestamptz
);
create index if not exists posts_thread_idx on public.posts (thread_id, created_at);
create index if not exists posts_author_time_idx on public.posts (author_id, created_at);

create table if not exists public.votes (
  post_id    bigint not null references public.posts(id) on delete cascade,
  user_id    uuid   not null,
  created_at timestamptz not null default now(),
  primary key (post_id, user_id)
);

-- One stable alias per (thread, author), e.g. "Anonymous Heron".
create table if not exists public.thread_aliases (
  thread_id bigint not null references public.threads(id) on delete cascade,
  author_id uuid   not null,
  alias     text   not null,
  primary key (thread_id, author_id),
  unique (thread_id, alias)
);

-- Realtime signal: ids only, never content or authors.
create table if not exists public.activity (
  id         bigint generated always as identity primary key,
  thread_id  bigint not null,
  kind       text   not null,           -- 'thread' | 'post' | 'edit' | 'moderate' | 'vote'
  created_at timestamptz not null default now()
);

-- ---------------------------------------------------------------------------
-- Lock the tables down
-- ---------------------------------------------------------------------------
alter table public.members        enable row level security;
alter table public.profiles       enable row level security;
alter table public.categories     enable row level security;
alter table public.threads        enable row level security;
alter table public.posts          enable row level security;
alter table public.votes          enable row level security;
alter table public.thread_aliases enable row level security;
alter table public.activity       enable row level security;

revoke all on public.members, public.profiles, public.categories, public.threads,
              public.posts, public.votes, public.thread_aliases, public.activity
  from anon, authenticated;

-- activity: members may read (needed for realtime); nobody may write directly.
grant select on public.activity to authenticated;
drop policy if exists activity_read on public.activity;

-- ---------------------------------------------------------------------------
-- Identity helpers
-- ---------------------------------------------------------------------------
create or replace function public.current_email() returns text
language sql stable
set search_path = public, pg_temp
as $$ select lower(coalesce(auth.jwt() ->> 'email', '')) $$;

create or replace function public.is_member() returns boolean
language sql stable security definer
set search_path = public, pg_temp
as $$
  select auth.uid() is not null
     and public.current_email() <> ''
     -- the address must have been verified (magic link), never just typed in
     and exists (select 1 from auth.users u
                 where u.id = auth.uid() and u.email_confirmed_at is not null)
     and not exists (select 1 from public.members m
                     where m.email = public.current_email() and m.role = 'blocked')
     and (public.current_email() like '%@ucr.edu'
          or public.current_email() like '%.ucr.edu'
          or exists (select 1 from public.members m
                     where m.email = public.current_email() and m.role <> 'blocked'))
$$;

create or replace function public.is_instructor() returns boolean
language sql stable security definer
set search_path = public, pg_temp
as $$
  select auth.uid() is not null and exists (
    select 1 from public.members m
    where m.email = public.current_email() and m.role in ('instructor', 'ta'))
$$;

-- Is a given user (by id) an instructor/TA?  Used only for the badge.
create or replace function public.user_is_instructor(uid uuid) returns boolean
language sql stable security definer
set search_path = public, pg_temp
as $$
  select exists (select 1 from public.profiles p join public.members m on m.email = p.email
                 where p.id = uid and m.role in ('instructor', 'ta'))
$$;

create policy activity_read on public.activity for select to authenticated
  using (public.is_member());

-- ---------------------------------------------------------------------------
-- Aliases
-- ---------------------------------------------------------------------------
create or replace function public.alias_for(t_id bigint, a_id uuid) returns text
language plpgsql security definer
set search_path = public, pg_temp
as $$
declare
  animals text[] := array['Heron','Otter','Lynx','Falcon','Badger','Marten','Ibis','Wren','Orca','Puffin',
                          'Gecko','Bison','Crane','Ferret','Kestrel','Lemur','Newt','Osprey','Panda','Quokka',
                          'Raven','Seal','Tapir','Urchin','Vole','Walrus','Yak','Zebu','Egret','Dingo',
                          'Coyote','Beaver','Alpaca','Magpie','Narwhal','Pelican','Sparrow','Tortoise','Wombat','Jackal'];
  n int := array_length(animals, 1);
  start int;
  a text;
  existing text;
  i int;
begin
  select alias into existing from thread_aliases where thread_id = t_id and author_id = a_id;
  if existing is not null then return existing; end if;
  start := abs(hashtext(t_id::text || ':' || a_id::text)) % n;
  for i in 0 .. n - 1 loop
    a := 'Anonymous ' || animals[1 + (start + i) % n];
    begin
      insert into thread_aliases (thread_id, author_id, alias) values (t_id, a_id, a);
      return a;
    exception when unique_violation then
      -- someone else holds this alias in this thread; try the next one
      select alias into existing from thread_aliases where thread_id = t_id and author_id = a_id;
      if existing is not null then return existing; end if;
    end;
  end loop;
  a := 'Anonymous #' || (select count(*) + 1 from thread_aliases where thread_id = t_id);
  insert into thread_aliases (thread_id, author_id, alias) values (t_id, a_id, a);
  return a;
end $$;

-- ---------------------------------------------------------------------------
-- Visibility of a thread to the caller
-- ---------------------------------------------------------------------------
create or replace function public.can_see_thread(t public.threads) returns boolean
language sql stable security definer
set search_path = public, pg_temp
as $$
  select public.is_member()
     and (public.is_instructor()
          or (not t.is_hidden
              and (not (select c.private_to_author from categories c where c.slug = t.category)
                   or t.author_id = auth.uid())))
$$;

-- ---------------------------------------------------------------------------
-- Read views (the only read path for clients)
-- ---------------------------------------------------------------------------
drop view if exists public.posts_view;
drop view if exists public.threads_view;
drop view if exists public.categories_view;

create view public.categories_view with (security_barrier = true) as
  select c.slug, c.name, c.description, c.sort, c.force_anonymous, c.private_to_author
  from public.categories c
  where public.is_member();

create view public.threads_view with (security_barrier = true) as
  select
    t.id, t.category, c.name as category_name, t.title,
    t.is_anonymous,
    case when t.is_anonymous then ta.alias else p.display_name end            as display_name,
    case when t.is_anonymous and public.is_instructor() and not c.force_anonymous
         then p.display_name || ' <' || p.email || '>' end                      as revealed_name,
    (not t.is_anonymous and public.user_is_instructor(t.author_id))            as author_is_instructor,
    (t.author_id = auth.uid())                                                  as is_mine,
    t.is_pinned, t.is_answered, t.is_locked, t.is_hidden,
    t.created_at, t.last_activity,
    case when op.is_hidden and not public.is_instructor() then '' else op.body end as body,
    (select count(*) from public.posts r
      where r.thread_id = t.id and not r.is_opening and not r.is_deleted
        and (not r.is_hidden or public.is_instructor()))                        as reply_count
  from public.threads t
  join public.categories c on c.slug = t.category
  left join public.profiles p on p.id = t.author_id
  left join public.thread_aliases ta on ta.thread_id = t.id and ta.author_id = t.author_id
  left join public.posts op on op.thread_id = t.id and op.is_opening
  where public.can_see_thread(t);

create view public.posts_view with (security_barrier = true) as
  select
    po.id, po.thread_id, po.is_opening, po.quote_post_id,
    case when po.is_deleted then '' else po.body end                            as body,
    po.is_anonymous,
    case when po.is_deleted then null
         when po.is_anonymous then ta.alias else p.display_name end              as display_name,
    case when po.is_anonymous and public.is_instructor() and not c.force_anonymous
         then p.display_name || ' <' || p.email || '>' end                      as revealed_name,
    (not po.is_anonymous and public.user_is_instructor(po.author_id))          as author_is_instructor,
    (po.author_id = auth.uid())                                                 as is_mine,
    po.is_endorsed, po.is_hidden, po.is_deleted,
    po.created_at, po.edited_at,
    (select count(*) from public.votes v where v.post_id = po.id)               as votes,
    exists (select 1 from public.votes v where v.post_id = po.id and v.user_id = auth.uid()) as voted
  from public.posts po
  join public.threads t on t.id = po.thread_id
  join public.categories c on c.slug = t.category
  left join public.profiles p on p.id = po.author_id
  left join public.thread_aliases ta on ta.thread_id = po.thread_id and ta.author_id = po.author_id
  where public.can_see_thread(t)
    and (not po.is_hidden or public.is_instructor());

revoke all on public.categories_view, public.threads_view, public.posts_view from anon, authenticated, public;
grant select on public.categories_view, public.threads_view, public.posts_view to authenticated;

-- ---------------------------------------------------------------------------
-- Session / profile
-- ---------------------------------------------------------------------------
create or replace function public.me() returns json
language sql stable security definer
set search_path = public, pg_temp
as $$
  select json_build_object(
    'email',         public.current_email(),
    'is_member',     public.is_member(),
    'is_instructor', public.is_instructor(),
    'display_name',  (select display_name from profiles where id = auth.uid()))
$$;

create or replace function public.set_display_name(new_name text) returns void
language plpgsql security definer
set search_path = public, pg_temp
as $$
begin
  if not public.is_member() then raise exception 'Not a forum member.'; end if;
  new_name := btrim(regexp_replace(coalesce(new_name, ''), '\s+', ' ', 'g'));
  if char_length(new_name) < 2 or char_length(new_name) > 60 then
    raise exception 'Display name must be 2-60 characters.';
  end if;
  if lower(new_name) like 'anonymous%' then
    raise exception 'Display names cannot start with "Anonymous".';
  end if;
  insert into profiles (id, email, display_name) values (auth.uid(), public.current_email(), new_name)
  on conflict (id) do update set display_name = excluded.display_name, email = excluded.email;
end $$;

-- ---------------------------------------------------------------------------
-- Guards shared by the write functions
-- ---------------------------------------------------------------------------
create or replace function public.guard_writer() returns void
language plpgsql security definer
set search_path = public, pg_temp
as $$
begin
  if not public.is_member() then raise exception 'Not a forum member.'; end if;
  if not exists (select 1 from profiles where id = auth.uid()) then
    raise exception 'Choose a display name first.';
  end if;
  if not public.is_instructor() and
     (select count(*) from posts where author_id = auth.uid()
        and created_at > now() - interval '10 minutes') >= 10 then
    raise exception 'Rate limit: at most 10 posts in 10 minutes. Please wait a little.';
  end if;
end $$;

create or replace function public.clean_body(b text) returns text
language plpgsql immutable
as $$
begin
  b := btrim(coalesce(b, ''));
  if char_length(b) = 0 then raise exception 'The post is empty.'; end if;
  if char_length(b) > 20000 then raise exception 'Posts are limited to 20,000 characters.'; end if;
  return b;
end $$;

create or replace function public.touch(t_id bigint, k text) returns void
language sql security definer
set search_path = public, pg_temp
as $$
  delete from activity where created_at < now() - interval '2 days';
  insert into activity (thread_id, kind) values (t_id, k);
$$;

-- ---------------------------------------------------------------------------
-- Writes
-- ---------------------------------------------------------------------------
create or replace function public.create_thread(cat text, title text, body text, anonymous boolean)
returns bigint
language plpgsql security definer
set search_path = public, pg_temp
as $$
declare
  c categories;
  t_id bigint;
  anon boolean := coalesce(anonymous, false);
begin
  perform public.guard_writer();
  select * into c from categories where slug = cat;
  if not found then raise exception 'Unknown category.'; end if;
  if c.force_anonymous and not public.is_instructor() then anon := true; end if;
  title := btrim(regexp_replace(coalesce(title, ''), '\s+', ' ', 'g'));
  if char_length(title) < 3 or char_length(title) > 200 then
    raise exception 'Titles must be 3-200 characters.';
  end if;
  insert into threads (category, title, author_id, is_anonymous)
    values (cat, title, auth.uid(), anon) returning id into t_id;
  if anon then perform public.alias_for(t_id, auth.uid()); end if;
  insert into posts (thread_id, author_id, is_anonymous, is_opening, body)
    values (t_id, auth.uid(), anon, true, public.clean_body(body));
  perform public.touch(t_id, 'thread');
  return t_id;
end $$;

create or replace function public.create_post(t_id bigint, body text, anonymous boolean, quote_id bigint default null)
returns bigint
language plpgsql security definer
set search_path = public, pg_temp
as $$
declare
  t threads;
  c categories;
  p_id bigint;
  anon boolean := coalesce(anonymous, false);
begin
  perform public.guard_writer();
  select * into t from threads where id = t_id;
  if not found or not public.can_see_thread(t) then raise exception 'Thread not found.'; end if;
  if t.is_locked and not public.is_instructor() then raise exception 'This thread is locked.'; end if;
  select * into c from categories where slug = t.category;
  if c.force_anonymous and not public.is_instructor() then anon := true; end if;
  if quote_id is not null and not exists (select 1 from posts where id = quote_id and thread_id = t_id) then
    quote_id := null;
  end if;
  if anon then perform public.alias_for(t_id, auth.uid()); end if;
  insert into posts (thread_id, author_id, is_anonymous, quote_post_id, body)
    values (t_id, auth.uid(), anon, quote_id, public.clean_body(body)) returning id into p_id;
  update threads set last_activity = now() where id = t_id;
  perform public.touch(t_id, 'post');
  return p_id;
end $$;

create or replace function public.edit_post(p_id bigint, body text, new_title text default null)
returns void
language plpgsql security definer
set search_path = public, pg_temp
as $$
declare
  po posts;
  t threads;
begin
  if not public.is_member() then raise exception 'Not a forum member.'; end if;
  select * into po from posts where id = p_id;
  if not found or po.is_deleted then raise exception 'Post not found.'; end if;
  select * into t from threads where id = po.thread_id;
  if po.author_id <> auth.uid() then raise exception 'You can only edit your own posts.'; end if;
  if (t.is_locked or po.is_hidden) and not public.is_instructor() then
    raise exception 'This post can no longer be edited.';
  end if;
  update posts set body = public.clean_body(edit_post.body), edited_at = now() where id = p_id;
  if po.is_opening and new_title is not null then
    new_title := btrim(regexp_replace(new_title, '\s+', ' ', 'g'));
    if char_length(new_title) < 3 or char_length(new_title) > 200 then
      raise exception 'Titles must be 3-200 characters.';
    end if;
    update threads set title = new_title where id = t.id;
  end if;
  perform public.touch(t.id, 'edit');
end $$;

-- Authors delete their own posts; instructors may delete any.
-- Deleting the opening post of a thread with no replies removes the thread;
-- otherwise the post is blanked ("deleted") so the replies keep their context.
create or replace function public.delete_post(p_id bigint) returns void
language plpgsql security definer
set search_path = public, pg_temp
as $$
declare
  po posts;
  t threads;
begin
  if not public.is_member() then raise exception 'Not a forum member.'; end if;
  select * into po from posts where id = p_id;
  if not found then raise exception 'Post not found.'; end if;
  select * into t from threads where id = po.thread_id;
  if po.author_id <> auth.uid() and not public.is_instructor() then
    raise exception 'You can only delete your own posts.';
  end if;
  if t.is_locked and not public.is_instructor() then raise exception 'This thread is locked.'; end if;
  if po.is_opening and not exists (select 1 from posts where thread_id = t.id and not is_opening and not is_deleted) then
    delete from threads where id = t.id;
  else
    update posts set is_deleted = true, body = '[deleted]', edited_at = now() where id = p_id;
  end if;
  perform public.touch(t.id, 'edit');
end $$;

create or replace function public.toggle_vote(p_id bigint) returns boolean
language plpgsql security definer
set search_path = public, pg_temp
as $$
declare
  po posts;
  t threads;
begin
  if not public.is_member() then raise exception 'Not a forum member.'; end if;
  select * into po from posts where id = p_id;
  if not found or po.is_deleted then raise exception 'Post not found.'; end if;
  select * into t from threads where id = po.thread_id;
  if not public.can_see_thread(t) then raise exception 'Post not found.'; end if;
  if po.author_id = auth.uid() then raise exception 'You cannot vote for your own post.'; end if;
  if exists (select 1 from votes where post_id = p_id and user_id = auth.uid()) then
    delete from votes where post_id = p_id and user_id = auth.uid();
    perform public.touch(t.id, 'vote');
    return false;
  end if;
  insert into votes (post_id, user_id) values (p_id, auth.uid());
  perform public.touch(t.id, 'vote');
  return true;
end $$;

-- Instructor moderation. NULL leaves a flag unchanged.
create or replace function public.moderate_thread(t_id bigint, pinned boolean default null,
    answered boolean default null, locked boolean default null, hidden boolean default null)
returns void
language plpgsql security definer
set search_path = public, pg_temp
as $$
begin
  if not public.is_instructor() then raise exception 'Instructors only.'; end if;
  update threads set
    is_pinned   = coalesce(pinned,   is_pinned),
    is_answered = coalesce(answered, is_answered),
    is_locked   = coalesce(locked,   is_locked),
    is_hidden   = coalesce(hidden,   is_hidden)
  where id = t_id;
  perform public.touch(t_id, 'moderate');
end $$;

create or replace function public.moderate_post(p_id bigint, endorsed boolean default null,
    hidden boolean default null)
returns void
language plpgsql security definer
set search_path = public, pg_temp
as $$
declare t_id bigint;
begin
  if not public.is_instructor() then raise exception 'Instructors only.'; end if;
  update posts set
    is_endorsed = coalesce(endorsed, is_endorsed),
    is_hidden   = coalesce(hidden,   is_hidden)
  where id = p_id returning thread_id into t_id;
  if t_id is not null then perform public.touch(t_id, 'moderate'); end if;
end $$;

-- ---------------------------------------------------------------------------
-- Function privileges: signed-in users only, and only the public API
-- ---------------------------------------------------------------------------
revoke all on function
  public.current_email(), public.is_member(), public.is_instructor(), public.user_is_instructor(uuid),
  public.alias_for(bigint, uuid), public.can_see_thread(public.threads), public.me(),
  public.set_display_name(text), public.guard_writer(), public.clean_body(text), public.touch(bigint, text),
  public.create_thread(text, text, text, boolean), public.create_post(bigint, text, boolean, bigint),
  public.edit_post(bigint, text, text), public.delete_post(bigint), public.toggle_vote(bigint),
  public.moderate_thread(bigint, boolean, boolean, boolean, boolean), public.moderate_post(bigint, boolean, boolean)
  from public, anon, authenticated;

-- The views call these on the caller's behalf, so `authenticated` needs EXECUTE;
-- each only reveals facts about the caller.
grant execute on function public.current_email(), public.is_member(), public.is_instructor(),
  public.user_is_instructor(uuid), public.can_see_thread(public.threads) to authenticated;
-- The client API.
grant execute on function public.me(), public.set_display_name(text),
  public.create_thread(text, text, text, boolean), public.create_post(bigint, text, boolean, bigint),
  public.edit_post(bigint, text, text), public.delete_post(bigint), public.toggle_vote(bigint),
  public.moderate_thread(bigint, boolean, boolean, boolean, boolean), public.moderate_post(bigint, boolean, boolean)
  to authenticated;

-- ---------------------------------------------------------------------------
-- Realtime: broadcast ids-only activity rows
-- ---------------------------------------------------------------------------
do $$
begin
  if exists (select 1 from pg_publication where pubname = 'supabase_realtime')
     and not exists (select 1 from pg_publication_tables
                     where pubname = 'supabase_realtime' and tablename = 'activity') then
    alter publication supabase_realtime add table public.activity;
  end if;
end $$;

-- ---------------------------------------------------------------------------
-- Categories (seed; re-running updates names and flags, keeps posts)
-- ---------------------------------------------------------------------------
insert into public.categories (slug, name, description, sort, force_anonymous, private_to_author) values
  ('general',  'General',             'Anything about the course.',                         10, false, false),
  ('s01',      'Session 1',           'Introduction; the optimal growth model.',            21, false, false),
  ('s02',      'Session 2',           'Incomplete markets, aggregate risk, OLG, policy.',   22, false, false),
  ('s03',      'Session 3',           'Introduction to computation; Python.',               23, false, false),
  ('s04',      'Session 4',           'Numerical methods I.',                               24, false, false),
  ('s05',      'Session 5',           'Perturbation and projection.',                       25, false, false),
  ('s06',      'Session 6',           'Heterogeneous agents I and II.',                     26, false, false),
  ('s07',      'Session 7',           'Machine learning and deep learning.',                27, false, false),
  ('s08',      'Session 8',           'Reinforcement learning; HA at the frontier.',        28, false, false),
  ('s09',      'Session 9',           'Language models and text as data.',                  29, false, false),
  ('s10',      'Session 10',          'Agentic AI; project presentations.',                 30, false, false),
  ('hw1',      'Homework 1',          'Questions about HW1. Ideas, not solution code, before the due date.', 40, false, false),
  ('hw2',      'Homework 2',          'Questions about HW2. Ideas, not solution code, before the due date.', 41, false, false),
  ('pitches',  'Pitches and teams',   'Post your HW1 research pitch; comment on others; form final-project teams (by Nov 5).', 45, false, false),
  ('project',  'Final project',       'Teams, topics, data, proposals.',                    50, false, false),
  ('labs',     'Labs and code',       'Installation, notebooks, errors.',                   60, false, false),
  ('ai-tools', 'AI tools',            'Using AI assistants for research: what works, what fails.', 70, false, false),
  ('feedback', 'Anonymous feedback',  'Feedback to the instructor. Always anonymous; only instructors can read it.', 90, true, true)
on conflict (slug) do update set
  name = excluded.name, description = excluded.description, sort = excluded.sort,
  force_anonymous = excluded.force_anonymous, private_to_author = excluded.private_to_author;

-- ---------------------------------------------------------------------------
-- First instructor (EDIT the e-mail, then run; add TAs the same way with role 'ta')
-- ---------------------------------------------------------------------------
insert into public.members (email, role, note)
values ('INSTRUCTOR-EMAIL@ucr.edu', 'instructor', 'course instructor')
on conflict (email) do update set role = excluded.role;
