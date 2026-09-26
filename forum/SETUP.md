# Course forum — one-time setup

The forum is a static page (`forum/index.html` + `app.js`) on the course's GitHub Pages site. Its
data lives in a free **Supabase** project: a Postgres database with e-mail sign-in. Until you finish
step 5, the page shows "The forum opens soon".

Allow about 10 minutes.

## 1. Create the project
1. Go to <https://supabase.com>, sign in (a GitHub login works), and click **New project**.
2. Name it `econ282e-forum`, choose a strong database password (store it; you rarely need it), and pick
   the region **West US**. The free plan is enough.
3. Wait a minute or two while it provisions.

## 2. Create the tables and permissions
1. Open `forum/schema.sql` and **change the e-mail on its last `insert`** from the placeholder `INSTRUCTOR-EMAIL@ucr.edu`
   to the address you will sign in with.
2. In the dashboard: **SQL Editor → New query**. Paste all of `schema.sql` and click **Run**. It should
   finish with "Success. No rows returned".

The script is safe to re-run later, for example after an update. It never deletes posts.

## 3. Sign-in settings
In **Authentication**:

- **Sign In / Providers → Email:** enabled. Keep **Confirm email** on. You may switch off
  password sign-ups: the forum only uses e-mail links and codes, and it admits an address only after
  it has been verified.
- **URL Configuration:**
  - **Site URL:** `https://vonzhg.github.io/Macro-2026Fall/forum/`
  - **Redirect URLs:** add `https://vonzhg.github.io/Macro-2026Fall/forum/`
    (and `http://localhost:8000/forum/` if you want to test locally).
- **Emails → Magic Link** template: add the one-time code so students who open the e-mail on another
  device can type it in. Replace the body with, for example:

  ```html
  <h2>Sign in to the ECON 282E forum</h2>
  <p><a href="{{ .ConfirmationURL }}">Sign in</a> (open it in the browser where you asked for it), or enter this code on the forum page: <strong>{{ .Token }}</strong></p>
  ```

  Do the same for the **Confirm signup** template, which is what a first-time user receives.
- **Rate limits:** Supabase's built-in e-mail sender allows only a few messages per hour. Before the
  class signs in, set up custom SMTP under **Authentication → Emails → SMTP Settings**. A UCR mail
  relay or any transactional provider (Resend, SendGrid, Postmark) works.

## 4. Make yourself an instructor
Step 2 already added one instructor. To add a **TA** or a guest, run this in the SQL Editor:

```sql
insert into public.members (email, role, note) values
  ('ta.name@ucr.edu', 'ta', 'TA, Fall 2026')          -- sees who wrote anonymous posts; can moderate
on conflict (email) do update set role = excluded.role;

-- a guest with a non-UCR address (reads and posts like a student):
insert into public.members (email, role) values ('visitor@example.edu', 'guest');

-- remove someone's access, even with a @ucr.edu address:
insert into public.members (email, role) values ('someone@ucr.edu', 'blocked')
on conflict (email) do update set role = 'blocked';
```

Roles: `instructor` and `ta` see authorship of anonymous posts (except in *Anonymous feedback*) and
can pin, mark answered, lock, hide and endorse. `student` and `guest` are ordinary members. `blocked`
has no access. Any verified `@ucr.edu` address is a member without being listed.

## 5. Connect the page
1. **Project Settings → API**: copy the **Project URL** and the **anon public** key.
2. Put them in `forum/config.js` (`supabaseUrl`, `supabaseAnonKey`), commit, and publish the page with
   the rest of the site.

The anon key is designed to be public: the database refuses anything the signed-in user is not allowed
to do. **Never** put the `service_role` key in the page.

## 6. Check it (5 minutes, two accounts)
1. Open the forum, sign in with your instructor address, and choose a display name.
2. In a private window, sign in as a student (`@ucr.edu`), post an anonymous thread in *Session 2*,
   and a named reply.
3. Back in the instructor window: the thread shows "Anonymous …" and, in brown, the student's name
   marked *visible to instructors only*. Pin it, mark it answered, and endorse the reply. The student
   window updates within a second or two.
4. As the student, post in *Anonymous feedback*. As the instructor, check that you can read it but see
   no name.

## Running it through the term
- **Paused project.** Free projects pause after about a week with no activity; the page then fails to
  load posts. In the dashboard, open the project and click **Restore**. Weekly class use normally
  keeps it awake.
- **Export posts** (for your records at the end of term): SQL Editor →
  `select t.title, c.name as category, p.created_at, p.is_anonymous, pr.display_name, pr.email, p.body
   from posts p join threads t on t.id = p.thread_id join categories c on c.slug = t.category
   left join profiles pr on pr.id = p.author_id order by p.created_at;`
  then **Download CSV**. The export contains authorship, including for anonymous posts: treat it as a
  student record.
- **Remove a post for good:** hiding it in the forum is usually enough. To delete it:
  `delete from posts where id = …;` (a thread: `delete from threads where id = …;`).
- **Categories** are rows in `categories`. Edit the seed at the end of `schema.sql` and re-run it, or
  insert a row by hand.

## What the database guarantees
The rules below are enforced by the database, not the page; the automated checks run against the
schema cover each one.

- A visitor who is not signed in, or whose address is neither a verified `@ucr.edu` address nor listed
  in `members`, reads **nothing** and can post nothing.
- Nobody can read or write the tables directly. Reads go through views that replace the author of an
  anonymous post with its alias; only the author, and instructors outside *Anonymous feedback*, see the
  name. No view exposes an author id.
- Only instructors can pin, lock, hide or endorse. Authors can edit or delete only their own posts, and
  not in a locked thread.
- Posts are limited to 20,000 characters, and to 10 posts per person per 10 minutes.
- The live-update channel carries only thread ids, never text or authors.
