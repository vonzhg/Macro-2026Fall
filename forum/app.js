/* ECON 282E course forum — front end.
 *
 * Plain JavaScript, no build step. Talks to Supabase:
 *   reads   : categories_view, threads_view, posts_view   (the database masks anonymous authors)
 *   writes  : rpc create_thread / create_post / edit_post / delete_post / toggle_vote /
 *             moderate_thread / moderate_post / set_display_name / me
 *   realtime: INSERTs on `activity` (ids only) trigger a refetch through the views.
 * Every permission is enforced in forum/schema.sql; nothing here is a security boundary.
 */
(function () {
  "use strict";

  const CFG = window.FORUM_CONFIG || {};
  const root = document.getElementById("forum-app");
  const FEEDBACK = "feedback";

  const state = {
    client: null,
    session: null,
    me: null,            // { email, is_member, is_instructor, display_name }
    categories: [],
    channel: null,
    refreshTimer: null,
    editing: null,       // post id being edited
    quote: null,         // { id, name, excerpt } for the reply box
  };

  // ------------------------------------------------------------------ helpers
  function h(tag, attrs, ...kids) {
    const el = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs || {})) {
      if (v === null || v === undefined || v === false) continue;
      if (k === "class") el.className = v;
      else if (k === "html") el.innerHTML = v;            // only ever given sanitized HTML
      else if (k.startsWith("on")) el.addEventListener(k.slice(2), v);
      else if (v === true) el.setAttribute(k, "");
      else el.setAttribute(k, v);
    }
    for (const kid of kids.flat()) {
      if (kid === null || kid === undefined || kid === false) continue;
      el.appendChild(typeof kid === "string" || typeof kid === "number" ? document.createTextNode(String(kid)) : kid);
    }
    return el;
  }
  function clear(el) { while (el.firstChild) el.removeChild(el.firstChild); }
  function mount(...nodes) { clear(root); nodes.flat().forEach(n => n && root.appendChild(n)); }

  function ago(ts) {
    const s = (Date.now() - new Date(ts).getTime()) / 1000;
    if (s < 60) return "just now";
    if (s < 3600) return Math.floor(s / 60) + " min ago";
    if (s < 86400) return Math.floor(s / 3600) + " h ago";
    if (s < 7 * 86400) return Math.floor(s / 86400) + " d ago";
    return new Date(ts).toLocaleDateString(undefined, { month: "short", day: "numeric", year: "numeric" });
  }
  function errText(e) {
    const m = (e && (e.message || e.error_description || e.msg)) || String(e);
    return m.replace(/^.*?ERROR:\s*/, "");
  }
  function flash(msg, kind) {
    const old = document.querySelector(".forum-flash");
    if (old) old.remove();
    const el = h("div", { class: "forum-flash notice" + (kind === "ok" ? " info" : ""), role: "status" }, msg);
    root.parentNode.insertBefore(el, root);
    setTimeout(() => el.remove(), 6000);
  }
  function catName(slug) {
    const c = state.categories.find(x => x.slug === slug);
    return c ? c.name : slug;
  }

  // ------------------------------------------------------- markdown + math
  // Math is lifted out before Markdown (so `_` and `*` inside it survive), the Markdown
  // HTML is sanitized, and KaTeX output is put back last. Code spans/blocks are left alone.
  function renderBody(src) {
    const math = [];
    const parts = String(src || "").split(/(```[\s\S]*?```|`[^`\n]*`)/g);
    const protectedSrc = parts.map((part, i) => {
      if (i % 2 === 1) return part; // code: untouched
      return part
        .replace(/\$\$([\s\S]+?)\$\$/g, (_, tex) => { math.push([tex, true]); return "@@MATH" + (math.length - 1) + "@@"; })
        .replace(/(^|[^\\$])\$([^\s$](?:[^$\n]*?[^\s\\$])?)\$(?!\d)/g, (_, pre, tex) => { math.push([tex, false]); return pre + "@@MATH" + (math.length - 1) + "@@"; });
    }).join("");
    let html = window.marked ? window.marked.parse(protectedSrc, { breaks: true, gfm: true }) : protectedSrc;
    html = window.DOMPurify ? window.DOMPurify.sanitize(html, { USE_PROFILES: { html: true } }) : "";
    html = html.replace(/@@MATH(\d+)@@/g, (_, i) => {
      const [tex, display] = math[+i];
      if (!window.katex) return (display ? "$$" : "$") + escapeHtml(tex) + (display ? "$$" : "$");
      try { return window.katex.renderToString(tex, { displayMode: display, throwOnError: false, trust: false, strict: "ignore" }); }
      catch (e) { return escapeHtml(tex); }
    });
    return html;
  }
  function escapeHtml(s) {
    return String(s).replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  }

  // ------------------------------------------------------------- routing
  function route() {
    // All navigation state lives in the hash; boot() moves a ?cat= from site links into it.
    const hash = new URLSearchParams(location.hash.replace(/^#/, ""));
    return {
      thread: hash.get("t") ? parseInt(hash.get("t"), 10) : null,
      compose: hash.has("new"),
      cat: hash.get("cat") || "",
      q: hash.get("q") || "",
    };
  }
  function go(params) {
    const p = new URLSearchParams();
    for (const [k, v] of Object.entries(params)) if (v !== null && v !== undefined && v !== "") p.set(k, v === true ? "" : v);
    const s = p.toString().replace(/=(&|$)/g, "$1");
    if (("#" + s) === location.hash || (!s && !location.hash)) render();
    else location.hash = s;
  }

  // ------------------------------------------------------------- boot
  async function boot() {
    if (!CFG.supabaseUrl || /YOUR-PROJECT/.test(CFG.supabaseUrl) || !CFG.supabaseAnonKey || /YOUR-ANON/.test(CFG.supabaseAnonKey)) {
      mount(h("div", { class: "notice info forum-soon" },
        h("strong", {}, "The forum opens soon. "),
        "The instructor is setting it up. Until then, bring questions to class or send an e-mail."));
      return;
    }
    if (!window.supabase || !window.supabase.createClient) {
      mount(h("div", { class: "notice" }, "Could not load the forum library. Check your connection, or try another browser."));
      return;
    }
    state.client = window.supabase.createClient(CFG.supabaseUrl, CFG.supabaseAnonKey, {
      auth: { flowType: "pkce", detectSessionInUrl: true, persistSession: true, autoRefreshToken: true },
    });
    // Supabase advises against calling the client from inside this callback; defer.
    state.client.auth.onAuthStateChange((event, session) => {
      const had = !!state.session;
      state.session = session;
      if (event === "SIGNED_OUT" || (!had && session)) setTimeout(start, 0);
    });
    const { data } = await state.client.auth.getSession();
    state.session = data.session;
    // Tidy the URL: drop the ?code=… left by the magic link, and turn a ?cat=s03 from a
    // "Discuss" link on the site into #cat=s03, where the router keeps its state.
    const qs = new URLSearchParams(location.search);
    if (qs.has("code") || qs.has("cat")) {
      const cat = qs.get("cat");
      const hash = location.hash || (cat ? "#cat=" + encodeURIComponent(cat) : "");
      history.replaceState(null, "", location.pathname + hash);
    }
    window.addEventListener("hashchange", render);
    start();
  }

  let starting = false;
  async function start() {
    if (starting) return;
    starting = true;
    try {
      if (!state.session) { state.me = null; unsubscribe(); renderSignIn(); return; }
      const { data: me, error } = await state.client.rpc("me");
      if (error) throw error;
      state.me = me;
      if (!me.is_member) { renderNotMember(); return; }
      if (!me.display_name) { renderProfile(true); return; }
      const cats = await state.client.from("categories_view").select("*").order("sort");
      if (cats.error) throw cats.error;
      state.categories = cats.data || [];
      subscribe();
      render();
    } catch (e) {
      mount(h("div", { class: "notice" }, "Something went wrong: " + errText(e)));
    } finally { starting = false; }
  }

  // ------------------------------------------------------------- realtime
  function subscribe() {
    if (state.channel) return;
    state.channel = state.client.channel("forum-activity")
      .on("postgres_changes", { event: "INSERT", schema: "public", table: "activity" }, payload => {
        const r = route();
        const tid = payload.new && payload.new.thread_id;
        if (r.compose || state.editing) return;                    // never clobber a form in use
        if (r.thread && tid !== r.thread) return;
        clearTimeout(state.refreshTimer);
        state.refreshTimer = setTimeout(() => render({ quiet: true }), 600);
      })
      .subscribe();
  }
  function unsubscribe() {
    if (state.channel) { state.client.removeChannel(state.channel); state.channel = null; }
  }

  // ------------------------------------------------------------- chrome
  function userBar() {
    const me = state.me || {};
    return h("div", { class: "forum-userbar" },
      h("span", {}, "Signed in as ", h("strong", {}, me.display_name || me.email || ""),
        me.is_instructor ? h("span", { class: "forum-badge forum-badge-instr" }, "Instructor") : null),
      h("span", { class: "forum-userbar-actions" },
        h("a", { href: "#", onclick: e => { e.preventDefault(); renderProfile(false); } }, "Change name"),
        h("a", { href: "#", onclick: async e => { e.preventDefault(); await state.client.auth.signOut(); } }, "Sign out")));
  }

  // ------------------------------------------------------------- sign in
  function renderSignIn(sentTo) {
    const email = h("input", { type: "email", required: true, autocomplete: "email", placeholder: "you@ucr.edu", value: sentTo || "" });
    const code = h("input", { type: "text", inputmode: "numeric", autocomplete: "one-time-code", placeholder: "6-digit code", maxlength: "10" });
    const status = h("p", { class: "forum-muted" });
    const sendForm = h("form", { class: "forum-form", onsubmit: async e => {
        e.preventDefault();
        const addr = email.value.trim().toLowerCase();
        status.textContent = "Sending…";
        const { error } = await state.client.auth.signInWithOtp({ email: addr, options: { emailRedirectTo: CFG.redirectUrl || location.href.split("#")[0] } });
        if (error) { status.textContent = errText(error); return; }
        renderSignIn(addr);
      } },
      h("label", {}, "Your UCR e-mail", email),
      h("button", { class: "btn", type: "submit" }, sentTo ? "Send again" : "Send me a sign-in link"),
      status);
    const codeForm = sentTo ? h("form", { class: "forum-form", onsubmit: async e => {
        e.preventDefault();
        status.textContent = "Checking…";
        const { error } = await state.client.auth.verifyOtp({ email: sentTo, token: code.value.trim(), type: "email" });
        if (error) status.textContent = errText(error);
      } },
      h("p", {}, "We sent a message to ", h("strong", {}, sentTo), ". Click the link in it ",
        h("em", {}, "in this browser"), ", or type the code from the message here:"),
      h("label", {}, "Code", code),
      h("button", { class: "btn btn-ghost", type: "submit" }, "Sign in with the code")) : null;
    mount(h("div", { class: "forum-panel" },
      h("h2", { class: "forum-h2" }, "Sign in"),
      h("p", {}, "The forum is open to course participants. Use your ", h("code", {}, "@ucr.edu"),
        " address; no password is needed. Instructors and invited guests use the address the instructor registered."),
      sentTo ? codeForm : sendForm,
      sentTo ? h("p", { class: "forum-muted" }, "Wrong address? ", h("a", { href: "#", onclick: e => { e.preventDefault(); renderSignIn(); } }, "Start again"), ".") : null));
  }

  function renderNotMember() {
    unsubscribe();
    mount(h("div", { class: "forum-panel" },
      h("h2", { class: "forum-h2" }, "This forum is for course participants"),
      h("p", {}, "You are signed in as ", h("strong", {}, state.me.email), ", which is not a ",
        h("code", {}, "@ucr.edu"), " address on the course list. Sign in with your UCR e-mail, or ask the instructor to add this address."),
      h("button", { class: "btn btn-ghost", onclick: () => state.client.auth.signOut() }, "Sign out")));
  }

  function renderProfile(first) {
    const name = h("input", { type: "text", required: true, minlength: "2", maxlength: "60", value: (state.me && state.me.display_name) || "" });
    const status = h("p", { class: "forum-muted" });
    mount(h("div", { class: "forum-panel" },
      h("h2", { class: "forum-h2" }, first ? "Welcome — choose a display name" : "Change your display name"),
      h("p", {}, "This name appears on your ", h("strong", {}, "named"), " posts. Your anonymous posts show an alias instead. Your real name is the usual choice."),
      h("form", { class: "forum-form", onsubmit: async e => {
          e.preventDefault();
          const { error } = await state.client.rpc("set_display_name", { new_name: name.value });
          if (error) { status.textContent = errText(error); return; }
          state.me.display_name = name.value.trim();
          start();
        } },
        h("label", {}, "Display name", name),
        h("button", { class: "btn", type: "submit" }, "Save"),
        first ? null : h("button", { class: "btn btn-ghost", type: "button", onclick: () => render() }, "Cancel"),
        status)));
  }

  // ------------------------------------------------------------- views
  async function render(opts) {
    if (!state.me || !state.me.is_member || !state.me.display_name) return;
    const r = route();
    try {
      if (r.compose) return renderCompose(r);
      if (r.thread) return await renderThread(r.thread, opts);
      return await renderBoard(r, opts);
    } catch (e) {
      mount(userBar(), h("div", { class: "notice" }, "Could not load: " + errText(e)));
    }
  }

  function categorySelect(value, onchange, includeAll) {
    const sel = h("select", { onchange });
    if (includeAll) sel.appendChild(h("option", { value: "" }, "All categories"));
    for (const c of state.categories) sel.appendChild(h("option", { value: c.slug, selected: c.slug === value }, c.name));
    sel.value = value || "";
    return sel;
  }

  function authorLine(x) {
    return h("span", { class: "forum-author" },
      h("strong", {}, x.display_name || (x.is_deleted ? "—" : "Unknown")),
      x.author_is_instructor ? h("span", { class: "forum-badge forum-badge-instr" }, "Instructor") : null,
      x.revealed_name ? h("span", { class: "forum-revealed", title: "Shown to instructors only" }, "(" + x.revealed_name + " — visible to instructors only)") : null);
  }

  async function renderBoard(r, opts) {
    let q = state.client.from("threads_view")
      .select("id,category,category_name,title,display_name,revealed_name,author_is_instructor,is_anonymous,is_pinned,is_answered,is_locked,is_hidden,created_at,last_activity,reply_count")
      .order("is_pinned", { ascending: false })
      .order("last_activity", { ascending: false })
      .limit(200);
    if (r.cat) q = q.eq("category", r.cat);
    const term = (r.q || "").replace(/[,()*%\\:"'.]/g, " ").trim();
    if (term) q = q.or(`title.ilike.*${term}*,body.ilike.*${term}*`);
    const { data, error } = await q;
    if (error) throw error;

    const search = h("input", { type: "search", placeholder: "Search titles and posts", value: r.q || "" });
    const toolbar = h("form", { class: "forum-toolbar", onsubmit: e => { e.preventDefault(); go({ cat: r.cat, q: search.value.trim() }); } },
      categorySelect(r.cat, e => go({ cat: e.target.value, q: r.q }), true),
      search,
      h("button", { class: "btn btn-ghost", type: "submit" }, "Search"),
      h("a", { class: "btn", href: "#" + new URLSearchParams(Object.assign({ new: "" }, r.cat ? { cat: r.cat } : {})).toString().replace("new=", "new") }, "New thread"));

    const catInfo = r.cat ? state.categories.find(c => c.slug === r.cat) : null;
    const list = h("div", { class: "forum-list" });
    if (!data.length) list.appendChild(h("p", { class: "forum-muted" }, term ? "No threads match." : "No threads here yet. Start one."));
    for (const t of data) {
      list.appendChild(h("a", { class: "forum-row" + (t.is_pinned ? " is-pinned" : "") + (t.is_hidden ? " is-hidden" : ""), href: "#t=" + t.id },
        h("span", { class: "forum-row-main" },
          h("span", { class: "forum-row-title" },
            t.is_pinned ? h("span", { class: "forum-badge forum-badge-pin" }, "Pinned") : null,
            t.is_answered ? h("span", { class: "forum-badge forum-badge-ok" }, "Answered") : null,
            t.is_locked ? h("span", { class: "forum-badge forum-badge-lock" }, "Locked") : null,
            t.is_hidden ? h("span", { class: "forum-badge forum-badge-lock" }, "Hidden") : null,
            t.title),
          h("span", { class: "forum-row-meta" },
            r.cat ? null : h("span", { class: "forum-chip" }, t.category_name),
            authorLine(t), " · ", ago(t.last_activity))),
        h("span", { class: "forum-row-count", title: "replies" }, String(t.reply_count), h("small", {}, t.reply_count == 1 ? "reply" : "replies"))));
    }
    mount(userBar(), toolbar,
      catInfo ? h("p", { class: "forum-catinfo" }, h("strong", {}, catInfo.name), " — ", catInfo.description) : null,
      list);
  }

  function anonControl(cat, checked) {
    const forced = cat === FEEDBACK && !state.me.is_instructor;
    const box = h("input", { type: "checkbox", checked: forced || checked, disabled: forced });
    const note = h("span", { class: "forum-muted forum-anon-note" });
    const update = (c) => {
      const isFb = (typeof c === "string" ? c : cat) === FEEDBACK && !state.me.is_instructor;
      note.textContent = isFb
        ? "Feedback is always anonymous: no one sees your name, instructors included."
        : "Classmates will see an alias such as “Anonymous Heron”. Instructors can see who wrote it.";
    };
    update(cat);
    const wrap = h("label", { class: "forum-anon" }, box, " Post anonymously ", note);
    wrap.checkbox = box;
    wrap.setCategory = c => {
      const f = c === FEEDBACK && !state.me.is_instructor;
      box.disabled = f; if (f) box.checked = true; update(c);
    };
    return wrap;
  }

  function editor(initial, placeholder) {
    const ta = h("textarea", { rows: "8", maxlength: "20000", placeholder: placeholder || "Write in Markdown; math with $…$ or $$…$$." }, initial || "");
    const preview = h("div", { class: "forum-body forum-preview", hidden: true });
    const toggle = h("button", { type: "button", class: "forum-linkbtn", onclick: () => {
        const showing = !preview.hidden;
        if (showing) { preview.hidden = true; ta.hidden = false; toggle.textContent = "Preview"; }
        else { preview.innerHTML = renderBody(ta.value) || "<p class='forum-muted'>Nothing to preview.</p>"; preview.hidden = false; ta.hidden = true; toggle.textContent = "Edit"; }
      } }, "Preview");
    const wrap = h("div", { class: "forum-editor" }, h("div", { class: "forum-editor-bar" }, toggle), ta, preview);
    wrap.textarea = ta;
    return wrap;
  }

  function renderCompose(r) {
    const title = h("input", { type: "text", required: true, minlength: "3", maxlength: "200", placeholder: "A specific title helps others find it" });
    const ed = editor("", "What is your question? Say what you tried and where it breaks.");
    const cat = r.cat && state.categories.some(c => c.slug === r.cat) ? r.cat : "general";
    const anon = anonControl(cat, false);
    const sel = categorySelect(cat, e => anon.setCategory(e.target.value), false);
    const status = h("p", { class: "forum-muted" });
    const submit = h("button", { class: "btn", type: "submit" }, "Post thread");
    mount(userBar(), h("div", { class: "forum-panel" },
      h("h2", { class: "forum-h2" }, "New thread"),
      h("form", { class: "forum-form", onsubmit: async e => {
          e.preventDefault();
          submit.disabled = true; status.textContent = "Posting…";
          const { data, error } = await state.client.rpc("create_thread", {
            cat: sel.value, title: title.value, body: ed.textarea.value, anonymous: anon.checkbox.checked });
          submit.disabled = false;
          if (error) { status.textContent = errText(error); return; }
          go({ t: data });
        } },
        h("label", {}, "Category", sel),
        h("label", {}, "Title", title),
        ed, anon,
        h("div", { class: "forum-actions" }, submit,
          h("a", { class: "btn btn-ghost", href: "#" + (r.cat ? "cat=" + r.cat : "") }, "Cancel")),
        status)));
    title.focus();
  }

  async function renderThread(id, opts) {
    const [tr, pr] = await Promise.all([
      state.client.from("threads_view").select("*").eq("id", id).maybeSingle(),
      state.client.from("posts_view").select("*").eq("thread_id", id).order("created_at", { ascending: true }),
    ]);
    if (tr.error) throw tr.error;
    if (pr.error) throw pr.error;
    const t = tr.data;
    if (!t) { mount(userBar(), h("div", { class: "notice" }, "This thread does not exist or is not visible to you. ", h("a", { href: "#" }, "Back to the board"))); return; }
    const posts = pr.data || [];
    const byId = Object.fromEntries(posts.map(p => [p.id, p]));
    const instr = state.me.is_instructor;

    // Keep an unsent reply across realtime refreshes.
    const oldReply = document.querySelector(".forum-reply textarea");
    const draft = opts && opts.quiet && oldReply ? oldReply.value : "";

    const head = h("div", { class: "forum-thread-head" },
      h("a", { href: "#" + (t.category ? "cat=" + t.category : "") }, "← ", t.category_name),
      h("h2", { class: "forum-h2" },
        t.is_pinned ? h("span", { class: "forum-badge forum-badge-pin" }, "Pinned") : null,
        t.is_answered ? h("span", { class: "forum-badge forum-badge-ok" }, "Answered") : null,
        t.is_locked ? h("span", { class: "forum-badge forum-badge-lock" }, "Locked") : null,
        t.is_hidden ? h("span", { class: "forum-badge forum-badge-lock" }, "Hidden") : null,
        t.title),
      instr ? h("div", { class: "forum-modbar" },
        h("span", { class: "forum-muted" }, "Instructor: "),
        modBtn(t.is_pinned ? "Unpin" : "Pin", { pinned: !t.is_pinned }),
        modBtn(t.is_answered ? "Mark unanswered" : "Mark answered", { answered: !t.is_answered }),
        modBtn(t.is_locked ? "Unlock" : "Lock", { locked: !t.is_locked }),
        modBtn(t.is_hidden ? "Unhide thread" : "Hide thread", { hidden: !t.is_hidden })) : null);

    function modBtn(label, flags) {
      return h("button", { class: "forum-linkbtn", onclick: async () => {
          const { error } = await state.client.rpc("moderate_thread", Object.assign({ t_id: t.id }, flags));
          if (error) flash(errText(error)); else render();
        } }, label);
    }

    const list = h("div", { class: "forum-posts" }, posts.map(p => postCard(p, t, byId, instr)));

    let reply = null;
    if (!t.is_locked || instr) {
      const ed = editor(draft, "Write a reply…");
      const anon = anonControl(t.category, false);
      const status = h("p", { class: "forum-muted" });
      const quoteBox = h("div", { class: "forum-quote-pending" });
      const showQuote = () => {
        clear(quoteBox);
        if (state.quote) quoteBox.appendChild(h("div", { class: "forum-quote" },
          "Replying to ", h("strong", {}, state.quote.name), ": ", state.quote.excerpt, " ",
          h("button", { type: "button", class: "forum-linkbtn", onclick: () => { state.quote = null; showQuote(); } }, "remove")));
      };
      showQuote();
      const submit = h("button", { class: "btn", type: "submit" }, "Post reply");
      reply = h("form", { class: "forum-form forum-reply", onsubmit: async e => {
          e.preventDefault();
          submit.disabled = true; status.textContent = "Posting…";
          const { error } = await state.client.rpc("create_post", {
            t_id: t.id, body: ed.textarea.value, anonymous: anon.checkbox.checked, quote_id: state.quote ? state.quote.id : null });
          submit.disabled = false;
          if (error) { status.textContent = errText(error); return; }
          state.quote = null;
          ed.textarea.value = "";
          render();
        } },
        h("h3", { class: "forum-h3" }, "Reply"), quoteBox, ed, anon, h("div", { class: "forum-actions" }, submit), status);
      reply.focusQuote = showQuote;
    } else {
      reply = h("p", { class: "notice info" }, "This thread is locked.");
    }
    state.replyForm = reply;
    mount(userBar(), head, list, reply);
  }

  function postCard(p, t, byId, instr) {
    const canEdit = p.is_mine && !p.is_deleted && (!t.is_locked || instr) && (!p.is_hidden || instr);
    const canDelete = !p.is_deleted && ((p.is_mine && (!t.is_locked || instr)) || instr);
    const body = h("div", { class: "forum-body" });
    if (p.is_deleted) body.appendChild(h("p", { class: "forum-muted" }, "[deleted]"));
    else body.innerHTML = renderBody(p.body);

    const quoted = p.quote_post_id && byId[p.quote_post_id];
    const card = h("article", { class: "forum-post" + (p.is_opening ? " is-opening" : "") + (p.is_endorsed ? " is-endorsed" : "") + (p.is_hidden ? " is-hidden" : ""), id: "p" + p.id },
      h("header", { class: "forum-post-head" },
        authorLine(p),
        h("span", { class: "forum-muted" }, " · ", ago(p.created_at), p.edited_at && !p.is_deleted ? " · edited" : ""),
        p.is_endorsed ? h("span", { class: "forum-badge forum-badge-ok" }, "Endorsed by instructor") : null,
        p.is_hidden ? h("span", { class: "forum-badge forum-badge-lock" }, "Hidden") : null),
      quoted ? h("a", { class: "forum-quote", href: "#t=" + t.id, onclick: e => { e.preventDefault(); const el = document.getElementById("p" + quoted.id); if (el) el.scrollIntoView({ behavior: "smooth" }); } },
        "Replying to ", h("strong", {}, quoted.display_name || "—"), ": ", excerpt(quoted.body)) : null,
      body);

    if (!p.is_deleted) {
      const actions = h("footer", { class: "forum-post-actions" });
      actions.appendChild(h("button", { class: "forum-vote" + (p.voted ? " is-on" : ""), disabled: p.is_mine, title: p.is_mine ? "Your own post" : "Mark as helpful",
        onclick: async () => { const { error } = await state.client.rpc("toggle_vote", { p_id: p.id }); if (error) flash(errText(error)); else render({ quiet: true }); } },
        "▲ helpful ", h("span", {}, String(p.votes))));
      if (!t.is_locked || instr) actions.appendChild(h("button", { class: "forum-linkbtn", onclick: () => {
          state.quote = { id: p.id, name: p.display_name || "—", excerpt: excerpt(p.body) };
          if (state.replyForm && state.replyForm.focusQuote) { state.replyForm.focusQuote(); state.replyForm.scrollIntoView({ behavior: "smooth" }); state.replyForm.querySelector("textarea").focus(); }
        } }, "Quote"));
      if (canEdit) actions.appendChild(h("button", { class: "forum-linkbtn", onclick: () => startEdit(card, body, p, t) }, "Edit"));
      if (canDelete) actions.appendChild(h("button", { class: "forum-linkbtn", onclick: async () => {
          if (!confirm(p.is_opening ? "Delete this thread's opening post? If nobody has replied, the whole thread goes." : "Delete this post?")) return;
          const { error } = await state.client.rpc("delete_post", { p_id: p.id });
          if (error) flash(errText(error)); else render();
        } }, "Delete"));
      if (instr) {
        actions.appendChild(h("button", { class: "forum-linkbtn", onclick: async () => {
            const { error } = await state.client.rpc("moderate_post", { p_id: p.id, endorsed: !p.is_endorsed });
            if (error) flash(errText(error)); else render();
          } }, p.is_endorsed ? "Un-endorse" : "Endorse"));
        actions.appendChild(h("button", { class: "forum-linkbtn", onclick: async () => {
            const { error } = await state.client.rpc("moderate_post", { p_id: p.id, hidden: !p.is_hidden });
            if (error) flash(errText(error)); else render();
          } }, p.is_hidden ? "Unhide" : "Hide"));
      }
      card.appendChild(actions);
    }
    return card;
  }

  function excerpt(s) {
    const t = String(s || "").replace(/```[\s\S]*?```/g, "[code]").replace(/\s+/g, " ").trim();
    return t.length > 120 ? t.slice(0, 117) + "…" : t;
  }

  function startEdit(card, body, p, t) {
    state.editing = p.id;
    const ed = editor(p.body);
    const title = p.is_opening ? h("input", { type: "text", value: t.title, maxlength: "200" }) : null;
    const status = h("p", { class: "forum-muted" });
    const form = h("form", { class: "forum-form", onsubmit: async e => {
        e.preventDefault();
        const args = { p_id: p.id, body: ed.textarea.value, new_title: title ? title.value : null };
        const { error } = await state.client.rpc("edit_post", args);
        if (error) { status.textContent = errText(error); return; }
        state.editing = null;
        render();
      } },
      title ? h("label", {}, "Title", title) : null, ed,
      h("div", { class: "forum-actions" },
        h("button", { class: "btn", type: "submit" }, "Save"),
        h("button", { class: "btn btn-ghost", type: "button", onclick: () => { state.editing = null; render(); } }, "Cancel")),
      status);
    body.replaceWith(form);
    const actions = card.querySelector(".forum-post-actions");
    if (actions) actions.remove();
  }

  // Exposed for tests only.
  window.__forum = { renderBody, state, route };

  boot();
})();
