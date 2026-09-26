// Course forum configuration. Fill in the two values from your Supabase project:
//   Dashboard -> Project Settings -> API -> "Project URL" and the "anon public" key.
// The anon key is meant to be public; every permission is enforced in the database
// (see forum/schema.sql). Never paste the "service_role" key here.
window.FORUM_CONFIG = {
  supabaseUrl: "https://YOUR-PROJECT-REF.supabase.co",
  supabaseAnonKey: "YOUR-ANON-PUBLIC-KEY",
  // Where the magic link sends people back to; must also be listed in
  // Supabase -> Authentication -> URL Configuration -> Redirect URLs.
  redirectUrl: "https://vonzhg.github.io/Macro-2026Fall/forum/"
};
