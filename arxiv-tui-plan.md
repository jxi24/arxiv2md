# arxiv-tui Implementation Plan

## Context

This plan targets the `claude/add-claude-documentation-sDdzH` branch of
`jxi24/arxiv-tui`. That branch already implements TF-IDF + 2-layer MLP ranking
(`Ranker`), project notes, sub-projects, and Markdown/JSON/BibTeX export on top
of the `main` branch's SQLite persistence, bookmark system, PDF download, and
FTXUI TUI.

The tasks below are ordered by dependency. Complete them in sequence. Each task
includes the files to touch, the exact change required, and the test obligation.

---

## Task 1 — Fix SQL parameterization in DatabaseManager

**Why first:** every subsequent task that writes to the database inherits this
bug if it isn't fixed now. The current code constructs all write queries via
`fmt::format` with a hand-rolled `EscapeString`, which mishandles Unicode in
author names and abstracts and is a correctness risk.

**Files:** `src/Arxiv/DatabaseManager.cc`, `include/Arxiv/DatabaseManager.hh`

**Changes:**

Replace every `ExecuteSQL(fmt::format(...))` write call with a
`sqlite3_prepare_v2` / `sqlite3_bind_*` / `sqlite3_step` / `sqlite3_finalize`
sequence. Affected methods:

- `AddArticle` — bind `link`, `title`, `authors`, `abstract`, `date`,
  `bookmarked`, `relevance_score` (add this column, see Task 3)
- `ToggleBookmark`
- `AddProject`
- `RemoveProject` (two statements inside a transaction — keep the transaction,
  use prepared statements for each)
- `LinkArticleToProject`
- `UnlinkArticleFromProject`
- `SetProjectNote`
- `SetRating`

Read queries already use `sqlite3_prepare_v2`; leave them as-is.

Delete `EscapeString` and its declaration once no callers remain.

**Tests:** The existing `DatabaseManager` mock-based tests do not exercise the
real SQL. Add a new `test/DatabaseManagerIntegration.cc` fixture that opens an
in-memory database (`":memory:"`), calls each write method with inputs
containing single-quotes, curly braces, and multi-byte UTF-8 (e.g.
`"Ré́mi Hébert"`, `"$\\hat{p}_T$"`), then reads back and asserts round-trip
equality.

---

## Task 2 — Apply LaTeX normalization before ranking

**Why:** `Ranker::Tokenise` currently receives raw article text. The
`Fetcher::StyleLatex` and `Fetcher::ReplaceLatexAccents` methods exist for
display purposes but are not called on the text path that feeds the ranker,
so tokens like `\\hat`, `\\textit{`, and `\\'e` pollute the vocabulary.

**Files:** `src/Arxiv/Ranker.cc`, `include/Arxiv/Ranker.hh`,
`include/Arxiv/Fetcher.hh`

**Changes:**

1. Move `StyleLatex` and `ReplaceLatexAccents` out of `Fetcher` and into a
   free-function header `include/Arxiv/LatexUtils.hh` so both `Fetcher` and
   `Ranker` can call them without a circular dependency.
2. At the top of `Ranker::Tokenise`, call `ReplaceLatexAccents` then
   `StyleLatex` on the input string before tokenizing.
3. Also strip inline math: add a pass that removes `$...$` and `\\(...\\)`
   spans before tokenizing (simple regex: remove everything between `$` pairs
   and between `\(` / `\)` pairs).
4. Update `Fetcher` to call the same functions from the new header.

**Tests:** Add `TEST_CASE("Ranker tokenises LaTeX cleanly")` that calls
`Tokenise` on a string like `"The $\\hat{p}_T$ distribution in Drell-Yan"`
and asserts that the returned tokens do not contain `$`, `\\hat`, or `{`.

---

## Task 3 — Keyword-based cold-start ranking

**Why:** The MLP ranker requires `MIN_TRAIN = 3` user ratings before it
produces any scores. New users see unranked articles. arxiv2md solves this
with a keyword file and TF-IDF cosine similarity that works immediately.
The goal is to use keyword similarity as the score when the MLP is untrained,
and blend the two as ratings accumulate.

**New file:** `arxiv_keywords.txt` in the repo root — one keyword or phrase
per line, same format as `jxi24/arxiv2md`. Start with the physics/ML example
from that repo as a default.

**Files:** `include/Arxiv/Config.hh`, `src/Arxiv/Config.cc`,
`include/Arxiv/Ranker.hh`, `src/Arxiv/Ranker.cc`,
`include/Arxiv/DatabaseManager.hh`, `src/Arxiv/DatabaseManager.cc`,
`src/Arxiv/AppCore.cc`

**Changes:**

### Config

Add to `ArticleSettings`:
```cpp
std::string keywords_file{"arxiv_keywords.txt"};
float cold_start_weight{1.0f};   // fades as ratings accumulate
int   cold_start_blend_at{20};   // number of ratings at which weight → 0
```
Load/save these in `Config::load_from_file` / `Config::save_to_file`.

### Ranker — keyword scorer

Add a new method pair:
```cpp
void   FitKeywords(const std::string &keywords_path);
float  PredictKeyword(const Article &article) const;
```

`FitKeywords` reads the file line by line, builds a vocabulary from the
keyword phrases (respecting multi-word phrases as n-grams exactly as
arxiv2md's `vectorize` does: infer min/max n-gram from the shortest and
longest phrase), computes an IDF over the article corpus already in
`m_vocab`, and stores the keyword TF-IDF vector in a new member
`m_keyword_vec` (same length as `MAX_FEATURES`, truncated/padded as needed).

`PredictKeyword` computes the TF-IDF vector of `article` (reuse `Vectorise`)
and returns the cosine similarity against `m_keyword_vec`, scaled to
`[1.0, 5.0]` using the same `ScaleOutput` used by the MLP path.

### Ranker — blended prediction

Add:
```cpp
float PredictBlended(const Article &article, int n_ratings, float cold_start_weight, int blend_at) const;
```

Implementation:
```
w = max(0.0f, cold_start_weight * (1.0f - (float)n_ratings / blend_at))
return w * PredictKeyword(article) + (1.0f - w) * (IsTrained() ? Predict(article) : 3.0f)
```

### DatabaseManager

Add a `relevance_score REAL DEFAULT 0.0` column to the `articles` table
(add a `CREATE TABLE IF NOT EXISTS` migration: check for column existence
with `PRAGMA table_info(articles)` and `ALTER TABLE articles ADD COLUMN
relevance_score REAL DEFAULT 0.0` if missing).

Add `void UpdateRelevanceScore(const std::string &link, float score)` and
`float GetRelevanceScore(const std::string &link)`.

### AppCore

After fetching new articles and inserting them into the DB, call
`m_ranker.FitKeywords(cfg.get_keywords_file())` then for each new article
call `PredictBlended` with `GetRatedArticles().size()` as `n_ratings`, and
call `UpdateRelevanceScore`. When the MLP finishes training
(`SpawnTrainingThread` completion), re-score all articles in the DB the
same way.

**Tests:**
- `TEST_CASE("Ranker keyword cold start")`: call `FitKeywords` with a temp
  file containing `"jet"` and `"QCD"`, call `PredictKeyword` on an article
  whose abstract contains those words, assert score > 3.0; assert score < 2.5
  on an unrelated abstract.
- `TEST_CASE("Ranker blended weight fades")`: assert `PredictBlended` with
  `n_ratings=0` equals `PredictKeyword` result; assert with
  `n_ratings=blend_at` it equals the MLP result (or 3.0 if untrained).

---

## Task 4 — Keyword editor in the TUI

**Why:** Users need to be able to tune keywords without leaving the TUI.
Editing a text file externally and restarting defeats the purpose.

**Files:** `include/Arxiv/Components.hh`, `src/Arxiv/Components.cc`,
`src/Arxiv/App.cc`, `include/Arxiv/KeyBindings.hh`

**Changes:**

1. Add a `KeywordEditorComponent` that renders as a full-screen FTXUI
   `Input` multi-line editor pre-populated with the contents of
   `arxiv_keywords.txt` (one keyword per line).
2. Wire the default key `K` (capital, to avoid collision) to open it as a
   modal overlay in `App`.
3. On save (`Enter` or a `Ctrl+S` binding): write the edited content back to
   `arxiv_keywords.txt`, call `AppCore::ReloadKeywords()` (new method: calls
   `FitKeywords` then re-scores all articles in DB and calls the UI refresh
   callback).
4. On cancel (`Escape`): discard changes.
5. Add `"keyword_editor"` to `KeyBindings` so the key is remappable via
   YAML config.

**Tests:** Unit-test `AppCore::ReloadKeywords` with a mock `DatabaseManager`
that expects `UpdateRelevanceScore` to be called for every article. No UI
tests needed.

---

## Task 5 — Daily-digest Markdown export

**Why:** Some users pipe the ranked output to Obsidian, static site
generators, or email. The existing export is project-scoped. A date-keyed
daily digest replicates the core workflow of `jxi24/arxiv2md` for users who
rely on that output format.

**Files:** `src/Arxiv/AppCore.cc`, `include/Arxiv/AppCore.hh`,
`src/Arxiv/App.cc`, `include/Arxiv/KeyBindings.hh`

**Changes:**

### AppCore

Add:
```cpp
bool ExportDailyDigest(const std::string &output_path, int days = 0) const;
```

`days = 0` means today's articles only. The method:
1. Calls `m_db->GetRecent(days == 0 ? 1 : days)` to get articles.
2. Sorts by `relevance_score` descending.
3. Writes `news/arxiv_YYMMDD.md` (create `news/` if absent) with this
   structure per article:

```markdown
## {title}

[Link]({link}) | {date}

**Authors:** {authors}

**Abstract:** {abstract}

**Relevance:** {relevance_score:.2f}
```

4. If an article has a project note, append it as a `> {note}` blockquote.
5. Returns `true` on success.

### YAML frontmatter variant

Add a boolean `AppCore::ExportDailyDigest(..., bool yaml_frontmatter)`. When
true, prepend each article section with:
```yaml
---
title: "{title}"
arxiv_id: "{id()}"
authors: [{authors}]
date: {YYYY-MM-DD}
relevance_score: {score}
tags: []
---
```
This makes the output importable as atomic notes in Obsidian/Logseq.

### CLI flag

In `main` (or wherever `App` is constructed), add `--export-today` and
`--export-yaml` flags that call `ExportDailyDigest` and exit without
launching the TUI.

### TUI binding

Add key `X` (default) to call `ExportDailyDigest` from within the running
TUI and show a status bar message with the output path.

**Tests:** `TEST_CASE("ExportDailyDigest writes sorted markdown")` with a
mock DB returning three articles with known `relevance_score` values; assert
the output file exists, assert the first heading matches the highest-scored
article, assert YAML frontmatter contains `arxiv_id` when flag is set.

---

## Task 6 — Author subscriptions

**Why:** Following specific authors is a high-value workflow — important
collaborators or prolific researchers whose papers should always surface
regardless of keyword or rating match.

**Files:** `include/Arxiv/DatabaseManager.hh`, `src/Arxiv/DatabaseManager.cc`,
`include/Arxiv/AppCore.hh`, `src/Arxiv/AppCore.cc`,
`include/Arxiv/Components.hh`, `src/Arxiv/Components.cc`

**Changes:**

### DatabaseManager

Add table (migration-safe, same `PRAGMA table_info` pattern as Task 3):
```sql
CREATE TABLE IF NOT EXISTS followed_authors (name TEXT PRIMARY KEY)
```

Add:
```cpp
void   FollowAuthor(const std::string &name);
void   UnfollowAuthor(const std::string &name);
std::vector<std::string> GetFollowedAuthors();
bool   IsFollowed(const std::string &name);
```

### AppCore

In `PredictBlended` (or in the score-update loop in `AppCore`): after
computing the blended score, check if any token in `article.authors` fuzzy-
matches a followed author (see Task 7 for fuzzy matching; for now, use
case-insensitive substring). If matched, add a configurable
`author_boost` (default `1.5`, exposed in `Config`) clamped to 5.0.

Add a `"Followed Authors"` entry to the filter list that shows only articles
where `IsFollowed` matches any author token.

### TUI

Add key `F` to follow/unfollow the author of the currently selected article
(toggle). Show a brief status bar confirmation. The author name is taken from
`article.authors` — if the field contains multiple authors, prompt with an
inline selector (FTXUI `Radiobox` over the split author list).

**Tests:** `TEST_CASE("FollowAuthor persists and filters")` on in-memory DB;
assert `GetFollowedAuthors()` returns the followed name; assert the filter
returns only matching articles.

---

## Task 7 — Fuzzy search

**Why:** The current `SearchArticles` uses `LIKE '%query%'`, which misses
partial author names and common typos. A fuzzy match significantly improves
search usability.

**Files:** `Dependencies.txt` (or `CMakeLists.txt` CPM section),
`include/Arxiv/DatabaseManager.hh`, `src/Arxiv/DatabaseManager.cc`

**Changes:**

1. Add `rapidfuzz` (header-only C++ port: `https://github.com/maxbachmann/rapidfuzz-cpp`)
   via CPM in `CMakeLists.txt`.
2. Change `SearchArticles` signature to add `int fuzzy_threshold = 80` (0–100
   score; 100 = exact, 0 = anything).
3. Keep the existing `LIKE` query as a first pass to avoid scanning all rows.
   Then post-filter results using `rapidfuzz::fuzz::partial_ratio` on title,
   authors, and abstract fields respectively, keeping results that exceed
   `fuzzy_threshold`.
4. When `query` is shorter than 4 characters, skip fuzzy and use exact `LIKE`
   only (too short for meaningful fuzzy matching).

**Tests:** `TEST_CASE("SearchArticles fuzzy")` on in-memory DB with articles
whose author is `"John Smith"`: assert a search for `"Jon Smth"` with
threshold 70 returns that article; assert threshold 95 does not.

---

## Task 8 — Automatic background feed refresh

**Why:** Users who leave the TUI open all day never see new articles unless
they manually trigger a fetch. A timer thread solves this with no user action
required.

**Files:** `include/Arxiv/AppCore.hh`, `src/Arxiv/AppCore.cc`,
`include/Arxiv/Config.hh`, `src/Arxiv/Config.cc`

**Changes:**

### Config

Add to `ArticleSettings`:
```cpp
int auto_refresh_minutes{0};  // 0 = disabled
```

### AppCore

Add `StartAutoRefresh()` and `StopAutoRefresh()` methods. `StartAutoRefresh`
spawns a `std::thread` that loops:
```
sleep for auto_refresh_minutes
call FetchArticles()
notify UI callback
```
Use a `std::condition_variable` (not `std::this_thread::sleep_for`) so that
`StopAutoRefresh` can wake the thread immediately on shutdown without waiting
for the full interval.

Guard against overlapping fetches with a `std::atomic<bool> m_fetching` flag
already likely present; skip the scheduled fetch if a fetch is in progress.

Call `StartAutoRefresh()` from the `AppCore` constructor when
`config.auto_refresh_minutes > 0`. Call `StopAutoRefresh()` in the
destructor before joining all threads (join order: training thread first,
then refresh thread).

Add `"auto_refresh_minutes: 360"` as a commented-out example in the
`.arxiv-tui.yml` config template written on first launch.

**Tests:** `TEST_CASE("AutoRefresh calls fetch after interval")`: set
interval to 1 (unit test only — not milliseconds), mock `Fetcher::Fetch` to
count calls, advance a fake clock, assert `Fetch` was called. Use dependency
injection for the clock (pass a `std::function<void()> sleep_fn` to
`StartAutoRefresh` defaulting to the real sleep, replaceable in tests).

---

## Completion checklist

Work through tasks in order 1 → 8. Each task should leave the test suite
green before moving to the next. The SQL fix (Task 1) and LaTeX normalization
(Task 2) are the only tasks with no user-visible surface — do them first so
all subsequent tasks build on a clean foundation.

After all tasks: run the full test suite, do a manual smoke test fetching
`hep-ph+cs.LG`, verify ranking produces non-uniform scores on day one
(cold-start keywords), rate three articles and verify the MLP blends in,
export a daily digest and open the file.
