# GitNexus Reference for CBO

This project is indexed by GitNexus as **CBO**.

Use this file as the detailed GitNexus reference. Keep `AGENTS.md` and `CLAUDE.md` short so agents see the essential rules quickly.

## Purpose

GitNexus gives code intelligence for the repository, including:

- symbol context
- callers and callees
- execution flows
- impact analysis
- safe renaming
- change detection

Use it when you need to understand the codebase, debug a flow, assess risk, or refactor safely.

## Core Workflow

### 1. Explore by concept

Use this when you know the feature or behavior but not the symbol name.

```js
gitnexus_query({ query: "training loop" })
```

Other useful examples:

```js
gitnexus_query({ query: "checkpoint saving" })
gitnexus_query({ query: "calibration metrics" })
gitnexus_query({ query: "OOD evaluation" })
gitnexus_query({ query: "AUROC logging" })
```

### 2. Inspect a symbol

Use this when you know the function, class, or method name.

```js
gitnexus_context({ name: "symbolName" })
```

This should show useful information such as callers, callees, and related execution flows.

### 3. Run impact analysis before editing

Before modifying any function, class, or method, run:

```js
gitnexus_impact({ target: "symbolName", direction: "upstream" })
```

Report the blast radius briefly:

- direct callers
- affected execution flows
- risk level

If risk is **HIGH** or **CRITICAL**, warn the user before editing.

### 4. Verify changes before finishing

After editing, run:

```js
gitnexus_detect_changes()
```

Use this to check whether the affected files, symbols, and execution flows match the intended scope.

## Debugging Workflow

When debugging a failure:

1. Search by symptom:

```js
gitnexus_query({ query: "error message or failing behavior" })
```

2. Inspect likely symbols:

```js
gitnexus_context({ name: "suspectFunction" })
```

3. Trace relevant execution flows, if available.

4. For regressions, compare against `main`:

```js
gitnexus_detect_changes({ scope: "compare", base_ref: "main" })
```

## Refactoring Workflow

### Renaming

Never rename symbols with plain find-and-replace.

Dry-run the rename first:

```js
gitnexus_rename({ symbol_name: "oldName", new_name: "newName", dry_run: true })
```

Review the preview carefully. Then apply the rename only if the preview is correct.

### Extracting or splitting code

Before extracting, splitting, or moving code:

1. Inspect the symbol:

```js
gitnexus_context({ name: "targetSymbol" })
```

2. Check impact:

```js
gitnexus_impact({ target: "targetSymbol", direction: "upstream" })
```

3. Update affected direct callers/importers.

4. Verify the final change set:

```js
gitnexus_detect_changes({ scope: "all" })
```

## Risk Levels

| Level | Meaning | Action |
|---|---|---|
| LOW | Local or limited impact | Proceed carefully |
| MEDIUM | Several dependents or flows affected | Test relevant paths |
| HIGH | Important shared logic or many dependents affected | Warn user before editing |
| CRITICAL | Core execution path or broad breakage risk | Warn user and proceed only with explicit care |

Depth-based impact may also appear:

| Depth | Meaning | Action |
|---|---|---|
| d=1 | Direct callers/importers | Must update or verify |
| d=2 | Indirect dependents | Should test |
| d=3 | Transitive dependents | Test if on a critical path |

## Useful Commands

| Task | Command |
|---|---|
| Find code by concept | `gitnexus_query({ query: "..." })` |
| Inspect symbol | `gitnexus_context({ name: "..." })` |
| Check blast radius | `gitnexus_impact({ target: "...", direction: "upstream" })` |
| Verify changes | `gitnexus_detect_changes()` |
| Compare branch to main | `gitnexus_detect_changes({ scope: "compare", base_ref: "main" })` |
| Dry-run rename | `gitnexus_rename({ symbol_name: "old", new_name: "new", dry_run: true })` |

## Useful Resources

If available, these GitNexus resources are useful:

| Resource | Use |
|---|---|
| `gitnexus://repo/CBO/context` | Codebase overview and index freshness |
| `gitnexus://repo/CBO/clusters` | Functional areas |
| `gitnexus://repo/CBO/processes` | Execution flows |
| `gitnexus://repo/CBO/process/{name}` | Step-by-step flow trace |

## Index Freshness

If GitNexus says the index is stale, run:

```bash
npx gitnexus analyze
```

If the project previously used embeddings, preserve them:

```bash
npx gitnexus analyze --embeddings
```

To check whether embeddings exist, inspect:

```bash
.gitnexus/meta.json
```

If `stats.embeddings` is greater than 0, use `--embeddings` when re-analyzing.

## Final Self-Check

Before finishing a code modification task, confirm:

1. Relevant symbols were inspected.
2. Impact analysis was run for modified symbols.
3. HIGH or CRITICAL risks were not ignored.
4. Direct callers/importers were updated or verified.
5. `gitnexus_detect_changes()` confirms the scope is expected.
