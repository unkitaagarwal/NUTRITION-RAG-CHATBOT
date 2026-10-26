# Extractor A/B results — old code vs new code

Run 2026-10-26. Branch `harsh_changes_oct_26`, commits a3da10b (old) vs 5957434 (new).

## Method

Both versions run as gunicorn on the SAME machine with the SAME (keyless) env,
so any difference is code and nothing else. An earlier local-vs-Render run was
discarded: it showed 22 "regressions" that were all missing credentials, not code.

No API keys were available for this run:
`OPENAI_API_KEY`, `SOCIAL_PROXY`, `SCRAPER_API_KEY`, cookies all unset.

## Headline

| | |
|---|---|
| URLs compared | 228 unique (106 recent website, 80 share-links, 42 historical website failures) |
| **REGRESSIONS** | **0** |
| both ok | 31 |
| both broken | 197 (almost all need cookies/keys this run did not have) |

A 31-URL regression net re-run against the final code reported 1 regression;
retested directly, that URL returns 17 ingredients on BOTH versions twice over.
It was the site rate-limiting repeated hits. **Net real regressions: 0.**

## Clipping (the biggest change) — proven

Same URLs, same format string, old vs new:

| source | OLD file | NEW file |
|---|---|---|
| 635s | 635s, 3.9MB, 2s | **300s**, 3.6MB, 5s |
| 322s | 322s, 2.2MB, 2s | 300s, 3.9MB, 5s |
| 19s  | 19s, 0.1MB | 19s, 0.1MB (untouched) |

The 635s row proves the old `download_sections` was a silent no-op: old code
asked for 0-300s and got the whole 635s file.

The 322s row showed clipping can be WORSE (bigger + slower) for small overage,
so `VIDEO_CLIP_MARGIN=1.2` was added: clip only above 360s.

YouTube downloaded with NO cookies and NO proxy — the `ios` player client
clears the bot challenge. That is the android_vr removal fixed.

## What this run could NOT measure

- **All Facebook** (138 URLs): 0/138 on BOTH versions — blocked without cookies.
  The login-wall fix is unevaluated.
- **Blog 403 ladder**: ScraperAPI unset, so there is no tier to escalate to.
- **TikTok/Instagram** (77% of real traffic): need cookies + OPENAI_API_KEY.

## Correction to an earlier claim

`share.google` and `pin.it` already worked on OLD code (1/1 and 6/14 on both).
The resolver extension was verified in isolation, not against old code
end-to-end. It shows no measurable benefit here. It is not a regression either.

## To finish

Fill `.env` (OPENAI_API_KEY, YTDLP_COOKIES_B64, SOCIAL_PROXY, SCRAPER_API_KEY)
and re-run:

    /tmp/run_batch.sh 160     # repeat until N/N complete
