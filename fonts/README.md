# Caption fonts

Fonts bundled for the **Caption Style** node. Anything dropped in this folder shows up
in that node's `font` dropdown after a ComfyUI restart, listed *before* the machine's
own fonts.

libass matches on the **family name recorded inside the file**, not the filename —
`ARIBLK.TTF` is `Arial Black`, `THEBOLDFONT.ttf` is `The Bold Font`. The node reads that
name out of each file's `name` table, so the dropdown always shows what libass will
actually accept.

## What is here

| Family | Files | Character |
|---|---|---|
| **Oswald** | 1 (variable) | Condensed grotesque. Long lines fit without shrinking — the default. |
| **Roboto** | 5 | Neutral workhorse. Disappears, which is usually the point. |
| **Luckiest Guy** | 1 | Cartoon poster weight, caps only. Loud and very legible. |
| **Shrikhand** | 1 | Heavy display italic. Loud in a different direction. |
| **Permanent Marker** | 1 | Handwritten marker. Informal, high texture. |
| **Pacifico** | 1 | Brush script. Low legibility at speed — pick it for mood, not reading. |
| **Comic Neue** | 4 | Softened Comic Sans. Friendly. |
| **Libre Baskerville** | 3 | Transitional serif. Reads considered rather than social. |
| **Fredericka the Great** | 1 | Etched and irregular. Decorative only. |
| **Nanum Pen** | 1 | Thin pen script, plus Hangul coverage. |
| **DejaVu Sans** | 2 | Wide glyph coverage. The safe multilingual fallback. |
| **Nunito** | 2 | Rounded humanist. **See the warning below.** |
| **Noto Sans Arabic** | 1 (variable) | Arabic script, right to left. |

All of the above are SIL Open Font License or Apache 2.0, which is why they can ship in
this repo.

## Nunito: only ExtraLight is here

The two Nunito files are ExtraLight and ExtraLight Italic. libass will not synthesise a
bold from them, so **Nunito + bold silently renders as Arial instead**. Verified: the two
frames come out byte-identical. Either leave `bold` off for Nunito, or drop the regular
and bold weights into this folder.

Worth knowing generally — a family whose weights are missing falls back without an error.
If a caption comes out in the wrong face, that is why.

## Adding your own

Copy `.ttf` or `.otf` files here and restart ComfyUI. For a folder you would rather keep
outside the repo, put its path in the node's `extra_fonts_dir` — that one is rescanned on
every run, so it does not need a restart to affect rendering (only to reach the dropdown).

Fonts already installed in Windows are picked up automatically and listed after these —
Arial, Impact, Segoe UI and about 180 others on a normal machine. They are **not**
bundled here: Arial in particular is Monotype's and cannot be redistributed.

## Chinese, Japanese, Korean

Not bundled — Noto Sans TC alone is 62 MB across its nine weights, which is more than the
rest of this folder by a factor of six. Download the weights you need from
[Google Fonts](https://fonts.google.com/noto) and drop them in.
