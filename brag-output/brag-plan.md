# Brag plan: Kavach

Run: brag (handed off to brag-slim on Opus 5.5). Input: this project + the Day 2 chat (PDF).
User direction: LinkedIn post about my experience in the Snapdragon AI Build & Present Challenge,
solo, max 40 s, no screen recording, no names to tag.

## Answers

- **What is it?** Kavach (कवच, "armour"): an offline AI scam shield for Windows PCs that catches
  screen-sharing scams like "digital arrest" while they happen.
- **Who is it for?** People talked into installing AnyDesk, opening net banking and reading out an OTP:
  parents, grandparents, first-time internet users.
- **What sets it apart?** It watches all three parts of the scam at once: the remote-access app, the screen
  (on-device OCR) and the call (on-device Whisper). Nothing leaves the PC. One click freezes the scammer.
- **Most impressive claim:** 10/10 scam scenarios caught, 0 false alarms in 10 normal ones, ~2 s median
  from scam phrase to warning, built solo in 2 days.
- **Visual hook:** the scam call itself, word by word: "This is the CBI. Don't disconnect the call.
  Install AnyDesk." Then the dark-red bilingual warning slams in.
- **Real UI to show:** the real overlay screenshots (`demo/screenshots/overlay.png`, `overlay_help.png`,
  `caution.png`), the real demo bank page (`demo/test_pages/mock_bank_transfer.html`, labelled DEMO),
  and the live dashboard rebuilt from `dashboard/app.py` (its colours, 270° risk gauge, band colours).
- **Tone:** freeform "scam-thriller cold open that flips into a proud build recap". Nearest preset:
  cinematic for the hook, polished afterwards.
- **Share caption:** "A scammer says 'This is the CBI. Install AnyDesk.' I spent 2 days building the AI that
  stops what happens next."

## Angle

Open on the scam, not on me. The viewer hears the call first, then sees Kavach catch it live, then sees the
build behind it (commits, tests, scenarios), and the numbers are the payoff.

Revision 2 (user request): removed every "Built solo in 2 days" line (the reveal chip and the sprint headline)
and the "Code in the comments ↓" button. No scene was cut; the timing and soundtrack are unchanged.

## Format

1080×1350 (4:5 portrait, LinkedIn's feed format), 30 fps, **38 s** (user cap: 40 s; brag's 15–25 s window is
overridden by the user's request for an experience video).

## Honesty rules

- Snapdragon numbers are AI Hub profiler estimates. Say "on the Snapdragon NPU (AI Hub profile)", never
  "on my Snapdragon laptop".
- "0 false alarms" = 0 alerts and 0 false cautions across the 10 normal scenarios (README, final config).
- Every name, number and page in the demo is fictional and labelled DEMO. No caller names are shown.
- "Phone froze" is the real AnyDesk test from the README (2026-09-30).

## Visual identity (from the code)

- Navy `#0F172A`, card `#1E293B`, text `#F8FAFC`, muted `#94A3B8`, grid `#334155` (dashboard/app.py)
- Bands: quiet `#22C55E`, caution `#F59E0B`, alert `#EF4444`; overlay red `#7A0000`; toast amber `#FFD54A`
- Type: Inter (stand-in for Segoe UI), Noto Sans Devanagari (stand-in for Nirmala UI), JetBrains Mono
  (stand-in for Consolas)

## Music

120 BPM, A minor, 4/4. One beat = 0.5 s = 15 frames; one bar = 2 s. Every cut lands on a bar line.
Progression: drone (hook) → Am F → Am F C G (demo, rising) → Am stabs (alert) → drop on the click
("freeze") → C G (relief) → Am F (the build) → F C G (results) → F → C (outro, rings out).
Effects in key, under the music: ring pulses (E5/A5), impacts on bar lines, soft plucks (A5 C6 E6) as each
signal lights, a click and an ice shimmer on "Stop and get help".

## Storyboard

| # | Time | Scene | On screen | Sound |
|---|---|---|---|---|
| 1 | 0.0–4.0 | **Hook: the call** | Black. Pill "Unknown caller · 00:07" with a pulsing red dot. Huge lines word by word: "This is the CBI." (0.2) / "Don't disconnect the call." (1.2) / "Install AnyDesk." (2.2, red). Voice waveform at the bottom. | Low A drone, heartbeat kick, ring pulses in key, riser into 4.0 |
| 2 | 4.0–8.0 | **Reveal** | Navy. Shield draws in. "Kavach" + "कवच". "An offline AI scam shield for Windows PCs". Pill: "Snapdragon AI Build & Present Challenge" (5.5). | Impact at 4.0, Am → F, beat enters |
| 3 | 8.0–16.0 | **Live demo** | Heading "It watches all 3 parts of the scam". A laptop screen shows the real DEMO bank page; the dashboard below (gauge + 3 signal rows). 8.5: Screen signal (+45), gauge 0→45 QUIET. 10.5: "AnyDesk · session live" pill, Remote signal (+30), gauge →69 CAUTION ("No scam talk yet: capped at caution", the tactic gate), real caution toast at 11.0. 12.0: "On the call" bubble "This is the CBI. You are under digital arrest." 13.5: Call signal (+26), gauge →100 ALERT. | Full groove Am F C G, plucks on each signal, riser 13.5–14.5 |
| 4 | 16.0–24.0 | **The warning + the freeze** | Background turns overlay red. The real full-screen warning fills the laptop screen. Caption "A full-screen warning, in Hindi and English" (16.3). 18.0: cursor glides to "Stop and get help", clicks at 19.0; hard cut to the real "You did the right thing" screen with an "AnyDesk paused" chip. Captions: "One click freezes the scammer's remote session" (19.5) / "Tested with real AnyDesk from my phone. The phone froze." (21.0) | Alert slam + Am stabs; on the click the drums drop for a beat with an ice shimmer; relief C → G |
| 5 | 24.0–28.0 | **The build** | Real commit messages scroll as texture. Centred counters pop in on the beat (24.0 / 24.5 / 25.0) and count up: 23 commits · 362 tests · 20 end-to-end scenarios. | Upbeat Am F, 16th hats, key ticks |
| 6 | 28.0–34.0 | **Results** | 2×2 cards: 10/10 scam scenarios caught · 0 false alarms in 10 normal ones · ~2 s scam phrase → warning · 6× faster speech AI on the Snapdragon NPU vs CPU (footnote: Whisper, 117 vs 723 ms per 5 s of audio, AI Hub profile). | F C G, a tick per card on the beat |
| 7 | 34.0–38.0 | **Outro** | Shield + "Kavach कवच". "Offline. On-device. Nothing leaves your PC." (centred vertically) | Impact, F → C rings out, fade |

Durations: 4 + 4 + 8 + 8 + 4 + 6 + 4 = 38 s.

## Poster

The demo at 15.3 s: gauge at 100 ALERT, all three signals lit, the scam quote on screen. It tells the whole
story in one still (the red warning is striking but its text is too small at thumbnail size). Baked in as frame 0.

## Build

`bash work/build.sh` re-creates `brag.mp4` and `brag.jpg` from `work/composition/index.html` (every frame is a pure
function of time, captured with headless Chromium) and `work/music.py` (numpy synth, normalized to -14 LUFS).
