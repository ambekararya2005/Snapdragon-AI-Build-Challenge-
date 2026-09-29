# Kavach scenario results

- runs: 2026-09-29 21:50 .. 2026-09-29 23:15
- scenarios: 20 (10 scam, 10 normal); 4 live under this config, 16 replayed (see Definitions)
- machine: Windows 10, AMD64 Family 25 Model 68 Stepping 1, AuthenticAMD
- models: asr aihub_whisper:DmlExecutionProvider, ocr native:DmlExecutionProvider
- fusion config: bands caution 50 / alert 70, no_tactic_max 69, signals weight/decay_s/cap: remote_tool 30/None/30, money_screen 25/120/25, otp_card 20/120/20, fake_alert_label 20/120/20, screen_tactic 15/120/30, call_tactic 13/180/65, combo_bonus 20/None/20

## Summary

| metric | value | target | status |
|---|---|---|---|
| scam caught (outcome >= expected) | 10/10 (100.0%) | >= 90% | OK |
| scam alerted | 9/10 | - |  |
| false alarms among normal | 0 alerts, 0 false cautions of 10 | 0 alerts, <= 1 caution | OK |
| time to alert, stimulus -> alert | p50 2.0 s, p95 4.2 s (n=9) | p95 <= 5 s | OK |
| time to alert, trigger -> decision | p50 0.60 s, p95 0.70 s | - |  |

Overall: **all targets met**

## Per scenario

| id | source | type | page | remote | call | expected | outcome | ok | max | final | alert @s | stimulus -> alert s | trigger -> decision ms | reason ids at max | notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| digital_arrest | replay | scam | mock_bank_transfer.html | fake | hero_call.txt | alert | alert | yes | 100 | 100 alert | 14.2 | 2.6 (speech:authority) | 719 | remote_tool:fakeremote screen:bank combo screen:otp_card call:authority | focus taken back from Antigravity IDE.exe x5, chrome.exe x1, explorer.exe x4 |
| fake_kyc | replay | scam | fake_kyc.html | none | kyc_call.txt | alert | alert | yes | 100 | 100 alert | 33.7 | 2.0 (speech:urgency) | 645 | screen:otp_card call:money_move call:secrecy call:threat call:urgency screen_tactic:authority |  |
| tech_support_remote | replay | scam | fake_defender_alert.html | fake | tech_support_call.txt | alert | alert | yes | 100 | 100 alert | 3.0 | 2.1 (page) | 432 | remote_tool:fakeremote screen:fake_alert call:authority call:threat screen_tactic:authority screen_tactic:money_move |  |
| fake_electricity | replay | scam | fake_electricity_bill.html | none | electricity_call.txt | alert | alert | yes | 81 | 81 alert | 29.6 | 2.6 (speech:urgency) | 521 | screen:upi_payment screen_tactic:threat screen_tactic:urgency call:threat call:urgency | focus taken back from Antigravity IDE.exe x2, chrome.exe x2 |
| safe_account | replay | scam | mock_bank_transfer.html | none | safe_account_call.txt | alert | alert | yes | 100 | 100 alert | 17.6 | 1.8 (speech:threat) | 543 | screen:bank screen:otp_card call:authority call:money_move call:secrecy call:threat | focus taken back from ShellHost.exe x9, explorer.exe x2 |
| courier_parcel | replay | scam | - | fake | courier_parcel_call.txt | alert | alert | yes | 82 | 82 alert | 69.6 | 0.7 (speech:money_move) | 589 | remote_tool:fakeremote call:authority call:money_move call:secrecy call:threat |  |
| kyc_call_no_page | live | scam | - | none | kyc_call.txt | caution | caution | yes | 65 | 65 caution | - | - | - | call:authority call:money_move call:secrecy call:threat call:urgency |  |
| tech_support_no_remote | replay | scam | fake_defender_alert.html | none | tech_support_call.txt | alert | alert | yes | 100 | 100 alert | 17.7 | 4.1 (speech:threat) | 606 | screen:fake_alert call:authority call:money_move call:threat call:urgency screen_tactic:authority |  |
| upi_collect | live | scam | upi_collect_request.html | none | upi_collect_call.txt | alert | alert | yes | 86 | 86 alert | 29.7 | 1.0 (speech:money_move) | 668 | screen:upi_payment screen:otp_card screen_tactic:money_move call:money_move call:urgency |  |
| hero_hinglish | replay | scam | mock_bank_transfer.html | fake | hero_hinglish.txt | alert | alert | yes | 100 | 100 alert | 9.7 | 4.3 (speech:authority) | 599 | remote_tool:fakeremote screen:bank combo screen:otp_card call:authority |  |
| net_banking | replay | normal | mock_bank_transfer.html | none | - | quiet | quiet | yes | 45 | 45 quiet | - | - | - | screen:bank screen:otp_card |  |
| it_help_anydesk | replay | normal | - | fake | normal_it_call.txt | quiet | quiet | yes | 30 | 30 quiet | - | - | - | remote_tool:fakeremote | AnyDesk not installed: waitfor.exe stands in (fake remote tool, user process) |
| news_article | replay | normal | news_article_scam.html | none | - | quiet | quiet | yes | 0 | 0 quiet | - | - | - | - | focus taken back from Antigravity IDE.exe x1 |
| news_audio | replay | normal | - | none | news_clip.txt | quiet | quiet | yes | 0 | 0 quiet | - | - | - | - |  |
| shopping_checkout | replay | normal | shopping_checkout.html | none | - | quiet | quiet | yes | 45 | 45 quiet | - | - | - | screen:upi_payment screen:otp_card | focus taken back from WhatsApp.Root.exe x1, explorer.exe x1 |
| family_call_banking | replay | normal | mock_bank_transfer.html | none | family_call.txt | quiet | quiet | yes | 45 | 45 quiet | - | - | - | screen:bank screen:otp_card |  |
| work_call | replay | normal | - | none | work_call.txt | quiet | quiet | yes | 0 | 0 quiet | - | - | - | - |  |
| email | replay | normal | plain_email.html | none | - | quiet | quiet | yes | 0 | 0 quiet | - | - | - | - |  |
| news_audio_bank | live | normal | mock_bank_transfer.html | none | news_clip.txt | quiet | quiet | yes | 45 | 45 quiet | - | - | - | screen:bank screen:otp_card | 3 live runs (row = worst): quiet 45, quiet 45, quiet 45 |
| anydesk_idle_bank | live | normal | mock_bank_transfer.html | fake | - | caution | caution | yes | 69 | 69 caution | - | - | - | remote_tool:fakeremote screen:bank screen:otp_card gate:no_tactic | AnyDesk not installed: waitfor.exe stands in (fake remote tool, user process) |

## Definitions

- source: live = measured live under the current config and lexicons; replay = the scenario's recorded signal stream (from an earlier live run) re-scored by the fusion engine under the current config. Replay assumes the detectors produce the same labels/tactics; the lexicon and guard changes since those runs were checked to leave these scenarios' pages and scripts unchanged.
- outcome: alert if an alert fired, else caution if the caution band was reached, else quiet. ok = scam outcome >= expected, normal outcome <= expected. A caution counts as a false alarm only where the scenario expects quiet (anydesk_idle_bank expects caution by design: tactic gate).
- stimulus -> alert: when the alert's trigger is speech, from the moment the tipping phrase was spoken (speech:<tactic>; found on the script text, placed within its line by character position) to the alert decision; otherwise from the latest runner action before the trigger (page on screen, remote tool started, spoken line finished).
- trigger -> decision: AlertEvent.decision_ms, from the observation time of the signal that crossed the alert band (frame capture, audio chunk end, process poll) to the fusion decision.
- Headless: no overlay, so the time for the warning window to appear (typically < 0.3 s) is not included.
- All pages and calls are fictional DEMO assets (demo/test_pages, demo/scripts); calls are spoken by SAPI TTS and captured through WASAPI loopback. Rows hold ids and numbers only.
