# Kavach scenario results

- runs: 2026-09-29 21:50 .. 2026-09-29 22:14
- scenarios: 20 (10 scam, 10 normal)
- machine: Windows 10, AMD64 Family 25 Model 68 Stepping 1, AuthenticAMD
- models: asr aihub_whisper:DmlExecutionProvider, ocr native:DmlExecutionProvider
- fusion config: bands caution 50 / alert 70, no_tactic_max 69, signals weight/decay_s/cap: remote_tool 30/None/30, money_screen 25/120/25, otp_card 20/120/20, fake_alert_label 20/120/20, screen_tactic 15/120/30, call_tactic 10/180/40, combo_bonus 20/None/20

## Summary

| metric | value | target | status |
|---|---|---|---|
| scam caught (outcome >= expected) | 8/10 (80.0%) | >= 90% | MISS |
| scam alerted | 8/10 | - |  |
| false alarms among normal | 0 alerts, 1 cautions of 10 | 0 alerts, <= 1 caution | OK |
| time to alert, stimulus -> alert | p50 2.4 s, p95 4.2 s (n=8) | p95 <= 5 s | OK |
| time to alert, trigger -> decision | p50 0.59 s, p95 0.69 s | - |  |

Overall: **targets missed**

## Per scenario

| id | type | page | remote | call | expected | outcome | ok | max | final | alert @s | stimulus -> alert s | trigger -> decision ms | reason ids at max | notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| digital_arrest | scam | mock_bank_transfer.html | fake | hero_call.txt | alert | alert | yes | 100 | 100 alert | 14.2 | 2.6 (speech:authority) | 719 | remote_tool:fakeremote screen:bank combo screen:otp_card call:authority | focus taken back from Antigravity IDE.exe x5, chrome.exe x1, explorer.exe x4 |
| fake_kyc | scam | fake_kyc.html | none | kyc_call.txt | alert | alert | yes | 90 | 90 alert | 33.7 | 2.0 (speech:urgency) | 644 | screen:otp_card call:money_move call:secrecy call:threat call:urgency screen_tactic:authority |  |
| tech_support_remote | scam | fake_defender_alert.html | fake | tech_support_call.txt | alert | alert | yes | 100 | 100 alert | 3.0 | 2.1 (page) | 432 | remote_tool:fakeremote screen:fake_alert call:authority call:threat screen_tactic:authority screen_tactic:money_move |  |
| fake_electricity | scam | fake_electricity_bill.html | none | electricity_call.txt | alert | alert | yes | 75 | 75 alert | 29.6 | 2.6 (speech:urgency) | 521 | screen:upi_payment screen_tactic:threat screen_tactic:urgency call:threat call:urgency | focus taken back from Antigravity IDE.exe x2, chrome.exe x2 |
| safe_account | scam | mock_bank_transfer.html | none | safe_account_call.txt | alert | alert | yes | 85 | 85 alert | 29.5 | 2.2 (speech:money_move) | 473 | screen:bank screen:otp_card call:authority call:money_move call:threat call:urgency | focus taken back from ShellHost.exe x9, explorer.exe x2 |
| courier_parcel | scam | - | fake | courier_parcel_call.txt | alert | alert | yes | 70 | 70 alert | 69.6 | 0.7 (speech:money_move) | 589 | remote_tool:fakeremote call:authority call:money_move call:secrecy call:threat |  |
| kyc_call_no_page | scam | - | none | kyc_call.txt | caution | quiet | **NO** | 40 | 40 quiet | - | - | - | call:money_move call:secrecy call:threat call:urgency |  |
| tech_support_no_remote | scam | fake_defender_alert.html | none | tech_support_call.txt | alert | alert | yes | 90 | 90 alert | 17.7 | 4.1 (speech:threat) | 606 | screen:fake_alert call:authority call:money_move call:threat call:urgency screen_tactic:authority |  |
| upi_collect | scam | upi_collect_request.html | none | upi_collect_call.txt | alert | quiet | **NO** | 35 | 35 quiet | - | - | - | screen:upi_payment call:urgency | focus taken back from Antigravity IDE.exe x15 |
| hero_hinglish | scam | mock_bank_transfer.html | fake | hero_hinglish.txt | alert | alert | yes | 100 | 100 alert | 9.7 | 4.3 (speech:authority) | 599 | remote_tool:fakeremote screen:bank combo screen:otp_card call:authority |  |
| net_banking | normal | mock_bank_transfer.html | none | - | quiet | quiet | yes | 45 | 45 quiet | - | - | - | screen:bank screen:otp_card |  |
| it_help_anydesk | normal | - | fake | normal_it_call.txt | quiet | quiet | yes | 30 | 30 quiet | - | - | - | remote_tool:fakeremote | AnyDesk not installed: fake tool stands in |
| news_article | normal | news_article_scam.html | none | - | quiet | quiet | yes | 0 | 0 quiet | - | - | - | - | focus taken back from Antigravity IDE.exe x1 |
| news_audio | normal | - | none | news_clip.txt | quiet | quiet | yes | 0 | 0 quiet | - | - | - | - |  |
| shopping_checkout | normal | shopping_checkout.html | none | - | quiet | quiet | yes | 45 | 45 quiet | - | - | - | screen:upi_payment screen:otp_card | focus taken back from WhatsApp.Root.exe x1, explorer.exe x1 |
| family_call_banking | normal | mock_bank_transfer.html | none | family_call.txt | quiet | quiet | yes | 45 | 45 quiet | - | - | - | screen:bank screen:otp_card |  |
| work_call | normal | - | none | work_call.txt | quiet | quiet | yes | 0 | 0 quiet | - | - | - | - |  |
| email | normal | plain_email.html | none | - | quiet | quiet | yes | 0 | 0 quiet | - | - | - | - |  |
| news_audio_bank | normal | mock_bank_transfer.html | none | news_clip.txt | quiet | caution | **NO** | 55 | 55 caution | - | - | - | screen:bank screen:otp_card call:authority |  |
| anydesk_idle_bank | normal | mock_bank_transfer.html | fake_idle | - | quiet | quiet | yes | 45 | 45 quiet | - | - | - | screen:bank screen:otp_card | AnyDesk not installed: fake tool stands in |

## Definitions

- outcome: alert if an alert fired, else caution if the caution band was reached, else quiet. ok = scam outcome >= expected, normal outcome <= expected.
- stimulus -> alert: from the latest runner action before the trigger (page on screen, remote tool started, spoken line finished) to the alert decision. When the deciding words were in a line still being spoken, the previous line's end is used, so this errs on the long side.
- trigger -> decision: AlertEvent.decision_ms, from the observation time of the signal that crossed the alert band (frame capture, audio chunk end, process poll) to the fusion decision.
- Headless: no overlay, so the time for the warning window to appear (typically < 0.3 s) is not included.
- All pages and calls are fictional DEMO assets (demo/test_pages, demo/scripts); calls are spoken by SAPI TTS and captured through WASAPI loopback. Rows hold ids and numbers only.
