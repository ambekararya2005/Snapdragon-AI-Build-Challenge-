# Kavach fusion tuning: before / after (replay of recorded scenario runs)

Every recorded scenario is re-scored from its derived signal stream (bench/results/scenario_runs.jsonl) under both configs; only the fusion section is re-evaluated.

## Config changes

| key | before | after |
|---|---|---|
| fusion.signals.call_tactic.cap | 40 | 65 |
| fusion.signals.call_tactic.weight | 10 | 13 |

## Summary

| metric | before | after | target |
|---|---|---|---|
| scam caught (outcome >= expected) | 8/10 (80.0%) | 9/10 (90.0%) | >= 90% |
| scam alerted | 8/10 | 8/10 | - |
| false alarms among normal | 0 alerts, 1 cautions of 10 | 0 alerts, 1 cautions of 10 | 0 alerts, <= 1 caution |
| time to alert, stimulus -> alert | p50 2.4 s, p95 4.2 s (n=8) | p50 2.3 s, p95 4.2 s (n=8) | p95 <= 5 s |
| time to alert, trigger -> decision | p50 0.59 s, p95 0.69 s | p50 0.59 s, p95 0.69 s | - |

## Per scenario

| id | type | expected | before | after | ok before | ok after | e2e s before | e2e s after | reason ids at max (after) |
|---|---|---|---|---|---|---|---|---|---|
| digital_arrest | scam | alert | alert 100 | alert 100 | yes | yes | 2.6 | 2.6 | remote_tool:fakeremote screen:bank combo screen:otp_card call:authority |
| fake_kyc | scam | alert | alert 90 | alert 100 | yes | yes | 2.0 | 2.0 | screen:otp_card call:money_move call:secrecy call:threat call:urgency screen_tactic:authority |
| tech_support_remote | scam | alert | alert 100 | alert 100 | yes | yes | 2.1 | 2.1 | remote_tool:fakeremote screen:fake_alert call:authority call:threat screen_tactic:authority screen_tactic:money_move |
| fake_electricity | scam | alert | alert 75 | alert 81 | yes | yes | 2.6 | 2.6 | screen:upi_payment screen_tactic:threat screen_tactic:urgency call:threat call:urgency |
| safe_account | scam | alert | alert 85 | alert 100 | yes | yes | 2.2 | 1.8 | screen:bank screen:otp_card call:authority call:money_move call:secrecy call:threat |
| courier_parcel | scam | alert | alert 70 | alert 82 | yes | yes | 0.7 | 0.7 | remote_tool:fakeremote call:authority call:money_move call:secrecy call:threat |
| kyc_call_no_page | scam | caution | quiet 40 | caution 52 | NO | yes | - | - | call:money_move call:secrecy call:threat call:urgency |
| tech_support_no_remote | scam | alert | alert 90 | alert 100 | yes | yes | 4.1 | 4.1 | screen:fake_alert call:authority call:money_move call:threat call:urgency screen_tactic:authority |
| upi_collect | scam | alert | quiet 35 | quiet 38 | NO | NO | - | - | screen:upi_payment call:urgency |
| hero_hinglish | scam | alert | alert 100 | alert 100 | yes | yes | 4.3 | 4.3 | remote_tool:fakeremote screen:bank combo screen:otp_card call:authority |
| net_banking | normal | quiet | quiet 45 | quiet 45 | yes | yes | - | - | screen:bank screen:otp_card |
| it_help_anydesk | normal | quiet | quiet 30 | quiet 30 | yes | yes | - | - | remote_tool:fakeremote |
| news_article | normal | quiet | quiet 0 | quiet 0 | yes | yes | - | - | - |
| news_audio | normal | quiet | quiet 0 | quiet 0 | yes | yes | - | - | - |
| shopping_checkout | normal | quiet | quiet 45 | quiet 45 | yes | yes | - | - | screen:upi_payment screen:otp_card |
| family_call_banking | normal | quiet | quiet 45 | quiet 45 | yes | yes | - | - | screen:bank screen:otp_card |
| work_call | normal | quiet | quiet 0 | quiet 0 | yes | yes | - | - | - |
| email | normal | quiet | quiet 0 | quiet 0 | yes | yes | - | - | - |
| news_audio_bank | normal | quiet | caution 55 | caution 58 | NO | NO | - | - | screen:bank screen:otp_card call:authority |
| anydesk_idle_bank | normal | quiet | quiet 45 | quiet 45 | yes | yes | - | - | screen:bank screen:otp_card |
