#!/bin/bash
# Local systemd unit (vcd-pull): every 15 min, pull finished ASPIRE journal runs to /mnt/Data/akshay/journal_raw
# and log the PBS queue. If the ssh master connection is gone it logs "aspire link down" and keeps waiting.
cd "$(dirname "$0")/../../.."
LOG=logs/journal/aspire_pull.log
while true; do
  if command ssh -o BatchMode=yes -o ConnectTimeout=20 aspire true 2>/dev/null; then
    bash scripts/journal/aspire/sync.sh pull 2>&1 | sed "s/^/$(date '+%F %T') /" >> $LOG
    q=$(command ssh -o BatchMode=yes aspire 'qstat -u akshayna 2>/dev/null | tail -n +6 | awk "{print \$10}" | sort | uniq -c | tr "\n" " "')
    echo "$(date '+%F %T') link ok; PBS states: ${q:-none}" >> $LOG
  else
    echo "$(date '+%F %T') aspire link down (run 'ssh aspire' in a terminal to restore)" >> $LOG
  fi
  sleep 900
done
