#!/usr/bin/env bash
# Hourly pull of the sandbox probe sweep (SANDBOX.md). Reconciles by listing: any remote
# file missing locally or with a different size is fetched, so a truncated transfer is
# simply retried next round (a timestamp marker lost files when a tar stream broke).
cd "$(dirname "$0")/.."
S="ssh -i .ssh -p 31223 -o ConnectTimeout=30 gw@lum.id"
while true; do
  $S 'bash -c "cd /home/riwk2015/fyp; find hrt/artifacts/runs_probe hrt/logs/probe_sweep.log -type f -printf \"%s %p\n\""' | sort > /tmp/pp_remote.lst
  find hrt/artifacts/runs_probe hrt/logs/probe_sweep.log -type f -printf "%s %p\n" 2>/dev/null | sort > /tmp/pp_local.lst
  comm -23 /tmp/pp_remote.lst /tmp/pp_local.lst | cut -d" " -f2- > /tmp/pp_need.lst
  if [ -s /tmp/pp_need.lst ]; then
    $S 'bash -c "cd /home/riwk2015/fyp && tar czf - -T -"' < /tmp/pp_need.lst | tar xzf -
  fi
  echo "$(date -u +%H:%M) fetched $(wc -l < /tmp/pp_need.lst) files; $(ls hrt/artifacts/runs_probe/*/*.json | wc -l) runs"
  grep -q "PROBE SWEEP COMPLETE" hrt/logs/probe_sweep.log 2>/dev/null && [ ! -s /tmp/pp_need.lst ] && break
  sleep 3600
done
