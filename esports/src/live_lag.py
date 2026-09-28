"""Measure the livestats feed's real-time publication delay on live games.

usage: python live_lag.py START_ISO MINUTES OUT.jsonl

Waits until START, then for MINUTES: every ~2 s, for every in-progress game of a live event, requests the window at
startingTime = now - X for the smallest X that returns frames (re-probed each minute), and logs
now - newest frame timestamp. Stream `offset` values from getLive are logged too (the site uses them to sync stats to video).
"""
import sys, time, json, datetime as dt
import lolapi
from collect_frames import iso

START = dt.datetime.fromisoformat(sys.argv[1].replace('Z', '+00:00'))
END = START + dt.timedelta(minutes=int(sys.argv[2]))
OUT = open(sys.argv[3], 'a')
now = lambda: dt.datetime.now(dt.timezone.utc)

while now() < START:
    time.sleep(30)

best_x, probed = {}, {}
while now() < END:
    try:
        ev = lolapi.get(lolapi.GW + '/getLive', {'hl': 'en-US'})['data']['schedule'].get('events') or []
    except Exception:
        time.sleep(5); continue
    games = []
    for e in ev:
        m = e.get('match') or {}
        offs = [(s.get('provider'), s.get('parameter'), s.get('offset'), s.get('statsStatus')) for s in e.get('streams', [])]
        for g in m.get('games', []):
            if g.get('state') == 'inProgress':
                games.append((g['id'], e['league']['slug'], offs))
    t_loop = time.time()
    while time.time() - t_loop < 60 and now() < END:
        for gid, lg, offs in games:
            n = now()
            if gid not in best_x or time.time() - probed.get(gid, 0) > 60:
                probed[gid] = time.time()
                for x in range(0, 121, 10):
                    w = lolapi.window(gid, iso(n - dt.timedelta(seconds=x)))
                    if w and w.get('frames'):
                        best_x[gid] = x
                        break
            x = best_x.get(gid)
            if x is None:
                OUT.write(json.dumps(dict(now=n.isoformat(), game_id=gid, league=lg, x=None, offsets=offs)) + '\n'); OUT.flush()
                continue
            n = now()
            w = lolapi.window(gid, iso(n - dt.timedelta(seconds=x)))
            if w and w.get('frames'):
                f = w['frames'][-1]
                ts = dt.datetime.fromisoformat(f['rfc460Timestamp'].replace('Z', '+00:00'))
                OUT.write(json.dumps(dict(now=n.isoformat(), game_id=gid, league=lg, x=x, ts=f['rfc460Timestamp'],
                                          age=(n - ts).total_seconds(), state=f['gameState'],
                                          kills=[f['blueTeam']['totalKills'], f['redTeam']['totalKills']], offsets=offs)) + '\n')
                OUT.flush()
        time.sleep(2)
