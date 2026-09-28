"""Thin client for the public lolesports web APIs (schedule, event details, livestats)."""
import time, json, requests

KEY = "0TvQnueqKa5mxJntVWt0w4LpLfEkrV1Ta8rQBb9Z"  # public key embedded in lolesports.com
GW = "https://esports-api.lolesports.com/persisted/gw"
FEED = "https://feed.lolesports.com/livestats/v1"
S = requests.Session()
S.headers["x-api-key"] = KEY


def get(url, params=None, tries=6):
    for i in range(tries):
        try:
            r = S.get(url, params=params, timeout=30)
            if r.status_code == 200:
                return r.json() if r.content else None
            if r.status_code in (204, 404):
                return None
        except requests.RequestException:
            pass
        time.sleep(1.5 * 2 ** i)
    return None


def leagues():
    return get(f"{GW}/getLeagues", {"hl": "en-US"})["data"]["leagues"]


def schedule_all(league_id):
    """All events for a league, walking the 'older' page tokens."""
    out, tok = [], None
    while True:
        p = {"hl": "en-US", "leagueId": league_id}
        if tok:
            p["pageToken"] = tok
        d = get(f"{GW}/getSchedule", p)
        if not d:
            break
        s = d["data"]["schedule"]
        out += s["events"]
        tok = s["pages"]["older"]
        if not tok:
            break
    return out


def event_details(match_id):
    d = get(f"{GW}/getEventDetails", {"hl": "en-US", "id": match_id})
    return d and d["data"]["event"]


def window(game_id, starting=None):
    return get(f"{FEED}/window/{game_id}", {"startingTime": starting} if starting else None)


def details(game_id, starting=None):
    return get(f"{FEED}/details/{game_id}", {"startingTime": starting} if starting else None)
