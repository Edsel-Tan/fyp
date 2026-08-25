import os, requests, datetime, json, collections, concurrent.futures as cf
B="https://lum.id/findata"; s=requests.Session()
s.headers["Authorization"]=f"Bearer {os.environ['LUMID_TOKEN']}"
s.mount("https://",requests.adapters.HTTPAdapter(pool_connections=16,pool_maxsize=16))
def g(p,_to=60,**kw):
    try: r=s.get(B+p,params=kw,timeout=_to)
    except Exception as e: return -1,type(e).__name__
    try: return r.status_code,r.json()
    except Exception: return r.status_code,None

# find RESOLVED markets whose end_date is inside the tape window
mkts={}
for q in ["fed","nba","election","bitcoin","trump","peru","nfl","israel","ukraine","inflation","oscar","cpi","jobs","game"]:
    st,d=g("/prediction-markets/markets/search",q=q,limit=50)
    if st==200 and isinstance(d,list):
        for m in d:
            if m.get("venue")!="polymarket": continue
            ed=str(m.get("end_date") or "")[:10]
            if "2026-05-25" <= ed <= "2026-08-14": mkts[m["market_id"]]=m
print("candidate resolved-in-window markets:",len(mkts))

def resolve(m):
    st,det=g(f"/prediction-markets/markets/polymarket/{m['market_id']}")
    if st!=200 or not isinstance(det,dict): return None
    op=det.get("outcome_prices") or []
    toks=det.get("clob_token_ids") or []
    if len(op)<2 or len(toks)<2: return None
    try: p=[float(x) for x in op]
    except: return None
    if max(p)<0.97 or min(p)>0.03: return None      # not cleanly resolved
    win = toks[0] if p[0]>p[1] else toks[1]
    return dict(cid=m["market_id"], q=det.get("question"), win=win, yes=toks[0], no=toks[1], p=p)

res=[]
with cf.ThreadPoolExecutor(10) as ex:
    for r in ex.map(resolve,list(mkts.values())):
        if r: res.append(r)
print("cleanly resolved:",len(res))

def tape(r):
    allt={}
    cur=datetime.datetime(2026,5,21,tzinfo=datetime.timezone.utc)
    end=datetime.datetime(2026,8,16,tzinfo=datetime.timezone.utc)
    step=datetime.timedelta(days=4)
    while cur<end:
        nxt=min(cur+step,end)
        st,d=g(f"/prediction-markets/trades/polymarket/{r['cid']}",
               **{"from":cur.strftime("%Y-%m-%dT%H:%M:%SZ"),"to":nxt.strftime("%Y-%m-%dT%H:%M:%SZ")},limit=1000)
        if st==200 and isinstance(d,list):
            for t in d: allt[t["trade_id"]]=t
        cur=nxt
    return r,list(allt.values())

rows=[]
with cf.ThreadPoolExecutor(6) as ex:
    for r,tr in ex.map(tape,res[:40]):
        if len(tr)<150: continue
        for t in tr:
            t["notional"]=t["price"]*t["size"]
            payoff = 1.0 if t["token_id"]==r["win"] else 0.0
            t["pnl"] = (payoff-t["price"])*t["size"] if t["side"]=="BUY" else (t["price"]-payoff)*t["size"]
        rows.append((r,tr))
print("markets with >=150 captured trades:",len(rows))
json.dump([[r,[{k:t[k] for k in ('taker','notional','pnl','ts')} for t in tr]] for r,tr in rows],open("multi.json","w"))

print(f"\n{'market':<50} {'trades':>7} {'notional$':>12} {'top3%ROI':>9} {'bot90%ROI':>10}")
agg_top=[0.0,0.0]; agg_bot=[0.0,0.0]
for r,tr in rows:
    bt=collections.defaultdict(lambda:[0.0,0.0])
    for t in tr:
        b=bt[t["taker"]]; b[0]+=t["notional"]; b[1]+=t["pnl"]
    rank=sorted(bt.items(),key=lambda kv:-kv[1][0]); n=len(rank)
    k=max(1,3*n//100)
    tno=sum(v[0] for _,v in rank[:k]); tpl=sum(v[1] for _,v in rank[:k])
    bno=sum(v[0] for _,v in rank[n//10:]); bpl=sum(v[1] for _,v in rank[n//10:])
    agg_top[0]+=tno; agg_top[1]+=tpl; agg_bot[0]+=bno; agg_bot[1]+=bpl
    print(f"{str(r['q'])[:48]:<50} {len(tr):>7} {sum(t['notional'] for t in tr):>12,.0f} "
          f"{tpl/tno if tno else 0:>+8.2%} {bpl/bno if bno else 0:>+9.2%}")
print(f"\nPOOLED top3%  ROI = {agg_top[1]/agg_top[0]:+.2%}  on ${agg_top[0]:,.0f}")
print(f"POOLED bot90% ROI = {agg_bot[1]/agg_bot[0]:+.2%}  on ${agg_bot[0]:,.0f}")
