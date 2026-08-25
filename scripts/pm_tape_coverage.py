import os, requests, concurrent.futures as cf, collections
B="https://lum.id/findata"; s=requests.Session()
s.headers["Authorization"]=f"Bearer {os.environ['LUMID_TOKEN']}"
a=requests.adapters.HTTPAdapter(pool_connections=16,pool_maxsize=16); s.mount("https://",a)
def g(p,_to=60,**kw):
    try: r=s.get(B+p,params=kw,timeout=_to)
    except Exception as e: return -1,type(e).__name__
    try: return r.status_code,r.json()
    except Exception: return r.status_code,r.text[:100]

# build a varied market sample
mkts={}
for q in ["trump","bitcoin","election","fed","nba","israel","ai","ukraine","nfl","ethereum","china","openai"]:
    st,d=g("/prediction-markets/markets/search",q=q,limit=40)
    if st==200 and isinstance(d,list):
        for m in d:
            if m.get("venue")=="polymarket": mkts[m["market_id"]]=m
print("sampled polymarket markets:",len(mkts))
sample=sorted(mkts.values(),key=lambda m:-(m.get("volume") or 0))[:24]

def probe(m):
    cid=m["market_id"]
    st,d=g(f"/prediction-markets/trades/polymarket/{cid}",limit=1000)
    if st!=200 or not isinstance(d,list) or not d: return (m,None,None,0,0)
    ts=sorted(x["ts"] for x in d)
    return (m,ts[0],ts[-1],len(d),len({x["taker"] for x in d}))

res=[]
with cf.ThreadPoolExecutor(10) as ex:
    for r in ex.map(probe,sample): res.append(r)

print("\n== trade tape per market (limit=1000, newest-N) ==")
firsts=[]
for m,t0,t1,n,tk in sorted(res,key=lambda r:-(r[0].get('volume') or 0))[:20]:
    if not t0: print(f"  {str(m['title'])[:44]:46s} NO TRADES"); continue
    firsts.append(t0)
    print(f"  {str(m['title'])[:44]:46s} n={n:4d} {t0[:16]} -> {t1[:16]} takers={tk}")
if firsts:
    print("\nearliest ts seen across sample:",min(firsts))
    print("latest  ts seen across sample:",max(r[2] for r in res if r[2]))
    c=collections.Counter(f[:7] for f in firsts)
    print("first-trade month histogram:",dict(sorted(c.items())))
