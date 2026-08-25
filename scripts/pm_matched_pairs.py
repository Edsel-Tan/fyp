import os, requests, concurrent.futures as cf, collections, json
B="https://lum.id/findata"; s=requests.Session()
s.headers["Authorization"]=f"Bearer {os.environ['LUMID_TOKEN']}"
s.mount("https://",requests.adapters.HTTPAdapter(pool_connections=16,pool_maxsize=16))
def g(p,_to=50,**kw):
    try: r=s.get(B+p,params=kw,timeout=_to)
    except Exception as e: return -1,type(e).__name__
    try: return r.status_code,r.json()
    except Exception: return r.status_code,r.text[:80]

mkts={}
for q in ["trump","bitcoin","election","fed","nba","israel","ukraine","nfl","ethereum","china","government","inflation","gdp","recession","tariff"]:
    st,d=g("/prediction-markets/markets/search",q=q,limit=40)
    if st==200 and isinstance(d,list):
        for m in d:
            if m.get("venue")=="polymarket": mkts[m["market_id"]]=m
sample=list(mkts.values())
print("polymarket markets sampled:",len(sample))

def pp(m):
    st,d=g(f"/prediction-markets/matched-pairs/polymarket/{m['market_id']}",limit=10)
    if st!=200 or not isinstance(d,list): return (m,None)
    return (m,d)
out=[]
with cf.ThreadPoolExecutor(12) as ex:
    for r in ex.map(pp,sample): out.append(r)

kinds=collections.Counter(); venues=collections.Counter()
nexact=0; withany=0; exact_examples=[]
for m,d in out:
    if not d: continue
    withany+=1
    for x in d:
        kinds[x.get("match_kind")]+=1; venues[x.get("other_venue")]+=1
    ex_=[x for x in d if x.get("match_kind")=="exact"]
    if ex_:
        nexact+=1
        if len(exact_examples)<10: exact_examples.append((m,ex_[0]))
print(f"markets with >=1 match: {withany}/{len(sample)} ({withany/len(sample):.0%})")
print(f"markets with an EXACT cross-venue match: {nexact} ({nexact/len(sample):.0%})")
print("match_kind counts:",dict(kinds))
print("other_venue counts:",dict(venues))
print("\nexact-match examples (polymarket -> kalshi):")
for m,x in exact_examples:
    print(f"  sim={x['similarity']:5.1f} {str(m['title'])[:52]:54s} -> {x['other_id']}")
