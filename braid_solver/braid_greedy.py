import sys, math, random
from itertools import combinations
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from braid import generate_candidates, paths_intersect

def build_adj(cands):
    m=len(cands); adj=[set() for _ in range(m)]
    for i in range(m):
        ci=cands[i]
        for j in range(i+1,m):
            if not paths_intersect(ci,cands[j]): adj[i].add(j); adj[j].add(i)
    return adj

def greedy_from(adj,s):
    clique={s}; cand=set(adj[s])
    while cand:
        v=max(cand,key=lambda u:len(adj[u]&cand))
        clique.add(v); cand&=adj[v]
    return len(clique)

def best_greedy(adj,starts):
    return max(greedy_from(adj,s) for s in starts)

def coprime(fs): return all(math.gcd(a,b)==1 for a,b in combinations(fs,2))

print("conjecture n*2^(n-1): 4D=32, 5D=80  (greedy gives a LOWER bound on the true max)")
print("\n4+1:")
for fs in [(3,5,7,8),(3,4,5,7),(3,5,8,11),(3,7,10,13),(3,4,7,11)]:
    if not coprime(fs): continue
    c=generate_candidates(4,list(fs),permutation=True); adj=build_adj(c)
    print(f"  {fs}: >= {best_greedy(adj,range(len(c)))}  ({len(c)} cand)",flush=True)
print("\n5+1:")
for fs in [(3,5,7,8,11),(3,4,5,7,11),(3,5,7,11,13)]:
    if not coprime(fs): continue
    c=generate_candidates(5,list(fs),permutation=True); adj=build_adj(c)
    random.seed(1); starts=random.sample(range(len(c)),min(300,len(c)))
    print(f"  {fs}: >= {best_greedy(adj,starts)}  ({len(c)} cand)",flush=True)
