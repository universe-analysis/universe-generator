"""Faster exact max-clique (Tomita-style greedy-coloring bound) for the solver."""
from braid import generate_candidates, paths_intersect
from itertools import combinations

def build_adj(cands):
    m=len(cands); adj=[set() for _ in range(m)]
    for i,j in combinations(range(m),2):
        if not paths_intersect(cands[i],cands[j]):
            adj[i].add(j); adj[j].add(i)
    return adj

def max_clique(adj):
    n=len(adj); best=[]
    def color_bound(P):
        # greedy coloring; returns list of (vertex, color) sorted by color
        order=sorted(P,key=lambda v:len(adj[v]&P),reverse=True)
        colored=[]; classes=[]
        for v in order:
            placed=False
            for c,cls in enumerate(classes):
                if not (adj[v]&cls):
                    cls.add(v); colored.append((v,c+1)); placed=True; break
            if not placed:
                classes.append({v}); colored.append((v,len(classes)))
        colored.sort(key=lambda x:x[1])
        return colored
    def expand(R,P):
        nonlocal best
        if not P:
            if len(R)>len(best): best=R[:]
            return
        cb=color_bound(P)
        for v,c in reversed(cb):
            if len(R)+c<=len(best): return
            expand(R+[v], P & adj[v])
            P=P-{v}
    expand([], set(range(n)))
    return best

def solve_free(dims,freqs):
    cands=generate_candidates(dims,freqs,permutation=False)
    adj=build_adj(cands)
    cl=max_clique(adj)
    return len(cl),[cands[i] for i in cl]
