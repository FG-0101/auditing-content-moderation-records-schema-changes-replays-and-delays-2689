"""Conditional fixed-routing baseline for the component-versus-total regression.
Run after 04_capacity.py. This checks arithmetic and a specified Bernoulli routing
model; it does not identify arrivals, staffing, or a causal capacity mechanism.
"""
import json
import numpy as np
import pandas as pd
s=pd.read_parquet('data/derived/capacity_sample.parquet')
x=s.log_nn.to_numpy(float).copy()
# Alternating orthogonal projections for the same two fixed-effect dimensions.
codes=[pd.factorize(s[c])[0] for c in ('pdow','month')]
for iteration in range(1000):
 old=x.copy()
 for c in codes:
  x-=np.bincount(c,weights=x)[c]/np.bincount(c)[c]
 if np.max(np.abs(x-old))<1e-12:break
else:raise RuntimeError('Fixed-effect projection did not converge')
for c in codes:assert np.max(np.abs(np.bincount(c,weights=x)/np.bincount(c)))<1e-10
den=x@x
obs=float(x@np.log1p(s.nn_human.to_numpy(float))/den)
pp=s.groupby('p').agg(h=('nn_human','sum'),n=('nn','sum'));p=s.p.map(pp.h/pp.n).to_numpy(float)
n=s.nn.to_numpy(np.int64);rng=np.random.default_rng(20261010);out=[]
for trial in range(1000):
 h=rng.binomial(n,p);out.append(float(x@np.log1p(h)/den))
z=np.array(out);pd.DataFrame({'draw':range(1000),'elasticity':z}).to_csv('results/route_null_draws.csv',index=False)
res=dict(seed=20261010,draws=1000,platforms=int(s.p.nunique()),days=len(s),observed=obs,null_mean=float(z.mean()),null_low=float(np.quantile(z,.025)),null_high=float(np.quantile(z,.975)),lower_tail=float((1+(z<=obs).sum())/1001),projection_iterations=iteration+1,assumption='fixed platform routing probability; independent Bernoulli records conditional on observed totals',interpretation='conditional model diagnostic, no capacity or causal identification')
json.dump(res,open('results/route_null.json','w'),indent=2)
with open('tex/route_null.tex','w') as f:
 f.write('\\section{A fixed-routing baseline}\\label{app:route-null}\n')
 f.write('The nonautomated count is a component of the total count used as its regressor. We therefore condition on every observed platform-day total and simulate independent Bernoulli route assignments with each platform\'s overall nonautomated probability. We fit the same platform-by-weekday and calendar-month fixed effects in each of 1,000 seeded draws. ')
 f.write(f'The observed elasticity is {obs:.3f}; the fixed-routing baseline has mean {z.mean():.3f} and a 95\\% simulation range from {np.quantile(z,.025):.3f} to {np.quantile(z,.975):.3f}. ')
 f.write('These comparisons quantify the contribution of component-total arithmetic under a fixed routing model. Differences from that model can arise from changing route composition, reporting practices or case selection as well as staffing; the counts do not separate these mechanisms.\n')
print(json.dumps(res,indent=2))
with open('tex/route_null_summary.tex','w') as f:
 f.write(f'The observed elasticity of {obs:.3f} lies below the fixed-routing mean of {z.mean():.3f}, whose 95\\% simulation range is {np.quantile(z,.025):.3f}--{np.quantile(z,.975):.3f} over 1,000 draws. This conditional comparison measures departure from fixed routing; it does not identify staffing or notice arrivals.\n')
