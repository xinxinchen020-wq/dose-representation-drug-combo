"""Response-free dose sampling and training-only prediction baselines."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

PROTOCOL='RANDOM_IN_RANGE'
VERSION='known-curve-random-noncorners-v1'
DRAW_SEED=20261016
GEOMETRY_COLUMNS=['PAIR','CELLNAME','CELL_IDX','DRUG1_IDX','DRUG2_IDX','CONC1_LOG','CONC2_LOG']

def grid_info(frame):
 g=frame.groupby(['PAIR','CELLNAME'],sort=False)
 r1=g.CONC1_LOG.rank(method='dense');r2=g.CONC2_LOG.rank(method='dense')
 n1=g.CONC1_LOG.transform('nunique');n2=g.CONC2_LOG.transform('nunique')
 unique=~frame.duplicated(['PAIR','CELLNAME','CONC1_LOG','CONC2_LOG'])
 count=unique.groupby([frame.PAIR,frame.CELLNAME],sort=False).transform('sum')
 finite=pd.Series(np.isfinite(frame[['CONC1_LOG','CONC2_LOG']].to_numpy()).all(axis=1),index=frame.index)
 all_finite=finite.groupby([frame.PAIR,frame.CELLNAME],sort=False).transform('all')
 complete=((n1*n2==count)&(n1>=3)&(n2>=3)&all_finite).to_numpy()
 return tuple(x.fillna(0).to_numpy(dtype=np.int64) for x in [r1,r2,n1,n2])+(complete,)

def sampled_positions(dataset,pair,n1,n2,draw_seed=DRAW_SEED):
 """One shared rank pattern for every cell of the same pair and grid shape."""
 k=(1 if n1%2 else 2)*(1 if n2%2 else 2)
 candidates=[]
 for i in range(1,n1+1):
  for j in range(1,n2+1):
   if i in [1,n1] and j in [1,n2]:continue
   payload=json.dumps([VERSION,int(draw_seed),dataset,str(pair),int(n1),int(n2),i,j],ensure_ascii=False,separators=(',',':')).encode()
   candidates.append((hashlib.sha256(payload).digest(),i,j))
 assert len(candidates)>=k
 return [(i,j) for _,i,j in sorted(candidates)[:k]]

def make_split(frame,dataset,pair_groups,draw_seed=DRAW_SEED):
 """All repeated observations of the same exact dose point move together."""
 r1,r2,n1,n2,complete=grid_info(frame);pairs=frame.PAIR.astype(str).to_numpy()
 unit,units=pd.factorize(pd.MultiIndex.from_arrays([pairs,n1,n2]),sort=False)
 width=int(max(n1.max(),n2.max()))+1;lookup=np.zeros((len(units),width*width),dtype=bool)
 for code,(pair,a,b) in enumerate(units):
  if a<3 or b<3:continue
  for i,j in sampled_positions(dataset,pair,int(a),int(b),draw_seed):lookup[code,i*width+j]=True
 held=complete&lookup[unit,r1*width+r2]
 val_pairs=set(map(str,pair_groups['validation_pairs']));test_pairs=set(map(str,pair_groups['test_pairs']));assert not val_pairs&test_pairs
 eligible=set(pairs[held]);assert eligible==val_pairs|test_pairs
 in_val=np.isin(pairs,list(val_pairs));idx=np.arange(len(frame),dtype=np.int64)
 interior=(r1>1)&(r1<n1)&(r2>1)&(r2<n2)
 corner=((r1==1)|(r1==n1))&((r2==1)|(r2==n2))
 assert not np.any(held&corner)
 return (idx[~held],idx[held&in_val],idx[held&~in_val]),dict(held=held,interior=interior,corner=corner,complete=complete,r1=r1,r2=r2,n1=n1,n2=n2)

def load_frozen_split(directory,n_rows):
 directory=Path(directory);metadata=json.loads((directory/'split_audit.json').read_text(encoding='utf-8'))
 assert metadata['split_version']==VERSION and metadata['n_rows']==n_rows
 with np.load(directory/'split_indices.npz') as z:indices=tuple(z[name].copy() for name in ['train','val','test'])
 assert sum(map(len,indices))==n_rows and len(np.unique(np.concatenate(indices)))==n_rows
 return indices,metadata

def training_only_baselines(frame,train_indices,target_indices):
 """Curve mean, and bracketing axis interpolation with a corner fallback.

 Only training responses are read. Target rows supply keys and coordinates only.
 Each axis uses the closest lower and upper measured training doses on the same
 row/column; average both estimates when available. If neither brackets, use
 bilinear interpolation from the four guaranteed training corners.
 """
 keys=['PAIR','CELLNAME'];train=frame.iloc[train_indices][keys+['CONC1_LOG','CONC2_LOG','PERCENTGROWTH']]
 train=train.groupby(keys+['CONC1_LOG','CONC2_LOG'],sort=False,as_index=False).PERCENTGROWTH.mean()
 target=frame.iloc[target_indices][keys+['CONC1_LOG','CONC2_LOG']]
 groups={k:g for k,g in train.groupby(keys,sort=False)};means=[];interpolated=[];routes=[]
 for key,x,y in zip(zip(target.PAIR,target.CELLNAME),target.CONC1_LOG,target.CONC2_LOG):
  g=groups[key];xs=g.CONC1_LOG.to_numpy();ys=g.CONC2_LOG.to_numpy();values=g.PERCENTGROWTH.to_numpy();mean=float(values.mean());means.append(mean)
  predictions=[]
  for coordinates,same in [(xs,ys==y),(ys,xs==x)]:
   t=x if coordinates is xs else y
   lower=np.flatnonzero(same&(coordinates<t));upper=np.flatnonzero(same&(coordinates>t))
   if not len(lower) or not len(upper):continue
   a=lower[np.argmax(coordinates[lower])];b=upper[np.argmin(coordinates[upper])]
   w=(t-coordinates[a])/(coordinates[b]-coordinates[a]);predictions.append(float((1-w)*values[a]+w*values[b]))
  if predictions:interpolated.append(float(np.mean(predictions)));routes.append(len(predictions));continue
  x0,x1=xs.min(),xs.max();y0,y1=ys.min(),ys.max();assert x0<x1 and y0<y1 and x0<=x<=x1 and y0<=y<=y1
  corners=[]
  for a,b in [(x0,y0),(x1,y0),(x0,y1),(x1,y1)]:
   match=np.flatnonzero((xs==a)&(ys==b));assert len(match)==1;corners.append(values[match[0]])
  tx=(x-x0)/(x1-x0);ty=(y-y0)/(y1-y0)
  interpolated.append(float((1-tx)*(1-ty)*corners[0]+tx*(1-ty)*corners[1]+(1-tx)*ty*corners[2]+tx*ty*corners[3]));routes.append(0)
 return dict(curve_mean=np.asarray(means),axis_interpolation=np.asarray(interpolated),axis_routes=np.asarray(routes))
