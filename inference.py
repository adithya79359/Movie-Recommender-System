"""Dependency-free inference from the fitted JSON artifact."""
from pathlib import Path
import json, math

class Recommender:
    def __init__(self, path=None):
        path=Path(path) if path else Path(__file__).parent/'artifacts/model.json'
        self.model=json.loads(path.read_text(encoding='utf-8'))
        self.items={m['id']:m for m in self.model['items']}
        self.profiles={p['id']:p for p in self.model['profiles']}

    def profile(self, request):
        pid=str(request.get('profile_id') or '')
        if pid:
            if pid not in self.profiles:raise ValueError('Unknown profile.')
            p=self.profiles[pid]
            return p['vector'],p['mean'],p['count'],set(p['seen'])
        ratings=request.get('ratings',{})
        if not isinstance(ratings,dict):raise ValueError('Ratings must be an object of movie IDs and ratings.')
        if len(ratings)>50:raise ValueError('Choose at most 50 rated movies.')
        vector=[0.0]*20; values=[];seen=set()
        for mid,rating in ratings.items():
            if mid not in self.items:raise ValueError('Unknown movie ID: '+mid)
            if isinstance(rating,bool):raise ValueError('Rating must be a number between 0.5 and 5.')
            try:v=float(rating)
            except (TypeError,ValueError):raise ValueError('Rating must be a number between 0.5 and 5.')
            if not math.isfinite(v) or not 0.5<=v<=5:raise ValueError('Rating must be between 0.5 and 5.')
            values.append(v);seen.add(mid)
            for j,e in enumerate(self.items[mid]['embedding']):vector[j]+=max(v-3,0)*e
        norm=math.sqrt(sum(x*x for x in vector))
        if norm:vector=[x/norm for x in vector]
        return vector,sum(values)/len(values) if values else self.model['global_mean'],len(values),seen

    def score(self,item,mode,vector,mean,count):
        if mode=='Popularity':return item['mean']*item['count']/(item['count']+10)
        e=item['embedding'];product=[a*b for a,b in zip(e,vector)]
        features=e+vector+product+[sum(product),mean,math.log1p(count),item['mean'],math.log1p(item['count'])]
        ridge=self.model['ridge']
        y=ridge['intercept']+sum((x-m)/s*c for x,m,s,c in zip(features,ridge['mean'],ridge['scale'],ridge['coef']))
        return min(5.0,max(0.5,y))

    def recommend(self,request):
        if not isinstance(request,dict):raise ValueError('Request must be a JSON object.')
        mode=request.get('model','Popularity')
        if mode not in ['Popularity','Ridge']:raise ValueError('Unknown model.')
        n=request.get('n',10)
        if isinstance(n,bool) or not isinstance(n,int) or not 1<=n<=50:raise ValueError('n must be an integer from 1 to 50.')
        vector,mean,count,seen=self.profile(request)
        excluded=request.get('exclude',[])
        if not isinstance(excluded,list):raise ValueError('exclude must be a list.')
        seen.update(str(x) for x in excluded)
        effective=mode if mode=='Popularity' or count else 'Popularity'
        genre=str(request.get('genre') or '')
        max_time=request.get('max_minutes')
        if max_time not in [None,'']:
            try:max_time=float(max_time)
            except (ValueError,TypeError):raise ValueError('Maximum runtime must be a positive number.')
            if not math.isfinite(max_time) or max_time<=0:raise ValueError('Maximum runtime must be positive.')
        else:max_time=None
        results=[]
        for item in self.model['items']:
            if item['count']==0 or item['id'] in seen:continue
            if genre and genre!=item['genre']:continue
            if max_time is not None and (item['minutes'] is None or item['minutes']>max_time):continue
            result={k:item[k] for k in ['id','title','description','genre','year','minutes','age_rating','count']}
            result['score']=self.score(item,effective,vector,mean,count)
            result['reason']='Highly rated across the reference audience' if effective=='Popularity' else 'Predicted preference from your profile and movie features'
            results.append(result)
        results.sort(key=lambda x:(-x['score'],self.items[x['id']]['index']))
        return {'model':effective,'requested_model':mode,'profile_ratings':count,
            'eligible_after_filters':len(results),'results':results[:n],
            'notice':'Historical catalogue; current streaming availability is unverified.',
            'fallback':mode!=effective}
