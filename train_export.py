"""Reproduce the frozen split and export the selected baseline and Ridge model.
Run: python train_export.py
Training uses train + validation only. Test labels are never used for fitting.
"""
from pathlib import Path
import hashlib, json, re, unicodedata, html
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

ROOT = Path(__file__).resolve().parent
SEED = 42

def title_key(title):
    s = re.sub(r'\s*\(\d{4}\)\s*$', '', str(title))
    s = unicodedata.normalize('NFKD', s).encode('ascii', 'ignore').decode().lower()
    s = re.sub(r',\s*(the|a|an)$', r' \1', s)
    return re.sub(r'[^a-z0-9]', '', s)

def export_model():
    data = ROOT / 'data'
    manifest = json.loads((data/'SHA256.json').read_text())
    for file, digest in manifest.items():
        if hashlib.sha256((data/file).read_bytes()).hexdigest() != digest:
            raise ValueError('Dataset checksum mismatch: '+file)
    movies = pd.read_csv(data/'hotstar.csv', dtype={'hotstar_id': str})
    movies = movies.loc[movies['type'].astype(str).str.lower().eq('movie')].copy()
    movies = movies.dropna(subset=['hotstar_id', 'title'])
    for c in ['title', 'description', 'genre', 'age_rating']:
        movies[c] = movies[c].fillna('').astype(str).str.strip()
    movies['title'] = movies['title'].map(html.unescape)
    movies = movies.loc[movies.title.ne('')].drop_duplicates('hotstar_id').copy()
    movies['_t'] = movies.title.str.lower()
    movies = (movies.sort_values('hotstar_id')
                    .drop_duplicates(['_t', 'year', 'description'])
                    .drop(columns='_t').reset_index(drop=True))
    print('Movies after removing duplicate listings:', len(movies))
    assert len(movies) == 4557, 'Unexpected catalogue snapshot after deduplication.'
    movies['genre'] = movies.genre.replace('', 'Unknown')
    for c in ['year', 'running_time']:
        movies[c] = pd.to_numeric(movies[c], errors='coerce')
    movies.loc[movies.running_time.le(0), 'running_time'] = np.nan
    ml = pd.read_csv(data/'ml-latest-small/movies.csv')
    ratings = pd.read_csv(data/'ml-latest-small/ratings.csv')
    movies['title_key'] = movies.title.map(title_key)
    ml['title_key'] = ml.title.map(title_key)
    ml['year'] = pd.to_numeric(ml.title.str.extract(r'\((\d{4})\)$')[0], errors='coerce')
    left = movies.loc[movies.year.notna() & movies.title_key.ne('')]
    right = ml.loc[ml.year.notna() & ml.title_key.ne('')]
    left = left.loc[~left.duplicated(['title_key', 'year'], keep=False)]
    right = right.loc[~right.duplicated(['title_key', 'year'], keep=False)]
    items = left.merge(right[['movieId', 'title', 'title_key', 'year']],
        on=['title_key', 'year'], suffixes=('', '_movielens'), validate='one_to_one')
    items = items.sort_values('hotstar_id').reset_index(drop=True)
    items['item_idx'] = np.arange(len(items))
    interactions = ratings.merge(items[['movieId', 'hotstar_id', 'item_idx']],
        on='movieId', validate='many_to_one').sort_values(['userId', 'movieId']).reset_index(drop=True)
    rng = np.random.default_rng(SEED)
    val, test = [], []
    for u, group in interactions.groupby('userId', sort=True):
        if len(group) >= 5:
            order = rng.permutation(group.index.to_numpy())
            n = max(1, int(len(group)*0.2))
            val.extend(order[:n]); test.extend(order[n:2*n])
    train = interactions.drop(index=val+test)
    dev = pd.concat([train, interactions.loc[val]]).sort_index()
    test_frame = interactions.loc[test]
    assert set(dev.index).isdisjoint(test_frame.index)
    assert (len(items), len(interactions), len(dev), len(test_frame)) == (304, 8125, 6726, 1399)
    vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1,2), max_features=12000, sublinear_tf=True)
    text = items.description+' '+items.genre+' '+items.genre
    T = vectorizer.fit_transform(text)
    reducer = TruncatedSVD(n_components=20, random_state=SEED)
    E = normalize(reducer.fit_transform(T)).astype(np.float64)
    U, M = int(interactions.userId.max())+1, len(items)
    def state(ref):
        u = ref.userId.to_numpy(int); i = ref.item_idx.to_numpy(int); y = ref.rating.to_numpy(float)
        uc = np.bincount(u, minlength=U).astype(float); us = np.bincount(u, weights=y, minlength=U)
        ic = np.bincount(i, minlength=M).astype(float); isum = np.bincount(i, weights=y, minlength=M)
        mu = float(y.mean())
        A = csr_matrix((np.maximum(y-3,0), (u,i)), shape=(U,M))
        weight = np.asarray(A.sum(axis=1)).ravel()
        p = normalize(np.asarray(A@E)/np.maximum(weight[:,None],1e-12))
        um = np.divide(us, uc, out=np.full(U,mu), where=uc>0)
        return {'profile':p, 'uc':uc, 'um':um, 'ic':ic, 'im':(isum+10*mu)/(ic+10), 'mu':mu}
    def features(st, u, i):
        u=np.asarray(u,int);i=np.asarray(i,int);p=st['profile'][u];e=E[i]
        return np.column_stack([e,p,e*p,np.sum(e*p,axis=1),st['um'][u],
            np.log1p(st['uc'][u]),st['im'][i],np.log1p(st['ic'][i])])
    ref = dev.reset_index(drop=True)
    X = np.empty((len(ref),65))
    for fit_rows, held_rows in KFold(5,shuffle=True,random_state=SEED).split(ref):
        st = state(ref.iloc[fit_rows]); held=ref.iloc[held_rows]
        X[held_rows] = features(st,held.userId,held.item_idx)
    ridge = make_pipeline(StandardScaler(),Ridge(alpha=10))
    ridge.fit(X,ref.rating)
    st = state(dev)
    scalar, reg = ridge.steps[0][1],ridge.steps[1][1]
    result = []
    for i,row in items.iterrows():
        result.append({'id':str(row.hotstar_id),'title':row.title,'genre':row.genre,
            'year':int(row.year),'minutes':None if pd.isna(row.running_time) else int(row.running_time),
            'age_rating':row.age_rating,'description':row.description,
            'count':int(st['ic'][i]),'mean':float(st['im'][i]),'embedding':E[i].tolist(),'index':int(i)})
    # Anonymous aggregate profiles; no names, contact details or individual rating lists.
    profiles=[]
    for user,g in dev.groupby('userId',sort=True):
        user=int(user)
        profiles.append({'id':str(user),'count':int(st['uc'][user]),'mean':float(st['um'][user]),
            'vector':st['profile'][user].tolist(),'seen':[str(x) for x in g.hotstar_id]})
    selection=json.loads((ROOT/'evaluation/model_selection.json').read_text())
    assert selection['selected_model']=='Popularity' and selection['selected_supervised']=='Ridge', 'Deployment supports Popularity and Ridge; review changed selection before export.'
    metrics=pd.read_csv(ROOT/'evaluation/test_ranking_metrics.csv').to_dict(orient='records')
    validation_metrics=pd.read_csv(ROOT/'evaluation/validation_ranking_metrics.csv').to_dict(orient='records')
    artifact={'schema_version':1,'name':'MovieMate','primary_model':'Popularity','personalized_model':'Ridge',
        'fit_scope':'train_plus_validation','training_ratings':len(dev),'test_ratings':len(test_frame),
        'global_mean':st['mu'],'historical_movies':len(movies),'matched_movies':len(items),
        'eligible_movies':int((st['ic']>0).sum()),'seed':SEED,'score_formula':'smoothed_mean * count / (count + 10)',
        'items':result,'profiles':profiles,'ridge':{'coef':reg.coef_.tolist(),'intercept':float(reg.intercept_),
        'mean':scalar.mean_.tolist(),'scale':scalar.scale_.tolist()},'test_metrics':metrics,'validation_metrics':validation_metrics}
    output=ROOT/'artifacts';output.mkdir(exist_ok=True)
    path=output/'model.json';path.write_text(json.dumps(artifact,ensure_ascii=False,allow_nan=False,separators=(',',':')),encoding='utf-8')
    pred=np.clip(ridge.predict(features(st,test_frame.userId,test_frame.item_idx)),0.5,5)
    rmse=float(np.sqrt(mean_squared_error(test_frame.rating,pred)))
    expected_rmse = float(pd.read_csv(ROOT/'evaluation/test_rating_metrics.csv').set_index('Model').loc['Ridge','RMSE'])
    if abs(rmse-expected_rmse)>1e-8:
        raise AssertionError(f'Reproduction mismatch: Ridge RMSE {rmse}')
    # Export exact scikit-learn expectations for independent Python/JS inference tests.
    cases=[]
    for user in [1,68,380]:
        candidate=np.flatnonzero(st['ic']>0)
        expected=np.clip(ridge.predict(features(st,np.full(len(candidate),user),candidate)),0.5,5)
        cases.append({'user':str(user),'items':[result[i]['id'] for i in candidate], 'ridge':expected.tolist()})
    (output/'parity_cases.json').write_text(json.dumps(cases))
    audit={'model_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'ridge_reproduced_test_rmse':rmse,
        'fit_ratings':len(dev),'test_ratings_excluded':len(test_frame),'candidate_movies':artifact['eligible_movies']}
    (output/'export_audit.json').write_text(json.dumps(audit,indent=2))
    print(json.dumps(audit,indent=2))
    return artifact

if __name__=='__main__':
    export_model()
