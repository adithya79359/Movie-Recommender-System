"""Run meaningful model export and API checks: python test_deployment.py."""
from pathlib import Path
import json, unittest
from inference import Recommender
from app import app

ROOT=Path(__file__).parent
class DeploymentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.engine=Recommender();cls.client=app.test_client()
    def test_health(self):
        r=self.client.get('/api/health');self.assertEqual(r.status_code,200);self.assertEqual(r.json['movies'],295)
    def test_page_and_artifact(self):
        self.assertEqual(self.client.get('/').status_code,200)
        self.assertEqual(self.client.get('/model.json').status_code,200)
    def test_sklearn_export_parity(self):
        cases=json.loads((ROOT/'artifacts/parity_cases.json').read_text())
        for c in cases:
            vector,mean,count,_=self.engine.profile({'profile_id':c['user']})
            for mid,expected in zip(c['items'],c['ridge']):
                self.assertAlmostEqual(self.engine.score(self.engine.items[mid],'Ridge',vector,mean,count),expected,places=10)
    def test_default_ranking(self):
        r=self.client.post('/api/recommend',json={})
        self.assertEqual(r.status_code,200);self.assertEqual(r.json['model'],'Popularity')
        self.assertEqual(len(r.json['results']),10)
        scores=[x['score'] for x in r.json['results']];self.assertEqual(scores,sorted(scores,reverse=True))
    def test_history_exclusion(self):
        r=self.engine.recommend({'model':'Ridge','profile_id':'380'})
        seen=set(self.engine.profiles['380']['seen'])
        self.assertFalse(seen & {x['id'] for x in r['results']})
    def test_filters(self):
        r=self.engine.recommend({'genre':'Comedy','max_minutes':120})
        self.assertTrue(all(x['genre']=='Comedy' and x['minutes']<=120 for x in r['results']))
    def test_empty(self):self.assertEqual(self.engine.recommend({'genre':'not-a-genre'})['results'],[])
    def test_fallback(self):self.assertTrue(self.engine.recommend({'model':'Ridge'})['fallback'])
    def test_custom_profile(self):
        mid='1660000038';r=self.engine.recommend({'model':'Ridge','ratings':{mid:5}})
        self.assertEqual(r['model'],'Ridge');self.assertNotIn(mid,[x['id'] for x in r['results']])
    def test_invalid_inputs(self):
        for payload in [None,[],{'model':'bad'},{'n':0},{'n':True},{'profile_id':'unknown'},
                        {'ratings':{'1660000038':8}},{'ratings':{'1660000038':True}},
                        {'ratings':{'unknown':5}},{'max_minutes':-1},{'exclude':'bad'}]:
            self.assertEqual(self.client.post('/api/recommend',json=payload).status_code,400)

if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(DeploymentTests))
    requests=[{}, {'model':'Ridge','profile_id':'380'}, {'model':'Ridge','ratings':{'1660000038':5,'1770000948':4}}, {'genre':'Comedy','max_minutes':120}, {'genre':'none'}, {'model':'Ridge'}]
    engine=Recommender()
    (ROOT/'artifacts/python_predictions.json').write_text(json.dumps([engine.recommend(x) for x in requests]))
    report={'python_tests':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'passed':result.wasSuccessful()}
    (ROOT/'artifacts/python_test_results.json').write_text(json.dumps(report,indent=2))
    raise SystemExit(0 if result.wasSuccessful() else 1)
