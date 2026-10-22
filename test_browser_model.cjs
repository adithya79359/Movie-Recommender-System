const fs=require('fs'),assert=require('assert'),path=require('path');
const model=JSON.parse(fs.readFileSync(path.join(__dirname,'artifacts/model.json'),'utf8'));
const engine=require('./web/inference.js');
const cases=JSON.parse(fs.readFileSync(path.join(__dirname,'artifacts/parity_cases.json'),'utf8'));
let checked=0,maxError=0;
for(const c of cases){const p=engine.profile(model,{profile_id:c.user});c.items.forEach((id,j)=>{const actual=engine.score(model,model.items.find(x=>x.id===id),'Ridge',p.vector,p.mean,p.count);const e=Math.abs(actual-c.ridge[j]);assert(e<1e-9);maxError=Math.max(maxError,e);checked++;});}
const requests=[{}, {model:'Ridge',profile_id:'380'},{model:'Ridge',ratings:{'1660000038':5,'1770000948':4}},{genre:'Comedy',max_minutes:120},{genre:'none'},{model:'Ridge'}];
const py=JSON.parse(fs.readFileSync(path.join(__dirname,'artifacts/python_predictions.json'),'utf8'));
requests.forEach((request,j)=>{const r=engine.recommend(model,request);assert.deepStrictEqual(r.results.map(x=>x.id),py[j].results.map(x=>x.id));r.results.forEach((x,i)=>assert(Math.abs(x.score-py[j].results[i].score)<1e-9));});
for(const request of [{model:'bad'},{n:0},{profile_id:'unknown'},{ratings:{'1660000038':6}},{max_minutes:-1}])assert.throws(()=>engine.recommend(model,request));
const report={passed:true,sklearn_score_comparisons:checked,python_js_request_comparisons:requests.length,max_absolute_error:maxError,dom_browser_tested:false};
fs.writeFileSync(path.join(__dirname,'artifacts/javascript_test_results.json'),JSON.stringify(report,null,2));console.log(report);
