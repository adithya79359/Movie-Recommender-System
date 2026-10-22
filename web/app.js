'use strict';
let model,apiAvailable=false,lastResults=[],ratings={},requestSerial=0;
const $=id=>document.getElementById(id);
function esc(s){return String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));}
function switchTab(which){const discover=which==='discover';$('discover-view').hidden=!discover;$('evaluation-view').hidden=discover;for(const name of ['discover','evaluation']){$(name+'-tab').classList.toggle('active',which===name);$(name+'-tab').setAttribute('aria-selected',String(which===name));}}
$('discover-tab').onclick=()=>switchTab('discover');$('evaluation-tab').onclick=()=>switchTab('evaluation');
function renderRatings(){
 $('rated-movies').innerHTML=Object.keys(ratings).map(id=>{const item=model.items.find(x=>x.id===id);return `<div class="rated-row"><span>${esc(item.title)}</span><select aria-label="Rating for ${esc(item.title)}" data-rate="${esc(id)}">${[.5,1,1.5,2,2.5,3,3.5,4,4.5,5].map(r=>`<option value="${r}" ${r===ratings[id]?'selected':''}>${r} ★</option>`).join('')}</select><button data-remove="${esc(id)}" aria-label="Remove ${esc(item.title)}">×</button></div>`}).join('');
 for(const node of document.querySelectorAll('[data-rate]'))node.onchange=()=>{ratings[node.dataset.rate]=Number(node.value);};
 for(const node of document.querySelectorAll('[data-remove]'))node.onclick=()=>{delete ratings[node.dataset.remove];renderRatings();};
}
$('movie-search').oninput=()=>{if(!model)return;const q=$('movie-search').value.trim().toLowerCase();$('search-results').innerHTML=q?model.items.filter(x=>x.title.toLowerCase().includes(q)&&!(x.id in ratings)).slice(0,7).map(x=>`<button data-add="${x.id}">${esc(x.title)} <small>(${x.year})</small></button>`).join(''):'';for(const b of document.querySelectorAll('[data-add]'))b.onclick=()=>{if(Object.keys(ratings).length>=50){$('status').textContent='Choose at most 50 movies.';return;}ratings[b.dataset.add]=5;$('sample-profile').value='';$('movie-search').value='';$('search-results').innerHTML='';renderRatings();};};
function updateMethod(){const personal=$('method').value==='Ridge';$('personal-controls').hidden=!personal;$('method-note').textContent=personal?'Rate a few films, or choose an anonymous sample profile.':'Ranks by audience ratings and support. No personal history needed.';}
$('method').onchange=()=>{updateMethod();run();};$('genre').onchange=run;$('runtime').onchange=run;$('recommend').onclick=run;
$('sample-profile').onchange=()=>{if($('sample-profile').value){ratings={};renderRatings();}run();};
$('reset').onclick=()=>{ratings={};renderRatings();$('sample-profile').value='';$('method').value='Popularity';$('genre').value='';$('runtime').value='';$('movie-search').value='';$('search-results').innerHTML='';updateMethod();run();};
$('demo').onclick=()=>{$('method').value='Ridge';$('sample-profile').value='380';ratings={};renderRatings();updateMethod();run();};
async function run(){
 if(!model)return;const serial=++requestSerial;$('recommend').disabled=true;$('status').textContent='Finding your next watch…';
 const request={model:$('method').value,ratings,profile_id:$('sample-profile').value,genre:$('genre').value,max_minutes:$('runtime').value||null,n:10};
 try{let data;if(apiAvailable){const r=await fetch('api/recommend',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(request)});data=await r.json();if(!r.ok)throw Error(data.error||'Recommendation failed.');}else data=MovieInference.recommend(model,request);
  if(serial!==requestSerial)return;lastResults=data.results;$('download').disabled=!lastResults.length;
  $('result-title').textContent=data.model==='Popularity'?'Audience favourites':'Picked for your profile';$('result-kicker').textContent=data.model==='Popularity'?'VALIDATION-SELECTED MODEL':'PERSONALIZED RIDGE MODEL';
  $('status').textContent=data.fallback?'Add ratings or select a sample profile for personalization. Showing audience favourites for now.':`${data.results.length} recommendations · ${data.eligible_after_filters} films match your filters${data.model==='Ridge'?' · '+data.profile_ratings+' profile ratings':''}`;
  const colors=['#43543e','#5a4b3c','#3f5052','#5a4147','#4d4a62','#5e5838'];
  $('cards').innerHTML=data.results.length?data.results.map((x,i)=>`<article class="movie-card"><div class="movie-art" style="--card-color:${colors[i%colors.length]}"><span class="movie-number">${String(i+1).padStart(2,'0')}</span><span class="movie-year">${x.year}</span></div><div class="movie-body"><h3>${esc(x.title)}</h3><p class="movie-meta">${esc(x.genre)} · ${x.minutes===null?'Runtime unknown':x.minutes+' min'}</p><div class="movie-score"><strong>${Number(x.score).toFixed(2)}</strong> ${data.model==='Ridge'?'predicted rating / 5':'ranking score'}</div><p class="reason">${esc(x.reason)}</p><details><summary>About this movie</summary><p>${esc(x.description)}</p><p>${esc(x.age_rating)} · ${x.count} reference ratings</p></details></div></article>`).join(''):'<div class="empty">No movies match these filters.<br>Try another genre or a longer runtime.</div>';
 }catch(e){if(serial===requestSerial){$('status').textContent=e.message;$('cards').innerHTML='';lastResults=[];$('download').disabled=true;}}
 finally{if(serial===requestSerial)$('recommend').disabled=false;}
}
$('download').onclick=()=>{const safe=x=>{let s=String(x??'');if(/^[=+@-]/.test(s))s="'"+s;return '"'+s.replace(/"/g,'""')+'"';};const lines=[['Title','Year','Genre','Minutes','Score','Reason'],...lastResults.map(x=>[x.title,x.year,x.genre,x.minutes,x.score,x.reason])];const blob=new Blob([lines.map(row=>row.map(safe).join(',')).join('\r\n')],{type:'text/csv;charset=utf-8'});const url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download='MovieMate_recommendations.csv';a.click();URL.revokeObjectURL(url);};
async function init(){try{
 const r=await fetch('model.json');if(!r.ok)throw Error('Could not load the model artifact.');model=await r.json();
 for(const genre of [...new Set(model.items.map(x=>x.genre))].sort()){$('genre').add(new Option(genre,genre));}
 for(const p of model.profiles.filter(x=>x.count>=10))$('sample-profile').add(new Option(`Profile ${p.id} · ${p.count} rated films`,p.id));
 const val=new Map(model.validation_metrics.map(x=>[x.Model,x]));$('metrics').innerHTML=model.test_metrics.map(x=>`<tr><td>${esc(x.Model)}${x.Model==='Popularity'?' · selected':''}</td><td>${val.get(x.Model)['NDCG@10'].toFixed(4)}</td><td>${x['NDCG@10'].toFixed(4)}</td><td>${x['Precision@10'].toFixed(4)}</td><td>${x['Recall@10'].toFixed(4)}</td><td>${(x['Coverage@10']*100).toFixed(1)}%</td></tr>`).join('');
 try{const health=await fetch('api/health');if(health.ok&&(health.headers.get('content-type')||'').includes('application/json'))apiAvailable=(await health.json()).status==='ok';}catch(_){}
 if(apiAvailable)$('runtime-label').textContent='This local deployment calls the Python inference API. The fitted artifact is loaded once; no training runs during a recommendation.';
 await run();
}catch(e){$('status').textContent='Unable to start: '+e.message;$('recommend').disabled=true;}}
init();
