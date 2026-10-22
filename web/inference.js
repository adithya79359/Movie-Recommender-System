(function(root){
'use strict';
function score(model,item,mode,vector,mean,count){
 if(mode==='Popularity')return item.mean*item.count/(item.count+10);
 const e=item.embedding,product=e.map((v,i)=>v*vector[i]);
 const x=[...e,...vector,...product,product.reduce((a,b)=>a+b,0),mean,Math.log1p(count),item.mean,Math.log1p(item.count)];
 const r=model.ridge;
 return Math.max(.5,Math.min(5,r.intercept+x.reduce((s,v,i)=>s+(v-r.mean[i])/r.scale[i]*r.coef[i],0)));
}
function profile(model,request){
 const ids=new Map(model.items.map(x=>[x.id,x]));
 if(request.profile_id){const p=model.profiles.find(x=>x.id===String(request.profile_id));if(!p)throw Error('Unknown profile.');return {vector:p.vector,mean:p.mean,count:p.count,seen:new Set(p.seen)};}
 const ratings=request.ratings||{};
 if(Array.isArray(ratings)||typeof ratings!=='object'||ratings===null)throw Error('Ratings must be an object.');
 if(Object.keys(ratings).length>50)throw Error('Choose at most 50 rated movies.');
 let vector=Array(20).fill(0),sum=0,count=0,seen=new Set();
 for(const [id,value] of Object.entries(ratings)){
  const item=ids.get(id),v=Number(value);
  if(!item)throw Error('Unknown movie ID: '+id);
  if(typeof value==='boolean'||value===null||value===''||!Number.isFinite(v)||v<.5||v>5)throw Error('Rating must be between 0.5 and 5.');
  sum+=v;count++;seen.add(id);vector=vector.map((x,j)=>x+Math.max(v-3,0)*item.embedding[j]);
 }
 const norm=Math.sqrt(vector.reduce((s,x)=>s+x*x,0));if(norm)vector=vector.map(x=>x/norm);
 return {vector,mean:count?sum/count:model.global_mean,count,seen};
}
function recommend(model,request={}){
 const mode=request.model||'Popularity',n=request.n===undefined?10:request.n;
 if(!['Popularity','Ridge'].includes(mode))throw Error('Unknown model.');
 if(!Number.isInteger(n)||n<1||n>50)throw Error('n must be an integer from 1 to 50.');
 const p=profile(model,request);if(request.exclude&&!Array.isArray(request.exclude))throw Error('exclude must be a list.');
 for(const x of request.exclude||[])p.seen.add(String(x));
 const effective=mode==='Ridge'&&!p.count?'Popularity':mode;
 let maximum=null;if(request.max_minutes!==undefined&&request.max_minutes!==null&&request.max_minutes!==''){
  maximum=Number(request.max_minutes);if(!Number.isFinite(maximum)||maximum<=0)throw Error('Maximum runtime must be positive.');
 }
 const results=model.items.filter(x=>x.count>0&&!p.seen.has(x.id)&&(!request.genre||request.genre===x.genre)&&(maximum===null||(x.minutes!==null&&x.minutes<=maximum))).map(item=>({
  ...item,score:score(model,item,effective,p.vector,p.mean,p.count),reason:effective==='Popularity'?'Highly rated across the reference audience':'Predicted preference from your profile and movie features'
 })).sort((a,b)=>b.score-a.score||a.index-b.index);
 return {model:effective,requested_model:mode,profile_ratings:p.count,eligible_after_filters:results.length,results:results.slice(0,n),fallback:mode!==effective};
}
const api={score,profile,recommend};if(typeof module!=='undefined')module.exports=api;root.MovieInference=api;
})(typeof globalThis!=='undefined'?globalThis:this);
