const $ = id => document.getElementById(id);
const messages = $("messages");

async function checkHealth(){
  try{
    const r=await fetch("/api/health"); const d=await r.json();
    $("status").textContent = d.groq_configured ? (d.gemini_configured ? "AI services configured" : "Groq ready · Gemini key missing") : "API keys required";
  }catch{$("status").textContent="Backend unavailable";}
}
function addMessage(role, text, data){
  $("emptyState").classList.add("hidden");
  const box=document.createElement("article"); box.className="message "+role;
  const meta=document.createElement("div"); meta.className="meta";
  if(role==="user"){
    meta.textContent="YOU";
  }else if(data?.intent){
    const chip=document.createElement("span");
    chip.className="intent-chip "+data.intent;
    chip.textContent="NLP INTENT · "+data.intent.toUpperCase();
    const confidence=document.createElement("span"); confidence.className="intent-sub";
    confidence.textContent=`${Math.round((data.intent_confidence||0)*100)}% confidence · ${data.intent_method||"classifier"}`;
    meta.append(chip,confidence);
  }else{
    meta.textContent="S3 AGENT";
  }
  const body=document.createElement("div"); body.className="answer"; body.textContent=text;
  box.append(meta,body);
  if(role==="assistant" && data){
    const bits=[];
    if(data.mode) bits.push(`Selected pipeline: ${data.mode}`);
    if(data.sql) bits.push(`Generated SQL:\n${data.sql}`);
    if(data.rows) bits.push(`Rows returned: ${data.row_count}\n${JSON.stringify(data.rows,null,2)}`);
    if(data.sources?.length) bits.push("Retrieved knowledge sources:\n"+data.sources.map(s=>`${s.source} · similarity ${s.similarity}`).join("\n"));
    if(bits.length){const details=document.createElement("details");details.className="details";const summary=document.createElement("summary");summary.textContent="Show how S3 Agent processed this question";const pre=document.createElement("pre");pre.textContent=bits.join("\n\n");details.append(summary,pre);box.append(details);}
  }
  messages.append(box);
  messages.scrollTop=messages.scrollHeight;
}
async function ask(q){
  q=q.trim(); if(!q)return;
  addMessage("user",q); $("question").value=""; $("send").disabled=true; $("send").textContent="Thinking…";
  const typing=document.createElement("div");typing.className="typing";typing.textContent="S3 Agent is identifying the intent and preparing a response…";messages.append(typing);messages.scrollTop=messages.scrollHeight;
  try{
    const r=await fetch("/api/ask",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({question:q})});
    const d=await r.json(); if(!r.ok) throw new Error(d.error||"Request failed");
    typing.remove(); addMessage("assistant",d.answer,d);
  }catch(e){typing.remove();addMessage("assistant",e.message,{mode:"error"});}
  finally{$("send").disabled=false;$ ("send").textContent="Send ↗";$ ("question").focus();}
}
$("send").addEventListener("click",()=>ask($("question").value));
$("question").addEventListener("keydown",e=>{if(e.key==="Enter"&&!e.shiftKey){e.preventDefault();ask($("question").value);}});
document.querySelectorAll("[data-q]").forEach(b=>b.addEventListener("click",()=>ask(b.dataset.q)));
document.querySelectorAll(".nav").forEach(b=>b.addEventListener("click",()=>{
  const evalMode=b.dataset.view==="evaluation";
  $("chatView").classList.toggle("hidden",evalMode);$("evaluationView").classList.toggle("hidden",!evalMode);
  $("headerTitle").textContent=evalMode?"NLP evaluation":"Ask a question";
  document.querySelectorAll(".nav").forEach(x=>x.classList.toggle("active",x===b));
}));
$("loadMetrics").addEventListener("click",async()=>{
  $("metrics").textContent="Calculating…";
  try{const r=await fetch("/api/metrics");const d=await r.json();$("metrics").textContent=JSON.stringify(d,null,2);}
  catch(e){$("metrics").textContent=e.message;}
});
checkHealth();
