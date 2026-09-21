const $ = (q) => document.querySelector(q);
const siteLabels = {linkedin:'LinkedIn',indeed:'Indeed',zip_recruiter:'ZipRecruiter',glassdoor:'Glassdoor',google:'Google',bayt:'Bayt',naukri:'Naukri',bdjobs:'BDJobs'};
let allJobs = [], taskId = null, pollTimer = null;

function toast(message) { const el=$('#toast'); el.textContent=message; el.classList.remove('hidden'); setTimeout(()=>el.classList.add('hidden'),3200); }
function escapeHtml(value='') { return String(value ?? '').replace(/[&<>'"]/g, c=>({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}[c])); }
function renderMarkdown(value='') {
  const inline=text=>escapeHtml(text).replace(/\*\*(.+?)\*\*/g,'<strong>$1</strong>');
  const output=[]; let inList=false;
  for(const raw of String(value||'').split(/\r?\n/)) {
    const line=raw.trim();
    if(!line) { if(inList){output.push('</ul>');inList=false;} continue; }
    if(line.startsWith('## ')) { if(inList){output.push('</ul>');inList=false;} output.push(`<h3>${inline(line.slice(3))}</h3>`); continue; }
    if(line.startsWith('### ')) { if(inList){output.push('</ul>');inList=false;} output.push(`<h4>${inline(line.slice(4))}</h4>`); continue; }
    if(line.startsWith('- ')) { if(!inList){output.push('<ul>');inList=true;} output.push(`<li>${inline(line.slice(2))}</li>`); continue; }
    if(inList){output.push('</ul>');inList=false;} output.push(`<p>${inline(line)}</p>`);
  }
  if(inList) output.push('</ul>');
  return output.join('');
}
function checked(id) { return $(`#${id}`).checked; }

async function init() {
  try {
    const meta = await fetch('/api/meta').then(r=>r.json());
    $('#sites').innerHTML = meta.sites.map((s,i)=>`<label class="site-chip"><input type="checkbox" value="${s}" ${['indeed','linkedin'].includes(s)?'checked':''}><span>${siteLabels[s]||s}</span></label>`).join('');
    $('#country_indeed').innerHTML = meta.countries.map(c=>`<option value="${c}" ${c==='singapore'?'selected':''}>${c.replace(/\b\w/g,x=>x.toUpperCase())}</option>`).join('');
    $('#rag-source').innerHTML += meta.sites.map(s=>`<option value="${s}">${siteLabels[s]||s}</option>`).join('');
    await loadKnowledge();
  } catch { toast('无法连接本地服务'); }
}

function payload() {
  return {
    sites:[...document.querySelectorAll('#sites input:checked')].map(x=>x.value),
    search_term:$('#search_term').value, location:$('#location').value,
    results_wanted:Number($('#results_wanted').value), country_indeed:$('#country_indeed').value,
    distance:Number($('#distance').value), job_type:$('#job_type').value,
    hours_old:$('#hours_old').value?Number($('#hours_old').value):null, offset:Number($('#offset').value),
    description_format:$('#description_format').value, google_search_term:$('#google_search_term').value,
    proxies:$('#proxies').value.split(/\n/).map(x=>x.trim()).filter(Boolean), ca_cert:$('#ca_cert').value,
    linkedin_company_ids:$('#linkedin_company_ids').value.split(',').map(x=>x.trim()).filter(Boolean).map(Number).filter(Number.isInteger),
    user_agent:$('#user_agent').value, is_remote:checked('is_remote'), easy_apply:checked('easy_apply'),
    linkedin_fetch_description:checked('linkedin_fetch_description'), enforce_annual_salary:checked('enforce_annual_salary'), verbose:1
  };
}

$('#search-form').addEventListener('submit', async e => {
  e.preventDefault(); clearInterval(pollTimer); allJobs=[]; renderJobs([]);
  const button=$('.primary'); button.disabled=true; button.querySelector('span').textContent='正在创建…';
  try {
    const response=await fetch('/api/search',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload())});
    const body=await response.json();
    if(!response.ok) throw new Error(typeof body.detail==='string'?body.detail:'请检查搜索参数');
    taskId=body.id; $('#status').classList.remove('hidden'); $('#results-section').classList.add('hidden'); updateStatus(body);
    pollTimer=setInterval(poll,900); await poll();
  } catch(err) { toast(err.message); button.disabled=false; button.querySelector('span').textContent='开始搜索'; }
});

async function poll() {
  try {
    const task=await fetch(`/api/search/${taskId}?include_results=true`).then(r=>r.json()); updateStatus(task);
    if(['completed','partial','failed'].includes(task.status)) {
      clearInterval(pollTimer); allJobs=task.results||[]; renderSummary(allJobs); renderJobs(allJobs);
      $('#results-section').classList.remove('hidden');
      $('#csv').href=`/api/search/${taskId}/export/csv`; $('#xlsx').href=`/api/search/${taskId}/export/xlsx`;
      const button=$('.primary'); button.disabled=false; button.querySelector('span').textContent='再次搜索';
      await loadKnowledge();
    }
  } catch { clearInterval(pollTimer); toast('读取搜索进度失败'); }
}

function updateStatus(task) {
  const n=task.completed_sites?.length||0, total=task.total_sites||1, pct=Math.round(n/total*100);
  const labels={queued:'等待执行',running:'正在并行抓取',completed:'搜索完成',partial:'部分网站完成',failed:'搜索失败'};
  const errors=Object.entries(task.errors||{}).map(([s,e])=>`${siteLabels[s]||s}：${e}`).join('<br>');
  const kb=task.ingestion?` · 知识库新增 ${task.ingestion.inserted}，更新 ${task.ingestion.updated}`:'';
  $('#status').innerHTML=`<div class="status-row"><strong>${labels[task.status]||task.status}</strong><span>${n} / ${total} 个网站 · 已找到 ${task.result_count||0} 条${kb}</span></div><div class="progress"><i style="width:${pct}%"></i></div>${errors?`<p class="errors">${escapeHtml(errors).replace(/&lt;br&gt;/g,'<br>')}</p>`:''}`;
}

function salary(j) {
  if(j.min_amount==null && j.max_amount==null) return '—';
  const f=n=>n==null?'?':Number(n).toLocaleString(); return `${j.currency||''} ${f(j.min_amount)}–${f(j.max_amount)}${j.interval?` / ${j.interval}`:''}`;
}
function renderSummary(jobs) {
  const counts={}; jobs.forEach(j=>counts[j.site]=(counts[j.site]||0)+1);
  $('#site-summary').innerHTML=Object.entries(counts).map(([s,n])=>`<span>${siteLabels[s]||s} · ${n}</span>`).join('');
}
function renderJobs(jobs) {
  $('#result-count').textContent=jobs.length; $('#empty').classList.toggle('hidden',jobs.length>0);
  $('#jobs').innerHTML=jobs.map((j,i)=>`<tr><td><div class="job-title">${escapeHtml(j.title)}</div><small>${j.is_remote?'● 远程':''}</small></td><td class="company">${escapeHtml(j.company||'—')}</td><td>${escapeHtml(j.location||'—')}</td><td><span class="source">${escapeHtml(siteLabels[j.site]||j.site)}</span></td><td>${escapeHtml(salary(j))}</td><td>${escapeHtml(j.date_posted||'—')}</td><td><button class="view" data-index="${allJobs.indexOf(j)}">详情 →</button></td></tr>`).join('');
}
$('#filter').addEventListener('input', e=>{ const q=e.target.value.toLowerCase(); renderJobs(allJobs.filter(j=>[j.title,j.company,j.location,j.site,j.skills].some(v=>String(v||'').toLowerCase().includes(q)))); });
$('#jobs').addEventListener('click',e=>{ const btn=e.target.closest('.view'); if(btn) showDetail(allJobs[Number(btn.dataset.index)]); });
function showDetail(j) {
  const fields=[['来源',siteLabels[j.site]||j.site],['职位类型',j.job_type],['薪资',salary(j)],['远程',j.is_remote?'是':'否'],['级别',j.job_level],['技能',j.skills]].filter(x=>x[1]);
  $('#detail-content').innerHTML=`<div class="detail-body"><p class="eyebrow">${escapeHtml(siteLabels[j.site]||j.site)}</p><h2>${escapeHtml(j.title)}</h2><p class="detail-meta">${escapeHtml(j.company||'未知公司')} · ${escapeHtml(j.location||'地点未知')} · ${escapeHtml(j.date_posted||'日期未知')}</p><div class="detail-grid">${fields.map(x=>`<div><b>${escapeHtml(x[0])}</b><br>${escapeHtml(x[1])}</div>`).join('')}</div><div class="description">${escapeHtml(j.description||'暂无职位描述')}</div>${j.job_url?`<a class="apply" href="${escapeHtml(j.job_url_direct||j.job_url)}" target="_blank" rel="noopener">查看原始职位 ↗</a>`:''}</div>`;
  $('#detail').showModal();
}
$('.close').addEventListener('click',()=>$('#detail').close()); $('#detail').addEventListener('click',e=>{if(e.target===$('#detail'))$('#detail').close()});

async function switchView(viewId) {
  document.querySelectorAll('.nav-tab').forEach(item=>item.classList.toggle('active',item.dataset.view===viewId));
  document.querySelectorAll('.app-view').forEach(view=>view.classList.toggle('hidden',view.id!==viewId));
  if(viewId==='kb-view') await loadKnowledge();
  window.scrollTo({top:0,behavior:'smooth'});
}
document.querySelectorAll('.nav-tab').forEach(button=>button.addEventListener('click',()=>{ if(button.dataset.view) switchView(button.dataset.view); }));
$('#go-kb').addEventListener('click',()=>switchView('kb-view'));

async function responseJson(response) {
  const body=await response.json().catch(()=>({}));
  if(!response.ok) throw new Error(body.detail||`请求失败（${response.status}）`);
  return body;
}

async function loadKnowledge() {
  try {
    const [stats,index,jobs]=await Promise.all([
      fetch('/api/kb/stats').then(responseJson),
      fetch('/api/kb/index').then(responseJson),
      fetch('/api/kb/jobs?limit=9').then(responseJson)
    ]);
    $('#kb-stats').innerHTML=[['有效岗位',stats.active_jobs],['已失效',stats.expired_jobs],['可检索岗位',stats.searchable_jobs],['有完整描述',stats.jobs_with_description]].map(([label,value])=>`<div class="stat-card"><strong>${value||0}</strong><span>${label}</span></div>`).join('');
    $('#coverage-status').innerHTML=`<i style="width:${stats.description_coverage_percent||0}%"></i><span>岗位描述覆盖率 ${stats.description_coverage_percent||0}% · ${stats.jobs_with_description||0} / ${stats.total_jobs||0}</span>`;
    $('#index-status').textContent=`${index.indexed_chunks} / ${index.total_chunks} 个知识片段已生成向量 · ${index.pending_chunks} 个待处理 · ${index.model}`;
    $('#knowledge-jobs').innerHTML=jobs.items.length?jobs.items.map(job=>`<article class="knowledge-card"><strong>${escapeHtml(job.title)}</strong><p>${escapeHtml(job.company||'未知公司')} · ${escapeHtml(job.location||'地点未知')} · ${escapeHtml(siteLabels[job.source]||job.source)}</p>${job.job_url?`<a href="${escapeHtml(job.job_url_direct||job.job_url)}" target="_blank" rel="noopener">查看岗位 ↗</a>`:''}</article>`).join(''):'<div class="empty">知识库还没有岗位，请先完成一次搜索。</div>';
  } catch(err) { toast(err.message); }
}

$('#index-button').addEventListener('click',async()=>{
  const button=$('#index-button'); button.disabled=true; button.textContent='本地模型处理中…';
  try {
    const result=await fetch('/api/kb/index',{method:'POST'}).then(responseJson);
    toast(`已生成 ${result.chunks_indexed} 个向量`); await loadKnowledge();
  } catch(err) { toast(err.message); }
  finally { button.disabled=false; button.textContent='生成本地向量'; }
});

$('#backfill-button').addEventListener('click',async()=>{
  const button=$('#backfill-button'); button.disabled=true; button.textContent='正在补全 LinkedIn 描述…';
  try {
    const result=await fetch('/api/kb/backfill/linkedin?limit=25',{method:'POST'}).then(responseJson);
    toast(`补全 ${result.descriptions_added} 个岗位，失败 ${result.failed} 个`); await loadKnowledge();
  } catch(err) { toast(err.message); }
  finally { button.disabled=false; button.textContent='补全缺失描述'; }
});

$('#ask-form').addEventListener('submit',async event=>{
  event.preventDefault(); const button=$('#ask-button'); button.disabled=true; button.querySelector('span').textContent='检索与分析中…';
  $('#rag-answer').classList.add('hidden');
  try {
    const request={query:$('#kb-question').value.trim(),source:$('#rag-source').value,location:$('#rag-location').value,title:$('#rag-title').value,job_type:$('#rag-job-type').value,top_k:Number($('#rag-top-k').value)};
    const result=await fetch('/api/kb/ask',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(request)}).then(responseJson);
    const sources=(result.sources||[]).map(source=>`<a href="${escapeHtml(source.url||'#')}" target="_blank" rel="noopener">[${source.number}] ${escapeHtml(source.title)} · ${escapeHtml(source.company||'未知公司')} · ${escapeHtml(source.location||'地点未知')} · ${escapeHtml(source.date_posted||'日期未知')} ↗</a>`).join('');
    const scope=result.analysis_scope?`<div class="analysis-scope"><span>检索范围：${escapeHtml(result.analysis_scope.scope_query||'—')}</span><span>内部候选 ${result.analysis_scope.candidate_job_count||0} 个</span><span>分析最相关岗位 ${result.analysis_scope.analyzed_job_count||0} 个</span></div>`:'';
    const validation=result.answer_validation?`<span class="validation-chip ${result.answer_validation.status==='passed'?'passed':'fallback'}">${result.answer_validation.status==='passed'?'引用校验通过':'已回退为可核查证据'}</span>`:'';
    $('#rag-answer').innerHTML=`<p class="eyebrow">EVIDENCE-BASED ANSWER</p>${validation}${scope}<div class="answer-text">${renderMarkdown(result.answer)}</div>${sources?`<div class="evidence-list">${sources}</div>`:''}`;
    $('#rag-answer').classList.remove('hidden');
  } catch(err) { toast(err.message); }
  finally { button.disabled=false; button.querySelector('span').textContent='分析岗位要求'; }
});
init();
