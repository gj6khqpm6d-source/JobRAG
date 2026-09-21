const $ = (q) => document.querySelector(q);
const categoryLabels = {skills:'技能', experience:'经验与职责', comparison:'岗位比较', filters:'过滤条件', insufficient_evidence:'证据不足'};
const modeLabels = {hybrid:'Hybrid', vector:'Vector', keyword:'Keyword'};
const state = {questions:[], selected:null, candidates:[], annotation:null};

function escapeHtml(value='') { return String(value ?? '').replace(/[&<>'"]/g, c=>({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}[c])); }
function toast(message) { const el=$('#toast'); el.textContent=message; el.classList.remove('hidden'); setTimeout(()=>el.classList.add('hidden'),3200); }
function responseJson(response) { return response.json().then(body=>{ if(!response.ok) throw new Error(body.detail||`请求失败（${response.status}）`); return body; }); }
function percent(value) { return `${Math.round(Number(value||0)*100)}%`; }
function milliseconds(value) { return `${Math.round(Number(value||0))} ms`; }
function formatDate(value) { return value ? new Date(value).toLocaleString('zh-CN',{hour12:false}) : '暂无'; }
function renderOps(report) {
  const requests=report.requests||{}, latency=report.latency_ms||{}, llm=report.llm||{}, kb=report.knowledge_base||{}, index=report.index||{}, versions=report.versions||{}, privacy=report.privacy||{};
  $('#ops-status').innerHTML=`<strong class="${requests.errors?'quality-fail':'quality-pass'}">${requests.errors?`${requests.errors} 个错误`:'运行正常'}</strong><span>近 ${escapeHtml(report.window_days||7)} 天 · ${escapeHtml(requests.total||0)} 次问答 · 更新于 ${escapeHtml(formatDate(report.generated_at))}</span>`;
  const cost=llm.cost_configured ? `$${Number(llm.estimated_cost_usd||0).toFixed(4)}` : '未配置单价';
  $('#ops-metrics').innerHTML=[['P50 延迟',milliseconds(latency.p50)],['P95 延迟',milliseconds(latency.p95)],['缓存命中',percent(requests.cache_hit_rate)],['失败 / 无结果',`${requests.errors||0} / ${requests.no_results||0}`],['Token / 成本',`${llm.total_tokens||0} · ${cost}`]].map(([label,value])=>`<div><strong>${escapeHtml(value)}</strong><span>${escapeHtml(label)}</span></div>`).join('');
  $('#ops-kb').textContent=`${kb.active_jobs||0} 个有效岗位`;
  $('#ops-freshness').textContent=`最新岗位 ${kb.latest_job_posted||'未知'} · 保留 ${kb.retention_days||60} 天`;
  const coverage=index.total_chunks ? Math.round((index.indexed_chunks||0)/index.total_chunks*100) : 0;
  $('#ops-index').textContent=`${coverage}% 已索引`;
  $('#ops-model').textContent=`${index.indexed_chunks||0}/${index.total_chunks||0} chunks · ${index.model||'未配置'}`;
  $('#ops-version').textContent=(versions.pipelines||[]).join(', ')||'等待首个请求';
  $('#ops-privacy').textContent=`不存原始问题/Prompt · 日志保留 ${privacy.log_retention_days||30} 天`;
  const recent=report.recent_requests||[];
  $('#ops-recent').innerHTML=recent.length?`<div class="ops-row ops-row-head"><span>时间 / 追踪号</span><span>状态</span><span>耗时</span><span>缓存</span><span>证据</span><span>Token</span></div>${recent.map(row=>`<div class="ops-row"><span>${escapeHtml(formatDate(row.created_at))}<small>${escapeHtml((row.request_id||'').slice(0,10))}</small></span><span class="${row.status==='success'?'quality-pass':'quality-fail'}">${row.status==='success'?'成功':escapeHtml(row.error_type||'失败')}</span><span>${escapeHtml(milliseconds(row.total_ms))}</span><span>${row.cache_hit?'命中':'未命中'}</span><span>${escapeHtml(row.source_count||0)}</span><span>${escapeHtml(row.total_tokens||0)}</span></div>`).join('')}</div>`:'<div class="empty">暂无问答请求记录。请在知识库页面完成一次提问后刷新。</div>';
}
async function loadOps() {
  try { renderOps(await fetch('/api/ops/summary?days=7').then(responseJson)); }
  catch(error) { $('#ops-status').textContent=error.message; }
}
function renderQuality(report) {
  if(!report) { $('#quality-status').textContent='尚未运行质量评估。'; $('#quality-metrics').innerHTML=''; $('#quality-gates').innerHTML=''; return; }
  const layers=report.layers||{}, retrieval=layers.retrieval||{}, answer=layers.answer||{}, operations=layers.operations||{}, dataset=layers.dataset||{};
  $('#quality-status').innerHTML=`<strong class="${report.passed?'quality-pass':'quality-fail'}">${report.passed?'质量门禁通过':'质量门禁未通过'}</strong><span>${escapeHtml(report.generated_at||'')} · ${escapeHtml(report.duration_seconds||0)} 秒 · ${dataset.annotations||0}/${dataset.questions||0} 题含人工校准</span>`;
  $('#quality-metrics').innerHTML=[['Recall@8',percent(retrieval.recall_at_8)],['NDCG@8',percent(retrieval.graded_ndcg_at_8)],['引用有效率',percent(answer.citation_validity_rate)],['拒答准确率',percent(answer.refusal_accuracy)],['索引覆盖率',percent(operations.index_coverage)]].map(([label,value])=>`<div><strong>${escapeHtml(value)}</strong><span>${escapeHtml(label)}</span></div>`).join('');
  $('#quality-gates').innerHTML=(report.gates||[]).map(gate=>`<div class="quality-gate ${gate.passed?'passed':'failed'}"><i></i><span>${escapeHtml(gate.name)}</span><b>${escapeHtml(gate.actual)} / ${escapeHtml(gate.expected)}</b></div>`).join('');
}
async function loadQuality() {
  try { const body=await fetch('/api/eval/quality').then(responseJson); renderQuality(body.report); }
  catch(error) { $('#quality-status').textContent=error.message; }
}
function currentJudgments() {
  const output = {};
  document.querySelectorAll('.judgment-select').forEach(select=>{ if(select.value!=='') output[select.dataset.jobId] = Number(select.value); });
  return output;
}
function currentEvidence() { return [...document.querySelectorAll('.evidence-check:checked')].map(input=>Number(input.value)); }
function annotationMaps(annotation) {
  return {
    judgments: Object.fromEntries((annotation?.job_judgments||[]).map(item=>[item.job_id, item.relevance])),
    evidence: new Set(annotation?.evidence_chunk_ids||[])
  };
}
function renderQuestions() {
  const done = state.questions.filter(q=>q.annotated).length;
  $('#progress-count').textContent = `${done} / ${state.questions.length}`;
  const groups = {};
  state.questions.forEach(question=>(groups[question.category] ||= []).push(question));
  $('#question-list').innerHTML = Object.entries(groups).map(([category, questions])=>`<div class="question-group"><p class="eyebrow">${escapeHtml(categoryLabels[category]||category)}</p>${questions.map(question=>`<button class="question-item ${state.selected===question.id?'active':''}" data-question-id="${escapeHtml(question.id)}"><div class="question-item-top"><span>${escapeHtml(question.id)}</span><i class="question-item-status ${question.annotated?'done':''}"></i></div><div class="question-item-query">${escapeHtml(question.query)}</div></button>`).join('')}</div>`).join('');
}
function renderQuestionHeader(question) {
  $('#question-category').textContent = `${categoryLabels[question.category]||question.category} · ${question.id}`;
  $('#question-query').textContent = question.query;
  const filters = Object.entries(question.filters||{}).map(([key,value])=>`<span class="filter-chip">${escapeHtml(key)}: ${escapeHtml(value)}</span>`);
  if(question.should_refuse) filters.push('<span class="filter-chip">预期：检查是否拒答</span>');
  $('#question-filters').innerHTML = filters.join('') || '<span class="filter-chip">无额外过滤</span>';
}
function renderCandidates() {
  const maps = annotationMaps(state.annotation);
  $('#candidate-list').innerHTML = state.candidates.length ? state.candidates.map(candidate=>`<article class="candidate-card"><div class="candidate-head"><div><h3 class="candidate-title">${escapeHtml(candidate.title)}</h3><div class="candidate-meta">${escapeHtml(candidate.company||'未知公司')} · ${escapeHtml(candidate.location||'地点未知')} · ${escapeHtml(candidate.job_type||'类型未知')} · ${escapeHtml(candidate.source)}</div></div><div class="candidate-badges">${candidate.retrieved_by.map(mode=>`<span class="mode-chip">${modeLabels[mode]||mode}</span>`).join('')}</div></div><div class="judgment"><label for="judgment-${escapeHtml(candidate.job_id)}">岗位相关性</label><select id="judgment-${escapeHtml(candidate.job_id)}" class="judgment-select" data-job-id="${escapeHtml(candidate.job_id)}"><option value="" ${maps.judgments[candidate.job_id]===undefined?'selected':''}>未判断</option><option value="0" ${maps.judgments[candidate.job_id]===0?'selected':''}>0 · 无关</option><option value="1" ${maps.judgments[candidate.job_id]===1?'selected':''}>1 · 部分相关</option><option value="2" ${maps.judgments[candidate.job_id]===2?'selected':''}>2 · 高度相关</option></select></div><div class="candidate-chunks">${candidate.chunks.map(chunk=>`<div class="evidence-chunk"><div class="chunk-top"><label><input class="evidence-check" type="checkbox" value="${chunk.chunk_id}" ${maps.evidence.has(chunk.chunk_id)?'checked':''}>作为回答证据</label><span class="chunk-type">${escapeHtml(chunk.section_type)}</span><span class="chunk-type">chunk ${chunk.chunk_id}</span></div><div class="chunk-content">${escapeHtml(chunk.content)}</div></div>`).join('')}</div></article>`).join('') : '<div class="empty">没有找到可标注的候选证据，请检查索引状态。</div>';
}
function renderAnnotationForm(question) {
  const annotation = state.annotation || {};
  $('#should-refuse').checked = annotation.should_refuse ?? question.should_refuse;
  $('#answer-key-points').value = (annotation.answer_key_points||[]).join('\n');
  $('#annotation-notes').value = annotation.notes||'';
  $('#saved-at').textContent = annotation.updated_at ? `上次保存：${annotation.updated_at}` : '';
}
async function loadCandidates(questionId) {
  state.selected = questionId; renderQuestions();
  $('#eval-empty').classList.add('hidden'); $('#eval-content').classList.remove('hidden');
  $('#candidate-list').innerHTML = '<div class="empty">正在生成查询向量并加载三种检索候选…</div>';
  $('#candidate-status').textContent = '本地模型处理中，首次加载可能需要一些时间。';
  try {
    const body = await fetch(`/api/eval/questions/${encodeURIComponent(questionId)}/candidates?top_k=20`).then(responseJson);
    state.candidates = body.candidates||[]; state.annotation = body.question.annotation;
    renderQuestionHeader(body.question); renderCandidates(); renderAnnotationForm(body.question);
    $('#candidate-status').textContent = `共 ${state.candidates.length} 个岗位候选，来自 Hybrid、Vector、Keyword 三种检索结果的合并。`;
  } catch(error) { $('#candidate-list').innerHTML = `<div class="empty">${escapeHtml(error.message)}</div>`; $('#candidate-status').textContent=''; toast(error.message); }
}
async function loadQuestions() {
  try { const body=await fetch('/api/eval/questions').then(responseJson); state.questions=body.items||[]; renderQuestions(); if(state.questions.length) await loadCandidates(state.questions[0].id); }
  catch(error) { $('#question-list').innerHTML=`<div class="empty">${escapeHtml(error.message)}</div>`; toast(error.message); }
}
$('#question-list').addEventListener('click', event=>{ const button=event.target.closest('.question-item'); if(button) loadCandidates(button.dataset.questionId); });
$('#reload-candidates').addEventListener('click',()=>{ if(state.selected) loadCandidates(state.selected); });
$('#save-annotation').addEventListener('click', async()=>{
  if(!state.selected) return;
  const button=$('#save-annotation'); button.disabled=true; button.querySelector('span').textContent='保存中…'; $('#save-status').textContent='';
  const payload={job_judgments:Object.entries(currentJudgments()).map(([job_id,relevance])=>({job_id,relevance})),evidence_chunk_ids:currentEvidence(),answer_key_points:$('#answer-key-points').value.split(/\r?\n/).map(item=>item.trim()).filter(Boolean),should_refuse:$('#should-refuse').checked,notes:$('#annotation-notes').value.trim()};
  try { const body=await fetch(`/api/eval/questions/${encodeURIComponent(state.selected)}/annotation`,{method:'PUT',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)}).then(responseJson); state.annotation=body.annotation; const question=state.questions.find(item=>item.id===state.selected); if(question) {question.annotated=true; question.annotation=body.annotation;} renderQuestions(); renderAnnotationForm(question||{}); $('#save-status').textContent='已保存'; toast('本题标注已保存'); }
  catch(error) { $('#save-status').textContent=error.message; toast(error.message); }
  finally { button.disabled=false; button.querySelector('span').textContent='保存标注'; }
});
$('#run-quality').addEventListener('click',async()=>{
  const button=$('#run-quality'); button.disabled=true; button.textContent='正在评估…'; $('#quality-status').textContent='正在运行检索、回答和索引检查，首次运行可能需要一些时间。';
  try { const report=await fetch('/api/eval/quality/run',{method:'POST'}).then(responseJson); renderQuality(report); toast(report.passed?'质量门禁通过':'评估完成，存在未通过项目'); }
  catch(error) { $('#quality-status').textContent=error.message; toast(error.message); }
  finally { button.disabled=false; button.textContent='运行完整评估'; }
});
$('#refresh-ops').addEventListener('click',async()=>{ const button=$('#refresh-ops'); button.disabled=true; await loadOps(); button.disabled=false; });
loadOps(); loadQuality(); loadQuestions();
