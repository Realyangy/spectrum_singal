'use strict';
const $ = id => document.getElementById(id);
const kinds = {sep: 'signal-separation', spec: 'spectrum-analysis'};
let libraries = {}, tab = 'sep', state = {}, shown = '', dismissedTask = '', dismissedFailure = '', imageRevision = 0, clearingModels = false;

async function request(url, options = {}) {
  const response = await fetch(url, options);
  const data = await response.json();
  if (!response.ok) throw Error(data.error || '请求失败');
  return data;
}
function rows() {
  const matches = (libraries[tab]?.samples || []).filter(x => tab === 'spec' || (x.soi_type === $('soi').value && x.interference === $('interference').value));
  if (tab === 'spec') return matches;
  return matches.slice().sort((a, b) => a.nominal_sinr_db - b.nominal_sinr_db || a.sample_number - b.sample_number)
    .map((sample, index) => ({...sample, display_number: index + 1}));
}
function selected() {
  return rows().find(x => x.id === $(tab === 'sep' ? 'sep-sample' : 'spec-sample').value);
}
function clearResult() {
  imageRevision += 1;
  $('result').hidden = true;
  $('result-empty').hidden = false;
  shown = '';
}
function updateSample() {
  const sample = selected();
  $('sample').textContent = sample ? `样本 ${tab === 'sep' ? sample.display_number : String(sample.sample_number).padStart(3, '0')}` : '—';
  if (tab === 'sep') $('sep-meta').textContent = sample ? `标称 SINR ${sample.nominal_sinr_db} dB · 样本 ${sample.display_number}` : '暂无样本';
  clearResult();
  render(state);
}
function fill() {
  const select = $(tab === 'sep' ? 'sep-sample' : 'spec-sample');
  const previous = select.value;
  select.replaceChildren(...rows().map(x => new Option(tab === 'sep' ? `样本 ${x.display_number} · ${x.nominal_sinr_db} dB` : `样本 ${String(x.sample_number).padStart(3, '0')}`, x.id)));
  if ([...select.options].some(x => x.value === previous)) select.value = previous;
  updateSample();
}
function switchTab(next) {
  tab = next;
  $('sep').hidden = next !== 'sep'; $('spec').hidden = next !== 'spec';
  for (const [id, active] of [['tab-sep', next === 'sep'], ['tab-spec', next === 'spec']]) {
    $(id).classList.toggle('active', active);
    $(id).setAttribute('aria-selected', String(active));
  }
  fill();
}
for (const id of ['soi', 'interference']) $(id).addEventListener('change', fill);
for (const id of ['sep-sample', 'spec-sample']) $(id).addEventListener('change', updateSample);
$('tab-sep').onclick = () => switchTab('sep'); $('tab-spec').onclick = () => switchTab('spec');
$('reset').onclick = () => {
  if (state.active) return;
  dismissedTask = state.task_id || '';
  dismissedFailure = !state.task_id && state.phase === 'error' ? JSON.stringify(state) : '';
  state = {};
  $('soi').selectedIndex = 0; $('interference').selectedIndex = 0;
  $('sep-sample').replaceChildren(); $('spec-sample').replaceChildren();
  $('summary').replaceChildren(); $('targets').replaceChildren();
  $('target-detail').replaceChildren(); $('result-modes').replaceChildren(); $('image').removeAttribute('src');
  switchTab('sep');
};
$('send').onclick = async () => {
  try {
    const sample = selected();
    if (!sample) throw Error('没有可用样本');
    dismissedFailure = '';
    $('send').disabled = true; $('reset').disabled = true;
    state = await request('/api/rf/start', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({task:kinds[tab], sample_id:sample.id})});
    clearResult(); render(state);
  } catch (error) { render(state); $('status').textContent = error.message; }
};
$('clear-models').onclick = async () => {
  if (state.active || clearingModels) return;
  const task = kinds[tab], name = tab === 'sep' ? '信号分离' : '频谱检测与识别';
  clearingModels = true; render(state);
  try {
    const info = await request('/api/rf/models/' + task, {cache:'no-store'});
    if (info.active) throw Error('任务运行中，暂不能删除模型');
    if (!info.files) { alert('没有找到可清理的' + name + '下发模型。'); return; }
    if (!confirm(`删除${name}任务已接收的 ${info.files} 个模型文件（约 ${(info.bytes / 1024 / 1024).toFixed(1)} MiB）？\n包含 Agent 接收文件和项目内副本，不影响样本和结果。`)) return;
    const result = await request('/api/rf/models/delete', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({task})});
    alert(`已删除 ${result.removed} 个文件，释放 ${(result.bytes / 1024 / 1024).toFixed(1)} MiB。${result.skipped ? `\n${result.skipped} 个文件因状态变化或读取失败而保留。` : ''}`);
  } catch (error) { alert(error.message); }
  finally { clearingModels = false; render(state); }
};
$('cancel').onclick = async () => {
  try { state = await request('/api/rf/cancel', {method:'POST'}); render(state); }
  catch (error) { $('status').textContent = error.message; }
};
function metric(label, value) {
  const item = document.createElement('div'); item.className = 'metric';
  const title = document.createElement('span'); title.textContent = label;
  const content = document.createElement('strong'); content.textContent = value;
  item.append(title, content); return item;
}
function detail(label, value) {
  const row = document.createElement('div'); row.className = 'detail-row';
  const title = document.createElement('span'); title.textContent = label;
  const content = document.createElement('strong'); content.textContent = value;
  row.append(title, content); return row;
}
function signalFamily(name) {
  if (name.startsWith('ofdm-')) return 'OFDM';
  if (name.includes('qam')) return 'QAM';
  if (name.includes('psk')) return 'PSK';
  if (name.endsWith('msk')) return 'MSK/GMSK';
  if (name.endsWith('fsk')) return 'FSK/GFSK';
  if (name.includes('ask') || name === 'ook') return 'ASK/OOK';
  if (name.startsWith('am-')) return 'AM';
  if (name === 'fm') return 'FM';
  return 'Other';
}
function matchSignals(detections, truth) {
  const boxes = truth?.boxes || [], names = truth?.class_names || [], used = new Set();
  return detections.map(detection => {
    const a = detection.box_xyxy_normalized;
    if (!Array.isArray(a) || a.length !== 4 || !a.every(Number.isFinite)) return null;
    let best = null, bestIou = 0;
    boxes.forEach((raw, index) => {
      if (used.has(index) || !names[index] || !Array.isArray(raw) || raw.length !== 5 || !raw.every(Number.isFinite)) return;
      const [, x, y, w, h] = raw, b = [x-w/2, y-h/2, x+w/2, y+h/2];
      const intersection = Math.max(0, Math.min(a[2], b[2])-Math.max(a[0], b[0])) * Math.max(0, Math.min(a[3], b[3])-Math.max(a[1], b[1]));
      const union = Math.max(0,a[2]-a[0])*Math.max(0,a[3]-a[1]) + Math.max(0,w)*Math.max(0,h) - intersection;
      const iou = union > 0 ? intersection / union : 0;
      if (iou >= bestIou) { best = index; bestIou = iou; }
    });
    if (best === null || bestIou < .5) return null;
    used.add(best);
    return {name:names[best], family:signalFamily(names[best]), iou:bestIou};
  });
}
function showSignalImage(taskId, detection, overviewUrl) {
  const revision = ++imageRevision;
  const display = $('image');
  const box = detection.box_xyxy_normalized;
  const fallback = () => {
    if (revision !== imageRevision) return;
    display.src = overviewUrl; display.alt = '整体时频图及检测结果';
    $('target-detail').append(detail('局部图像', '暂不可用，显示整体时频图'));
  };
  if (!Array.isArray(box) || box.length !== 4 || !box.every(Number.isFinite)) return fallback();
  const [x0, y0, x1, y1] = box.map(value => Math.max(0, Math.min(1, value)));
  if (x1 <= x0 || y1 <= y0) return fallback();
  const source = new Image();
  source.onload = () => {
    if (revision !== imageRevision) return;
    const size = Math.min(1, Math.max(0.12, (x1 - x0) * 1.4, (y1 - y0) * 1.4));
    const left = Math.max(0, Math.min(1 - size, (x0 + x1 - size) / 2));
    const top = Math.max(0, Math.min(1 - size, (y0 + y1 - size) / 2));
    const canvas = document.createElement('canvas');
    canvas.width = canvas.height = 512;
    const context = canvas.getContext('2d');
    if (!context) return fallback();
    context.drawImage(source, left * source.width, top * source.height, size * source.width, size * source.height, 0, 0, 512, 512);
    context.strokeStyle = '#ff8a00'; context.lineWidth = 3;
    context.strokeRect((x0 - left) / size * 512, (y0 - top) / size * 512, (x1 - x0) / size * 512, (y1 - y0) / size * 512);
    display.src = canvas.toDataURL('image/png');
  };
  source.onerror = fallback;
  source.src = '/runs/' + encodeURIComponent(taskId) + '/model-input/yolo.png';
}
function renderResult(s) {
  const result = s.result;
  if (!result || !s.task_id) return;
  $('result').hidden = false; $('result-empty').hidden = true;
  $('result').classList.toggle('separation-result', result.task === 'signal-separation');
  $('result').classList.remove('spectrum-overview', 'spectrum-signal');
  const path = name => '/runs/' + encodeURIComponent(s.task_id) + '/' + encodeURIComponent(name);
  const modes = $('result-modes'), targets = $('targets'), summary = $('summary');
  modes.replaceChildren(); targets.replaceChildren(); summary.replaceChildren(); $('target-detail').replaceChildren();
  if (result.task === 'signal-separation') {
    summary.append(metric('输入 BER', `${(100 * result.input_ber).toFixed(3)}%`), metric('恢复后 BER', `${(100 * result.output_ber).toFixed(3)}%`), metric('恢复 MSE', Number(result.mse).toPrecision(5)), metric('错误比特数', `${result.output_bit_errors} / ${result.bit_count}`));
    for (const [index, name] of (result.images || []).entries()) {
      const button = document.createElement('button'); button.type = 'button'; button.textContent = name === 'waveforms.png' ? 'I/Q 波形' : '时频图';
      button.onclick = () => { $('image').src = path(name); for (const item of modes.children) item.classList.remove('active'); button.classList.add('active'); };
      modes.append(button); if (!index) button.click();
    }
  } else {
    const detections = result.detections || [];
    const sample = selected();
    const truth = sample?.ground_truth;
    const labels = truth?.class_names || [];
    const matches = matchSignals(detections, truth);
    const overviewUrl = path((result.images || [])[0] || 'detections.png');
    const addMode = (label, action) => {
      const button = document.createElement('button'); button.type = 'button'; button.textContent = label;
      button.onclick = () => {
        for (const item of modes.children) { item.classList.remove('active'); item.setAttribute('aria-pressed', 'false'); }
        button.classList.add('active'); button.setAttribute('aria-pressed', 'true'); action();
      };
      button.setAttribute('aria-pressed', 'false'); modes.append(button); return button;
    };
    const overview = addMode('整体时频图', () => {
      imageRevision += 1;
      $('result').classList.remove('spectrum-signal'); $('result').classList.add('spectrum-overview');
      $('image').src = overviewUrl; $('image').alt = '整体时频图及检测结果';
      const matched = matches.filter(Boolean).length;
      summary.replaceChildren(metric('当前样本', sample ? `样本 ${String(sample.sample_number).padStart(3, '0')}` : '—'),
        metric('标注信号', String(labels.length)), metric('检测信号', String(detections.length)), metric('匹配信号', String(matched)));
      $('target-detail').replaceChildren();
      const heading = document.createElement('h4'); heading.textContent = '标准信号家族'; $('target-detail').append(heading);
      const families = new Map();
      labels.forEach(name => { const family = signalFamily(name); families.set(family, (families.get(family) || 0) + 1); });
      for (const [family, count] of families) $('target-detail').append(detail(family, `${count} 个信号`));
      if (!labels.length) $('target-detail').append(detail('标注信息', '暂无'));
      const note = document.createElement('p'); note.className = 'summary-note';
      note.textContent = '检测框与标注框按 IoU ≥ 0.5 一对一匹配；点击信号可查看对比。'; $('target-detail').append(note);
    });
    detections.forEach((detection, index) => {
      addMode(`信号 ${detection.detection_id}`, () => {
        $('result').classList.remove('spectrum-overview'); $('result').classList.add('spectrum-signal');
        summary.replaceChildren();
        const features = detection.features || {}, xcit = detection.xcit || {}, match = matches[index];
        $('image').alt = `信号 ${detection.detection_id} 的检测框局部时频图`;
        $('target-detail').replaceChildren(
          detail('标准调制家族', match ? match.family : '未匹配'),
          detail('预测调制家族', xcit.family_top1 || detection.xcit_error || '未识别'),
          detail('YOLO 置信度', Number.isFinite(detection.confidence) ? `${(detection.confidence * 100).toFixed(1)}%` : '—'),
          detail('中心频率偏移', Number.isFinite(features.center_frequency_mhz_offset) ? `${features.center_frequency_mhz_offset.toFixed(3)} MHz` : '—'),
          detail('信号带宽', Number.isFinite(features.bandwidth_mhz) ? `${features.bandwidth_mhz.toFixed(3)} MHz` : '—'),
          detail('信号时长', Number.isFinite(features.duration_ms) ? `${features.duration_ms.toFixed(3)} ms` : '—')
        );
        showSignalImage(s.task_id, detection, overviewUrl);
      });
    });
    overview.click();
  }
}
function render(next) {
  const dismissed = next && !next.active && ((next.task_id && next.task_id === dismissedTask) || (!next.task_id && dismissedFailure && JSON.stringify(next) === dismissedFailure));
  state = dismissed ? {} : (next || {});
  const active = !!state.active;
  $('send').disabled = active || clearingModels; $('reset').disabled = active || clearingModels; $('clear-models').disabled = active || clearingModels; $('cancel').hidden = !active;
  for (const id of ['soi', 'interference', 'sep-sample', 'spec-sample']) $(id).disabled = active;
  $('status').textContent = state.message || '等待发送';
  $('model').textContent = [state.model, state.quantization].filter(x => typeof x === 'string').join(' · ') || '等待选择';
  const percent = Math.round(100 * (state.receive_progress || 0));
  $('pct').textContent = percent + '%'; $('progress').value = percent;
  $('task-id').textContent = state.task_id ? '任务编号：' + state.task_id : '';
  if (state.result && state.task_type === kinds[tab] && selected()?.id === state.sample_id) {
    const signature = state.task_id + JSON.stringify(state.result);
    if (shown !== signature) { shown = signature; renderResult(state); }
  }
}
async function poll() {
  try { render(await request('/api/rf/status', {cache:'no-store'})); }
  catch (error) { $('status').textContent = '端侧服务连接失败：' + error.message; }
  finally { setTimeout(poll, 2000); }
}
Promise.all(Object.values(kinds).map(kind => request('/api/rf/samples/' + kind)))
  .then(([separation, spectrum]) => { libraries = {sep:separation, spec:spectrum}; fill(); poll(); })
  .catch(error => { $('status').textContent = '样本库读取失败：' + error.message; });

function percent(value) { return `${(Number(value) * 100).toFixed(2)}%`; }
function experimentMetric(label, value, note) {
  const box = document.createElement('div'); box.className = 'experiment-metric';
  const heading = document.createElement('span'); heading.textContent = label;
  const number = document.createElement('strong'); number.textContent = value;
  const description = document.createElement('small'); description.textContent = note;
  box.append(heading, number, description); return box;
}
function renderExperiments(data) {
  const separation = data.separation, spectrum = data.spectrum, overall = separation.overall;
  $('sep-experiment-metrics').replaceChildren(
    experimentMetric('样本总数', String(overall.samples), '6 组 · 每组 50 条'),
    experimentMetric('输入 BER', percent(overall.input_ber), '全部比特统计'),
    experimentMetric('恢复后 BER', percent(overall.output_ber), '全部比特统计'),
    experimentMetric('平均恢复 MSE', `${Number(overall.mean_mse_db).toFixed(2)} dB`, '线性平均后转换')
  );
  const groups = $('sep-experiment-groups'); groups.replaceChildren();
  const order = {QPSK: 0, OFDMQPSK: 1};
  [...separation.groups].sort((a, b) => order[a.soi_type] - order[b.soi_type] || a.interference.localeCompare(b.interference)).forEach(group => {
    const row = document.createElement('div'); row.className = 'experiment-group';
    const label = document.createElement('div'); label.className = 'group-label';
    const title = document.createElement('strong'); title.textContent = `${group.soi_type === 'OFDMQPSK' ? 'OFDM-QPSK' : group.soi_type} · ${group.interference}`;
    const count = document.createElement('span'); count.textContent = `${group.samples} 条样本`;
    label.append(title, count);
    const bars = document.createElement('div'); bars.className = 'group-bars';
    for (const [kind, value, name] of [['input', group.input_ber, '输入 BER'], ['output', group.output_ber, '恢复后 BER']]) {
      const line = document.createElement('div'); line.className = 'group-bar-line';
      const track = document.createElement('div'); track.className = 'group-bar-track';
      const fill = document.createElement('div'); fill.className = `group-bar-fill ${kind}`; fill.style.width = `${Math.min(100, Math.max(0, Number(value) * 100))}%`;
      const number = document.createElement('span'); number.textContent = percent(value);
      track.setAttribute('role', 'img'); track.setAttribute('aria-label', `${name} ${percent(value)}`);
      track.append(fill); line.append(track, number); bars.append(line);
    }
    row.append(label, bars); groups.append(row);
  });
  $('spectrum-experiment-metrics').replaceChildren(
    experimentMetric('样本总数', String(spectrum.sample_count), '宽带频谱样本'),
    experimentMetric('检测精确率', percent(spectrum.detection_precision), '正确检测 / 全部检测'),
    experimentMetric('检测召回率', percent(spectrum.detection_recall), '正确检测 / 全部真实目标'),
    experimentMetric('调制家族准确率', percent(spectrum.matched_family_accuracy), '匹配且成功分类的目标')
  );
  $('spectrum-gt').textContent = `${spectrum.counts.gt} 个`;
  $('spectrum-matched').textContent = `${spectrum.counts.matched_classified} 个`;
  $('spectrum-family-correct').textContent = `${spectrum.counts.family_correct} 个`;
}
fetch('experiment-data.json', {cache:'no-store'})
  .then(response => { if (!response.ok) throw Error('实验结果文件不可用'); return response.json(); })
  .then(renderExperiments)
  .catch(() => { for (const id of ['sep-experiment-metrics', 'spectrum-experiment-metrics']) $(id).textContent = '实验结果暂时无法加载'; });
