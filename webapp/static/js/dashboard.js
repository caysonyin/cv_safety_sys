const state = {
  selected: new Set(),
  seenAlerts: new Set(),
  polling: false,
  audioUnlocked: false,
};

const elements = {
  sidebarMonitor: document.getElementById('sidebar-monitor-state'),
  sidebarDuration: document.getElementById('sidebar-session-duration'),
  sidebarFps: document.getElementById('sidebar-fps'),
  badgeStage: document.getElementById('badge-stage'),
  badgeAlertCount: document.getElementById('badge-alert-count'),
  overlayState: document.getElementById('overlay-monitor-state'),
  overlayDanger: document.getElementById('overlay-danger-count'),
  overlayPersonRisk: document.getElementById('overlay-person-risk'),
  overlayPersonTotal: document.getElementById('overlay-person-total'),
  overlayRelicSelected: document.getElementById('overlay-relic-selected'),
  totalAlerts: document.getElementById('total-alerts'),
  dailyTotal: document.getElementById('daily-total'),
  dailyIntrusions: document.getElementById('daily-intrusions'),
  dailyDangerous: document.getElementById('daily-dangerous'),
  weeklyTotal: document.getElementById('weekly-total'),
  weeklyIntrusions: document.getElementById('weekly-intrusions'),
  weeklyDangerous: document.getElementById('weekly-dangerous'),
  statFps: document.getElementById('stat-fps-value'),
  statStatusMessage: document.getElementById('stat-status-message'),
  liveAlerts: document.getElementById('live-alerts'),
  liveAlertCount: document.getElementById('live-alert-count'),
  historyList: document.getElementById('history-list'),
  relicGrid: document.getElementById('relic-grid'),
  relicCount: document.getElementById('relic-count'),
  protectAll: document.getElementById('protect-all'),
  clearSelection: document.getElementById('clear-selection'),
};

function formatNumber(value, digits = 0) {
  return Number(value || 0).toFixed(digits);
}

function formatTimeLabel(isoString) {
  if (!isoString) return '--:--:--';
  const date = new Date(isoString);
  if (Number.isNaN(date.getTime())) return '--:--:--';
  return date.toLocaleTimeString('zh-CN', { hour12: false });
}

function ensureAudioContext() {
  if (state.audioUnlocked) {
    return state.audioContext;
  }

  const AudioContext = window.AudioContext || window.webkitAudioContext;
  if (!AudioContext) return null;
  state.audioContext = new AudioContext();
  const unlock = () => {
    if (state.audioContext?.state === 'suspended') {
      state.audioContext.resume();
    }
    state.audioUnlocked = true;
    document.body.removeEventListener('click', unlock);
  };
  document.body.addEventListener('click', unlock, { once: true });
  return state.audioContext;
}

function playAlertTone() {
  const ctx = ensureAudioContext();
  if (!ctx) return;
  if (ctx.state === 'suspended') {
    ctx.resume();
  }
  const oscillator = ctx.createOscillator();
  const gain = ctx.createGain();
  oscillator.type = 'triangle';
  oscillator.frequency.setValueAtTime(880, ctx.currentTime);
  oscillator.frequency.exponentialRampToValueAtTime(660, ctx.currentTime + 0.4);
  gain.gain.setValueAtTime(0.0001, ctx.currentTime);
  gain.gain.exponentialRampToValueAtTime(0.35, ctx.currentTime + 0.02);
  gain.gain.exponentialRampToValueAtTime(0.0001, ctx.currentTime + 0.6);
  oscillator.connect(gain);
  gain.connect(ctx.destination);
  oscillator.start();
  oscillator.stop(ctx.currentTime + 0.65);
}

function renderLiveAlerts(alerts) {
  elements.liveAlerts.innerHTML = '';
  const fragment = document.createDocumentFragment();
  let newAlerts = 0;

  alerts.forEach((alert) => {
    const item = document.createElement('li');
    item.className = `alert-item ${alert.category || 'notice'}`;
    item.dataset.alertId = alert.id;

    const content = document.createElement('div');
    content.className = 'message';
    content.textContent = alert.message;
    item.appendChild(content);

    const meta = document.createElement('div');
    meta.className = 'meta';
    meta.innerHTML = `<span>${alert.category === 'intrusion' ? '入侵预警' : alert.category === 'danger' ? '危险携带' : '提示'}</span><span>${formatTimeLabel(alert.timestamp)}</span>`;
    item.appendChild(meta);

    if (!state.seenAlerts.has(alert.id)) {
      item.classList.add('alert-highlight', 'animate__animated', 'animate__fadeInRight');
      newAlerts += 1;
      state.seenAlerts.add(alert.id);
    }

    fragment.appendChild(item);
  });

  elements.liveAlerts.appendChild(fragment);
  elements.liveAlertCount.textContent = `${alerts.length} 条`;

  if (newAlerts > 0) {
    playAlertTone();
  }
}

function renderHistory(history) {
  elements.historyList.innerHTML = '';
  const fragment = document.createDocumentFragment();
  history.forEach((item) => {
    const li = document.createElement('li');
    li.className = 'history-item animate__animated animate__fadeInUp';
    li.innerHTML = `<div class="time">${formatTimeLabel(item.timestamp)}</div><div>${item.message}</div>`;
    fragment.appendChild(li);
  });
  elements.historyList.appendChild(fragment);
}

function renderRelics(relics) {
  elements.relicGrid.innerHTML = '';
  const fragment = document.createDocumentFragment();
  let selectedCount = 0;

  relics.forEach((relic) => {
    const card = document.createElement('div');
    const active = state.selected.has(relic.id) || relic.is_selected;
    if (active) {
      selectedCount += 1;
    }
    card.className = `relic-card${active ? ' active' : ''}`;
    card.dataset.relicId = relic.id;
    card.innerHTML = `
      <div class="indicator"></div>
      <div class="title">文物 #${relic.id}</div>
      <div class="confidence">类别：${relic.label}</div>
      <div class="confidence">置信度：${formatNumber(relic.confidence, 2)}</div>
      <div class="confidence">价值指数：${formatNumber(relic.score, 2)}</div>
      <div class="toggle">${active ? '已纳入保护' : '点击纳入保护'}</div>
    `;
    card.addEventListener('click', () => toggleRelic(relic.id));
    fragment.appendChild(card);
  });

  elements.relicGrid.appendChild(fragment);
  elements.relicCount.textContent = `${relics.length} 项`;
  elements.overlayRelicSelected.textContent = selectedCount;
}

async function toggleRelic(id) {
  if (state.selected.has(id)) {
    state.selected.delete(id);
  } else {
    state.selected.add(id);
  }
  await syncSelection();
}

async function syncSelection() {
  const payload = { selected: Array.from(state.selected) };
  try {
    const response = await fetch('/api/relics/select', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    const result = await response.json();
    if (response.ok && result.selected) {
      state.selected = new Set(result.selected);
    }
  } catch (error) {
    console.error('同步文物选择失败', error);
  }
}

function applyStatus(data) {
  const monitoringText = data.monitoring_active ? '实时监控中' : '待命';
  elements.sidebarMonitor.textContent = monitoringText;
  elements.overlayState.textContent = monitoringText;
  elements.badgeStage.textContent = `阶段：${data.stage}`;
  elements.badgeAlertCount.textContent = `累计报警 ${data.totals.alerts}`;
  elements.totalAlerts.textContent = data.totals.alerts;
  elements.sidebarDuration.textContent = data.session_duration;
  elements.sidebarFps.textContent = `${formatNumber(data.frame_fps, 1)} fps`;
  elements.statFps.textContent = `${formatNumber(data.frame_fps, 1)} fps`;
  elements.overlayDanger.textContent = data.dangerous_items;
  elements.overlayPersonTotal.textContent = data.people.total;
  elements.overlayPersonRisk.textContent = data.people.risky;

  elements.dailyTotal.textContent = data.daily_summary.total;
  elements.dailyIntrusions.textContent = data.daily_summary.intrusions;
  elements.dailyDangerous.textContent = data.daily_summary.dangerous;

  elements.weeklyTotal.textContent = data.weekly_summary.total;
  elements.weeklyIntrusions.textContent = data.weekly_summary.intrusions;
  elements.weeklyDangerous.textContent = data.weekly_summary.dangerous;

  if (data.status_message) {
    elements.statStatusMessage.textContent = data.status_message;
  } else {
    elements.statStatusMessage.textContent = '系统稳定运行中';
  }

  state.selected = new Set(data.selected_relics || []);
  renderRelics(data.relics || []);
  renderLiveAlerts(data.live_alerts || []);
  renderHistory(data.alert_history || []);
}

async function fetchStatus() {
  if (state.polling) return;
  state.polling = true;
  try {
    const response = await fetch('/api/status');
    if (!response.ok) throw new Error('获取状态失败');
    const data = await response.json();
    applyStatus(data);
  } catch (error) {
    console.error(error);
  } finally {
    state.polling = false;
  }
}

async function protectAll() {
  try {
    const response = await fetch('/api/relics/select', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ selected: 'all' }),
    });
    const data = await response.json();
    if (response.ok && data.selected) {
      state.selected = new Set(data.selected);
      await fetchStatus();
    }
  } catch (error) {
    console.error('一键保护失败', error);
  }
}

async function clearSelection() {
  try {
    const response = await fetch('/api/relics/clear', { method: 'POST' });
    const data = await response.json();
    if (response.ok) {
      state.selected = new Set();
      await fetchStatus();
    }
  } catch (error) {
    console.error('清空选择失败', error);
  }
}

function bootstrap() {
  ensureAudioContext();
  fetchStatus();
  setInterval(fetchStatus, 1200);
  elements.protectAll.addEventListener('click', protectAll);
  elements.clearSelection.addEventListener('click', clearSelection);
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', bootstrap);
} else {
  bootstrap();
}
