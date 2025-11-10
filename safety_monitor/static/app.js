const personsEl = document.getElementById('stat-persons');
const relicsEl = document.getElementById('stat-relics');
const dangerousEl = document.getElementById('stat-dangerous');
const fpsEl = document.getElementById('stat-fps');
const alertsList = document.getElementById('alerts-list');
const historyList = document.getElementById('history-list');
const sessionAlertsEl = document.getElementById('session-alerts');
const sessionIntrusionsEl = document.getElementById('session-intrusions');
const sessionDangerousEl = document.getElementById('session-dangerous');
const sessionUptimeEl = document.getElementById('session-uptime');
const videoAlert = document.getElementById('video-alert');
const alertSound = document.getElementById('alert-sound');

const personStatCard = personsEl.closest('.stat');
const relicStatCard = relicsEl.closest('.stat');
const dangerousStatCard = dangerousEl.closest('.stat');

let lastAlertTimestamp = 0;

function formatSeconds(value) {
  const total = Math.floor(value);
  const hours = Math.floor(total / 3600);
  const minutes = Math.floor((total % 3600) / 60);
  const seconds = total % 60;
  if (hours > 0) {
    return `${hours}h ${minutes}m ${seconds}s`;
  }
  if (minutes > 0) {
    return `${minutes}m ${seconds}s`;
  }
  return `${seconds}s`;
}

function playAlertSound() {
  if (!alertSound) return;
  alertSound.currentTime = 0;
  alertSound.play().catch(() => {});
}

function updateStats(data) {
  const persons = data.persons ?? 0;
  const relics = data.relics ?? 0;
  const dangerous = data.dangerous ?? 0;

  personsEl.textContent = persons;
  relicsEl.textContent = relics;
  dangerousEl.textContent = dangerous;
  fpsEl.textContent = (data.fps ?? 0).toFixed(1);

  if (personStatCard) {
    personStatCard.classList.toggle('active', persons > 0);
  }
  if (relicStatCard) {
    relicStatCard.classList.toggle('active', relics > 0);
  }
  if (dangerousStatCard) {
    dangerousStatCard.classList.toggle('active', dangerous > 0);
  }

  const session = data.session ?? {};
  sessionAlertsEl.textContent = session.total_alerts ?? 0;
  sessionIntrusionsEl.textContent = session.total_intrusions ?? 0;
  sessionDangerousEl.textContent = session.total_dangerous ?? 0;
  sessionUptimeEl.textContent = formatSeconds(session.elapsed ?? 0);

  alertsList.innerHTML = '';
  const alerts = data.alerts ?? [];
  alerts.forEach((alert) => {
    const li = document.createElement('li');
    li.textContent = alert;
    li.classList.add('active');
    alertsList.appendChild(li);
  });

  const history = session.recent_alerts ?? [];
  historyList.innerHTML = '';
  history
    .slice()
    .reverse()
    .forEach(([timestamp, message]) => {
      const li = document.createElement('li');
      const date = new Date(timestamp * 1000);
      const timeLabel = date.toLocaleTimeString();
      li.textContent = `${timeLabel} · ${message}`;
      historyList.appendChild(li);
    });

  if (alerts.length > 0) {
    videoAlert.classList.remove('hidden');
    if (Date.now() - lastAlertTimestamp > 1000) {
      playAlertSound();
      lastAlertTimestamp = Date.now();
    }
  } else {
    videoAlert.classList.add('hidden');
  }
}

async function fetchStatus() {
  try {
    const response = await fetch('/api/status');
    if (!response.ok) {
      throw new Error(`请求失败: ${response.status}`);
    }
    const data = await response.json();
    updateStats(data);
  } catch (error) {
    console.error('获取状态失败', error);
  }
}

setInterval(fetchStatus, 800);
fetchStatus();
