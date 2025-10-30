const socket = io();

let isRunning = false;
let relicData = [];
let personData = [];
let dangerData = [];
let lastUpdateTime = Date.now();
let frameCount = 0;

const videoStream = document.getElementById('video-stream');
const btnStart = document.getElementById('btn-start');
const btnStop = document.getElementById('btn-stop');
const btnClear = document.getElementById('btn-clear');
const btnScreenshot = document.getElementById('btn-screenshot');
const statusEl = document.getElementById('status');
const fpsEl = document.getElementById('fps');
const totalRelicsEl = document.getElementById('total-relics');
const selectedCountEl = document.getElementById('selected-count');
const personCountEl = document.getElementById('person-count');
const dangerCountEl = document.getElementById('danger-count');
const relicListEl = document.getElementById('relic-list');
const personListEl = document.getElementById('person-list');
const alertListEl = document.getElementById('alert-list');
const personPanel = document.getElementById('person-panel');

btnStart.addEventListener('click', startDetection);
btnStop.addEventListener('click', stopDetection);
btnClear.addEventListener('click', clearSelection);
btnScreenshot.addEventListener('click', takeScreenshot);
videoStream.addEventListener('click', handleVideoClick);

socket.on('connect', () => {
    console.log('Connected to server');
    updateStatus('已连接', 'success');
});

socket.on('disconnect', () => {
    console.log('Disconnected from server');
    updateStatus('连接断开', 'error');
});

socket.on('data_update', (data) => {
    relicData = data.relics || [];
    personData = data.persons || [];
    dangerData = data.dangers || [];
    updateAllLists();
    updateFPS();
});

socket.on('selection_changed', (data) => {
    console.log('Selection changed:', data);
    updateAllLists();
});

async function startDetection() {
    try {
        const sourceType = document.querySelector('input[name="source-type"]:checked').value;
        
        const requestData = {
            source_type: sourceType
        };
        
        if (sourceType === 'camera') {
            requestData.camera_id = parseInt(document.getElementById('camera-id').value) || 0;
        } else if (sourceType === 'video') {
            const videoPath = document.getElementById('video-path').value.trim();
            if (!videoPath) {
                alert('请输入视频文件路径');
                return;
            }
            requestData.video_path = videoPath;
        }
        
        const response = await fetch('/api/start', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify(requestData)
        });
        const result = await response.json();
        
        if (result.status === 'success') {
            isRunning = true;
            videoStream.src = '/video_feed?' + new Date().getTime();
            btnStart.disabled = true;
            btnStop.disabled = false;
            updateStatus('检测运行中', 'success');
            personPanel.style.display = 'block';
            startAlertPolling();
        } else {
            alert('启动失败: ' + result.message);
        }
    } catch (error) {
        alert('启动失败: ' + error.message);
    }
}

async function stopDetection() {
    try {
        const response = await fetch('/api/stop', {method: 'POST'});
        const result = await response.json();
        
        if (result.status === 'success') {
            isRunning = false;
            videoStream.src = '';
            btnStart.disabled = false;
            btnStop.disabled = true;
            updateStatus('已停止', 'warning');
            stopAlertPolling();
        }
    } catch (error) {
        alert('停止失败: ' + error.message);
    }
}

async function clearSelection() {
    try {
        const response = await fetch('/api/clear_selection', {method: 'POST'});
        const result = await response.json();
        
        if (result.status === 'success') {
            console.log('Selection cleared');
        }
    } catch (error) {
        console.error('Clear failed:', error);
    }
}

function takeScreenshot() {
    if (!isRunning) {
        alert('请先启动检测');
        return;
    }
    
    const canvas = document.createElement('canvas');
    canvas.width = videoStream.naturalWidth || videoStream.width;
    canvas.height = videoStream.naturalHeight || videoStream.height;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(videoStream, 0, 0);
    
    canvas.toBlob((blob) => {
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `screenshot_${Date.now()}.jpg`;
        a.click();
        URL.revokeObjectURL(url);
    }, 'image/jpeg', 0.95);
}

function handleVideoClick(event) {
    if (!isRunning) return;
    
    const rect = videoStream.getBoundingClientRect();
    const scaleX = videoStream.naturalWidth / rect.width;
    const scaleY = videoStream.naturalHeight / rect.height;
    
    const x = (event.clientX - rect.left) * scaleX;
    const y = (event.clientY - rect.top) * scaleY;
    
    socket.emit('click_video', {x: Math.round(x), y: Math.round(y)});
}

function updateAllLists() {
    updateRelicList();
    updatePersonList();
    updateStats();
}

function updateRelicList() {
    if (relicData.length === 0) {
        relicListEl.innerHTML = '<div class="empty-state">未检测到文物</div>';
        totalRelicsEl.textContent = '0';
        selectedCountEl.textContent = '0';
        return;
    }
    
    const selectedCount = relicData.filter(r => r.selected).length;
    totalRelicsEl.textContent = relicData.length;
    selectedCountEl.textContent = selectedCount;
    
    relicListEl.innerHTML = relicData.map(relic => `
        <div class="relic-item ${relic.selected ? 'selected' : ''}" 
             onclick="toggleRelicSelection(${relic.track_id})">
            <div class="relic-header">
                <span class="relic-id">ID: ${relic.track_id}</span>
                <span class="relic-confidence">${relic.confidence}%</span>
            </div>
            <div class="relic-info">
                <span class="relic-class">${relic.class_name}</span>
                ${relic.selected ? '<span class="badge-selected">已选择</span>' : ''}
            </div>
        </div>
    `).join('');
}

function updatePersonList() {
    if (personData.length === 0) {
        personListEl.innerHTML = '<div class="empty-state">未检测到人员</div>';
        return;
    }
    
    personListEl.innerHTML = personData.map(person => {
        const riskClass = person.is_risky ? 'person-risky' : '';
        const poseIcon = person.has_pose ? '(P)' : '';
        return `
            <div class="person-item ${riskClass}">
                <div class="person-header">
                    <span class="person-id">人员 ${person.track_id || '?'} ${poseIcon}</span>
                    ${person.is_risky ? '<span class="badge-danger">风险</span>' : ''}
                </div>
                ${person.risk_messages.length > 0 ? `
                    <div class="person-risks">
                        ${person.risk_messages.map(msg => `<div class="risk-msg">${msg}</div>`).join('')}
                    </div>
                ` : ''}
            </div>
        `;
    }).join('');
}

function updateStats() {
    personCountEl.textContent = personData.length;
    dangerCountEl.textContent = dangerData.length;
}

async function toggleRelicSelection(trackId) {
    try {
        const response = await fetch('/api/toggle_selection', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({track_id: trackId})
        });
        const result = await response.json();
        
        if (result.status === 'success') {
            const relic = relicData.find(r => r.track_id === trackId);
            if (relic) {
                relic.selected = result.selected;
                updateRelicList();
            }
        }
    } catch (error) {
        console.error('Toggle failed:', error);
    }
}

let alertPollingInterval;

function startAlertPolling() {
    alertPollingInterval = setInterval(fetchAlerts, 2000);
}

function stopAlertPolling() {
    if (alertPollingInterval) {
        clearInterval(alertPollingInterval);
    }
}

async function fetchAlerts() {
    try {
        const response = await fetch('/api/alerts');
        const data = await response.json();
        updateAlertList(data.alerts || []);
    } catch (error) {
        console.error('Fetch alerts failed:', error);
    }
}

function updateAlertList(alerts) {
    if (alerts.length === 0) {
        alertListEl.innerHTML = '<div class="empty-state">暂无报警</div>';
        return;
    }
    
    const recentAlerts = alerts.slice(-10).reverse();
    
    alertListEl.innerHTML = recentAlerts.map(alert => {
        const time = new Date(alert.timestamp * 1000).toLocaleTimeString('zh-CN');
        const typeIcon = alert.type === 'danger' ? '[!]' : '[X]';
        return `
            <div class="alert-item">
                <div class="alert-time">${time} ${typeIcon}</div>
                <div class="alert-message">${alert.message}</div>
            </div>
        `;
    }).join('');
}

function updateFPS() {
    frameCount++;
    const now = Date.now();
    const elapsed = (now - lastUpdateTime) / 1000;
    
    if (elapsed >= 1.0) {
        const fps = Math.round(frameCount / elapsed);
        fpsEl.textContent = `FPS: ${fps}`;
        frameCount = 0;
        lastUpdateTime = now;
    }
}

function updateStatus(text, type) {
    statusEl.textContent = text;
    statusEl.className = `status-${type}`;
}
