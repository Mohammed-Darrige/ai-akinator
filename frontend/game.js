/* ═══════════════════════════════════════
   Master Inquisitor Engine – Game Logic v4
   Single-Call Belief-State Architecture
   ═══════════════════════════════════════ */

let sessionId = null;
let questionCount = 0;
let isWaiting = false;

/* ─── Screen Management ─── */
function showScreen(id) {
  document.querySelectorAll('.screen').forEach(s => s.classList.remove('active'));
  const target = document.getElementById(id);
  if (target) target.classList.add('active');
}

/* ─── Terminal Logic ─── */
function clearTerminal() {
  const area = document.getElementById('reasoning-area');
  area.innerHTML = '';
}

function addToTerminal(tag, content, colorClass = '') {
  const area = document.getElementById('reasoning-area');
  const line = document.createElement('div');
  line.className = 'term-line';

  let tagClass = '';
  if (tag === 'OVERRIDE') tagClass = 'tag-warning';
  if (tag === 'SYSTEM')   tagClass = 'tag-info';
  if (tag === 'SUCCESS')  tagClass = 'tag-success';
  if (tag === 'GUARD')    tagClass = 'tag-warning';

  line.innerHTML = `<span class="term-tag ${tagClass}">${tag}</span> <span class="term-val ${colorClass}">${content}</span>`;
  area.appendChild(line);

  setTimeout(() => { area.scrollTop = area.scrollHeight; }, 50);
}

function processReasoning(reasoning) {
  if (!reasoning) return;
  addToTerminal('ANALYSIS', reasoning.analysis || '—');
  addToTerminal('STRATEGY', reasoning.strategy || '—', 'text-accent2');
  const conf = typeof reasoning.confidence === 'number'
    ? (reasoning.confidence * 100).toFixed(1) + '%'
    : '—';
  const cands = reasoning.candidates_remaining ?? '—';
  addToTerminal('CONFIDENCE', conf, reasoning.confidence >= 0.88 ? 'text-success' : '');
  addToTerminal('CANDIDATES', `~${cands} remaining`);
}

/* ─── Chat Logic ─── */
function addChatMessage(text, role = 'ai') {
  const area = document.getElementById('chat-area');
  const msg = document.createElement('div');
  msg.className = `msg msg-${role}`;
  msg.textContent = text;
  area.appendChild(msg);
  area.scrollTop = area.scrollHeight;
}

/* ─── Rate-limit countdown helper ─── */
async function waitWithCountdown(seconds, label) {
  const loaderText = document.querySelector('#screen-loading p');
  const original = loaderText ? loaderText.textContent : '';
  let remaining = seconds;

  return new Promise(resolve => {
    const tick = () => {
      if (loaderText) loaderText.textContent = `${label}: ${remaining}s`;
      if (remaining <= 0) {
        if (loaderText) loaderText.textContent = original;
        resolve();
        return;
      }
      remaining--;
      setTimeout(tick, 1000);
    };
    tick();
  });
}

/* ─── START GAME ─── */
async function startGame() {
  const old = sessionId;
  sessionId = null;
  questionCount = 0;
  isWaiting = false;

  document.getElementById('chat-area').innerHTML = '';
  clearTerminal();
  addToTerminal('SYSTEM', 'Belief-State Engine initializing...');

  showScreen('screen-loading');

  try {
    const startUrl = `/api/v1/start${old ? '?old_session_id=' + old : ''}`;
    const res = await fetch(startUrl, { method: 'POST' });

    if (res.status === 503) {
      const retryAfter = parseInt(res.headers.get('retry-after') || '60', 10);
      addToTerminal('GUARD', `API limit — auto-retrying in ${retryAfter}s`);
      await waitWithCountdown(retryAfter, 'API limiti — yeniden deneniyor');
      return startGame();
    }

    if (!res.ok) {
      const detail = await res.json().catch(() => ({}));
      throw new Error(detail.detail || `HTTP ${res.status}`);
    }

    const data = await res.json();
    sessionId = data.session_id;
    questionCount = 1;

    showScreen('screen-game');
    document.getElementById('q-number').textContent = questionCount;

    addToTerminal('READY', 'Engine: ONLINE | Guards: ACTIVE');
    processReasoning(data.reasoning);
    addChatMessage(data.question);

  } catch (err) {
    showScreen('screen-intro');
    console.error(err);
    showToast('Bağlantı hatası: ' + err.message);
  }
}

/* ─── SEND ANSWER ─── */
async function sendAnswer(ans) {
  if (isWaiting || !sessionId) return;
  isWaiting = true;

  setButtonsDisabled(true);
  addChatMessage(ans, 'user');
  addToTerminal('INPUT', `User: ${ans}`);

  try {
    const res = await fetch(`/api/v1/ask?session_id=${sessionId}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ user_answer: ans })
    });

    if (res.status === 503) {
      const retryAfter = parseInt(res.headers.get('retry-after') || '60', 10);
      addToTerminal('GUARD', `API limit — retrying in ${retryAfter}s`);
      showToast(`API limiti — ${retryAfter}s sonra tekrar denenecek`);
      await new Promise(r => setTimeout(r, retryAfter * 1000));
      isWaiting = false;
      setButtonsDisabled(false);
      return sendAnswer(ans);
    }

    if (!res.ok) {
      const detail = await res.json().catch(() => ({}));
      throw new Error(detail.detail || `HTTP ${res.status}`);
    }

    const data = await res.json();
    processReasoning(data.reasoning);

    if (data.action === 'ask_question') {
      questionCount++;
      document.getElementById('q-number').textContent = questionCount;
      addChatMessage(data.question);
      setButtonsDisabled(false);
    } else if (data.action === 'guess') {
      addToTerminal('SUCCESS', 'Belief entropy minimized. Identity confirmed.');
      document.getElementById('result-profession').textContent = data.guess || '—';
      document.getElementById('result-q-count').textContent = questionCount;
      showScreen('screen-result');
    }

  } catch (err) {
    console.error(err);
    showToast('Hata: ' + err.message);
    setButtonsDisabled(false);
  } finally {
    isWaiting = false;
  }
}

/* ─── Helpers ─── */
function setButtonsDisabled(disabled) {
  document.querySelectorAll('.btn-ans').forEach(b => b.disabled = disabled);
}

function restartGame() {
  if (sessionId) {
    fetch(`/api/v1/restart?session_id=${sessionId}`, { method: 'POST' }).catch(() => {});
  }
  sessionId = null;
  showScreen('screen-intro');
}

function showToast(msg) {
  const toast = document.getElementById('toast');
  const toastMsg = document.getElementById('toast-msg');
  if (!toast || !toastMsg) { alert(msg); return; }
  toastMsg.textContent = msg;
  toast.classList.add('visible');
  setTimeout(() => toast.classList.remove('visible'), 3500);
}
