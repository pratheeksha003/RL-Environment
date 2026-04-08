import streamlit as st
import numpy as np
import random
import time

st.set_page_config(
    page_title="🔥 Fire Escape Simulator",
    layout="centered",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Orbitron:wght@700&display=swap');

html, body, [class*="css"] {
    font-family: 'Share Tech Mono', monospace;
    background-color: #0c0c0c;
    color: #e0e0d0;
}

h1 {
    font-family: 'Orbitron', monospace !important;
    color: #ff6a00 !important;
    letter-spacing: 4px;
    font-size: 1.5rem !important;
}

.stButton > button {
    font-family: 'Share Tech Mono', monospace;
    background: transparent;
    border: 1px solid #444;
    color: #aaa;
    letter-spacing: 1px;
    font-size: 12px;
    padding: 6px 18px;
    transition: all 0.15s;
}
.stButton > button:hover {
    border-color: #ff6a00;
    color: #ff6a00;
    background: rgba(255,106,0,0.08);
}

.stSelectbox > div > div {
    background: #111;
    border: 1px solid #333;
    color: #aaa;
    font-family: 'Share Tech Mono', monospace;
}

.stSlider > div {
    color: #aaa;
}

.metric-card {
    background: #111;
    border: 1px solid #222;
    border-radius: 4px;
    padding: 12px 16px;
    text-align: center;
    font-family: 'Share Tech Mono', monospace;
}
.metric-label {
    font-size: 10px;
    color: #555;
    letter-spacing: 2px;
    margin-bottom: 4px;
}
.metric-value {
    font-size: 22px;
    color: #e0e0d0;
    font-weight: bold;
}

.grid-container {
    display: grid;
    grid-template-columns: repeat(6, 72px);
    gap: 5px;
    margin: 16px 0;
}
.cell {
    width: 72px;
    height: 72px;
    border-radius: 5px;
    border: 1px solid #1e1e1e;
    background: #111;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 28px;
    transition: background 0.2s;
}
.cell-agent  { background: #0d2b45; border-color: #378ADD; }
.cell-fire   { background: #2d0a00; border-color: #ff4422; }
.cell-goal   { background: #0a1e0a; border-color: #639922; }
.cell-dead   { background: #3d0000; border-color: #ff0000; }

.status-bar {
    padding: 8px 14px;
    border-radius: 3px;
    font-size: 12px;
    letter-spacing: 1px;
    border-left: 3px solid transparent;
    margin-bottom: 12px;
}
.status-idle { border-left-color:#333; background:#111; color:#555; }
.status-win  { border-left-color:#639922; background:rgba(99,153,34,0.1); color:#97C459; }
.status-lose { border-left-color:#ff4422; background:rgba(255,68,34,0.1); color:#ff8866; }
.status-info { border-left-color:#378ADD; background:rgba(55,138,221,0.07); color:#85B7EB; }

.legend {
    display: flex;
    gap: 18px;
    flex-wrap: wrap;
    font-size: 10px;
    color: #555;
    margin-bottom: 14px;
    letter-spacing: 1px;
}
.legend-item { display: flex; align-items: center; gap: 5px; }
.legend-dot {
    width: 11px; height: 11px;
    border-radius: 2px;
    border: 1px solid;
    display: inline-block;
}

.kbd {
    display: inline-block;
    background: #1a1a1a;
    border: 1px solid #333;
    border-radius: 3px;
    padding: 1px 7px;
    font-size: 11px;
    color: #888;
    margin: 2px;
}

section[data-testid="stSidebar"] {
    background: #0e0e0e;
    border-right: 1px solid #1e1e1e;
}
</style>
""", unsafe_allow_html=True)


# ─── Constants ────────────────────────────────────────────────────────────────
SIZE = 6
ACTIONS = [(-1,0),(1,0),(0,-1),(0,1)]   # up, down, left, right
ALPHA, GAMMA, EPSILON = 0.7, 0.9, 0.3
TRAIN_EPISODES = 1500
ICONS = {"agent": "🧑", "fire": "🔥", "goal": "🚪", "empty": "", "dead": "💀"}


# ─── RL helpers ───────────────────────────────────────────────────────────────
def take_step(pos, action):
    x, y = pos
    dx, dy = ACTIONS[action]
    nx, ny = x + dx, y + dy
    return [nx, ny] if 0 <= nx < SIZE and 0 <= ny < SIZE else list(pos)


def spread_fire(fire):
    new = [list(f) for f in fire]
    for f in fire:
        if random.random() < 0.22:
            dx, dy = random.choice(ACTIONS)
            nx, ny = f[0] + dx, f[1] + dy
            if 0 <= nx < SIZE and 0 <= ny < SIZE and [nx, ny] not in new:
                new.append([nx, ny])
    return new


def in_fire(pos, fire):
    return list(pos) in fire


# ─── Train Q-table (cached so it only runs once) ──────────────────────────────
@st.cache_resource
def train_q():
    Q = np.zeros((SIZE, SIZE, 4))
    for _ in range(TRAIN_EPISODES):
        agent = [0, 0]
        goal  = [SIZE-1, SIZE-1]
        fire  = [[2, 2], [2, 3]]
        for _ in range(50):
            x, y = agent
            a = (random.randint(0, 3) if random.random() < EPSILON
                 else int(np.argmax(Q[x, y])))
            npos = take_step(agent, a)
            fire = spread_fire(fire)
            r = (100  if npos == goal else
                 -100 if in_fire(npos, fire) else -1)
            nx, ny = npos
            Q[x, y, a] += ALPHA * (r + GAMMA * np.max(Q[nx, ny]) - Q[x, y, a])
            agent = npos
            if r != -1:
                break
    return Q

Q = train_q()


# ─── Session state init ───────────────────────────────────────────────────────
def init_state():
    if "agent" not in st.session_state:
        st.session_state.agent  = [0, 0]
        st.session_state.fire   = [[2, 2], [2, 3]]
        st.session_state.goal   = [SIZE-1, SIZE-1]
        st.session_state.score  = 0
        st.session_state.steps  = 0
        st.session_state.status = ("idle", "[ waiting — find the exit ]")
        st.session_state.game_over = False

init_state()


# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ SETTINGS")
    mode  = st.selectbox("Mode", ["AI Auto", "Manual Play"], key="mode_select")
    speed = st.slider("AI Speed (ms)", 300, 1200, 600, step=100)
    st.markdown("---")

    if st.button("🔄 Reset Game"):
        st.session_state.agent     = [0, 0]
        st.session_state.fire      = [[2, 2], [2, 3]]
        st.session_state.steps     = 0
        st.session_state.game_over = False
        st.session_state.status    = ("idle", "[ game reset ]")
        st.rerun()

    st.markdown("---")
    st.markdown("""
**HOW IT WORKS**

The AI uses **Q-Learning** (RL) trained over 1,500 episodes.
It learns to navigate from `[0,0]` to the exit `[5,5]`
while avoiding spreading fire.

**Reward structure:**
- Reach exit: `+100`
- Hit fire: `-100`
- Each step: `-1`
""")


# ─── Title ────────────────────────────────────────────────────────────────────
st.markdown("# 🔥 FIRE ESCAPE")
st.markdown('<p style="font-size:11px;color:#555;letter-spacing:3px;margin-top:-12px">AI REINFORCEMENT LEARNING SIMULATOR</p>', unsafe_allow_html=True)


# ─── Stats row ────────────────────────────────────────────────────────────────
c1, c2 = st.columns(2)
with c1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">SCORE</div>
        <div class="metric-value">{st.session_state.score}</div>
    </div>""", unsafe_allow_html=True)
with c2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">STEPS</div>
        <div class="metric-value">{st.session_state.steps}</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# ─── Legend ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="legend">
  <div class="legend-item"><span class="legend-dot" style="background:#0d2b45;border-color:#378ADD"></span>agent</div>
  <div class="legend-item"><span class="legend-dot" style="background:#2d0a00;border-color:#ff4422"></span>fire</div>
  <div class="legend-item"><span class="legend-dot" style="background:#0a1e0a;border-color:#639922"></span>exit</div>
  <div class="legend-item"><span class="legend-dot" style="background:#111;border-color:#222"></span>empty</div>
</div>
""", unsafe_allow_html=True)


# ─── Grid renderer ────────────────────────────────────────────────────────────
def render_grid():
    agent = st.session_state.agent
    fire  = st.session_state.fire
    goal  = st.session_state.goal

    html = '<div class="grid-container">'
    for i in range(SIZE):
        for j in range(SIZE):
            is_agent = (agent == [i, j])
            is_fire  = in_fire([i, j], fire)
            is_goal  = (goal  == [i, j])

            if is_agent and is_fire:
                css, icon = "cell cell-dead",  ICONS["dead"]
            elif is_agent:
                css, icon = "cell cell-agent", ICONS["agent"]
            elif is_fire:
                css, icon = "cell cell-fire",  ICONS["fire"]
            elif is_goal:
                css, icon = "cell cell-goal",  ICONS["goal"]
            else:
                css, icon = "cell", ICONS["empty"]

            html += f'<div class="{css}">{icon}</div>'
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)

render_grid()


# ─── Status bar ───────────────────────────────────────────────────────────────
stype, smsg = st.session_state.status
st.markdown(f'<div class="status-bar status-{stype}">{smsg}</div>', unsafe_allow_html=True)


# ─── Manual play controls ─────────────────────────────────────────────────────
if mode == "Manual Play" and not st.session_state.game_over:
    st.markdown("**KEYBOARD / BUTTON CONTROLS**")
    _, col_up, _ = st.columns([1, 1, 1])
    with col_up:
        if st.button("⬆️  UP", use_container_width=True):
            st.session_state.agent = take_step(st.session_state.agent, 0)
            st.session_state.steps += 1

    col_l, col_d, col_r = st.columns(3)
    with col_l:
        if st.button("⬅️  LEFT", use_container_width=True):
            st.session_state.agent = take_step(st.session_state.agent, 2)
            st.session_state.steps += 1
    with col_d:
        if st.button("⬇️  DOWN", use_container_width=True):
            st.session_state.agent = take_step(st.session_state.agent, 1)
            st.session_state.steps += 1
    with col_r:
        if st.button("➡️  RIGHT", use_container_width=True):
            st.session_state.agent = take_step(st.session_state.agent, 3)
            st.session_state.steps += 1

    # Spread fire after every manual move
    st.session_state.fire = spread_fire(st.session_state.fire)

    st.markdown("""
    <p style="font-size:10px;color:#444;letter-spacing:1px;margin-top:6px">
    Use buttons above — or press <span class="kbd">W A S D</span> / arrow keys
    (click anywhere on the page first to capture keyboard focus)
    </p>
    """, unsafe_allow_html=True)


# ─── AI auto-move ─────────────────────────────────────────────────────────────
if mode == "AI Auto" and not st.session_state.game_over:
    x, y = st.session_state.agent
    a = (random.randint(0, 3) if random.random() < 0.12
         else int(np.argmax(Q[x, y])))
    new_pos = take_step(st.session_state.agent, a)
    if new_pos == st.session_state.agent:          # stuck — explore
        new_pos = take_step(st.session_state.agent, random.randint(0, 3))
    st.session_state.agent = new_pos
    st.session_state.fire  = spread_fire(st.session_state.fire)
    st.session_state.steps += 1


# ─── Win / Lose checks ────────────────────────────────────────────────────────
def check_state():
    agent = st.session_state.agent
    fire  = st.session_state.fire
    goal  = st.session_state.goal

    if agent == goal:
        st.session_state.score += 10
        st.session_state.status = ("win", "[ ✅ ESCAPED! +10 pts — resetting... ]")
        st.session_state.game_over = True
        return "win"

    if in_fire(agent, fire):
        st.session_state.score -= 5
        st.session_state.status = ("lose", "[ 🔥 BURNED! -5 pts — resetting... ]")
        st.session_state.game_over = True
        return "lose"

    st.session_state.status = ("info", f"[ step {st.session_state.steps} — navigating... ]")
    return "ok"

result = check_state()


# ─── Auto-reset after win/lose ────────────────────────────────────────────────
if result in ("win", "lose"):
    time.sleep(1.5)
    st.session_state.agent     = [0, 0]
    st.session_state.fire      = [[2, 2], [2, 3]]
    st.session_state.steps     = 0
    st.session_state.game_over = False
    st.session_state.status    = ("idle", "[ new round — find the exit ]")
    st.rerun()


# ─── AI auto-refresh loop ─────────────────────────────────────────────────────
if mode == "AI Auto" and not st.session_state.game_over:
    time.sleep(speed / 1000)
    st.rerun()
