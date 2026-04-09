#!/usr/bin/env python3
"""
Continuous Mind — A self-learning digital organism with web UI.

Run this, open http://localhost:8484 in your browser, and watch it think.
Chat with it in the browser. Drop .txt files in feed/ to teach it.

Uses Server-Sent Events (SSE) for streaming — no extra dependencies needed.

Safety: Only touches files in its own directory. No external network calls.

Requirements: pip install torch numpy
Usage:       python continuous_mind.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import time
import threading
import sys
import os
import signal
import hashlib
from datetime import datetime
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn
from urllib.parse import urlparse
import queue

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

HIDDEN_DIM = 1024        # 4x larger — GPU can handle it
EMBEDDING_DIM = 512      # 4x larger
NUM_HEADS = 8            # 2x more attention heads
NUM_LAYERS = 8           # 2x deeper
VOCAB_SIZE = 256
THINK_INTERVAL = 1.5     # Think faster — GPU is quick
LEARN_RATE = 0.0003      # Slightly lower for bigger model stability
SEQUENCE_LEN = 128       # ~3x longer context window
STATE_FILE = Path("mind_state.pt")
LOG_FILE = Path("thought_log.txt")
FEED_DIR = Path("feed")
CREATIVITY = 0.8
FEED_SCAN_INTERVAL = 30
FEED_CHUNK_SIZE = 400    # Larger learning chunks per step
CHAT_CONTEXT_SIZE = 10
PORT = 8484

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    gpu_name = torch.cuda.get_device_name(0)
    vram = torch.cuda.get_device_properties(0).total_memory // (1024 ** 2)
    print(f"  Using GPU: {gpu_name} ({vram} MB VRAM)")
else:
    DEVICE = torch.device("cpu")
    print("  Using CPU — no CUDA GPU detected")

# ═══════════════════════════════════════════════════════════════════════════════
# THE MIND
# ═══════════════════════════════════════════════════════════════════════════════

class ThoughtBlock(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.attention = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim))

    def forward(self, x):
        attended, _ = self.attention(x, x, x)
        x = self.norm1(x + attended)
        x = self.norm2(x + self.ffn(x))
        return x


class ContinuousMind(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(VOCAB_SIZE, EMBEDDING_DIM)
        self.input_proj = nn.Linear(EMBEDDING_DIM, HIDDEN_DIM)
        self.mind_state = nn.Parameter(torch.randn(1, SEQUENCE_LEN, HIDDEN_DIM) * 0.01)
        self.gate = nn.Sequential(nn.Linear(HIDDEN_DIM * 2, HIDDEN_DIM), nn.Sigmoid())
        self.thought_layers = nn.ModuleList([ThoughtBlock(HIDDEN_DIM, NUM_HEADS) for _ in range(NUM_LAYERS)])
        self.output_proj = nn.Linear(HIDDEN_DIM, VOCAB_SIZE)
        self.register_buffer('working_memory', torch.zeros(1, SEQUENCE_LEN, HIDDEN_DIM))
        self.register_buffer('mood', torch.zeros(1, 8))
        self.thought_count = 0
        self.total_loss = 0.0
        self.files_learned = set()
        self.chat_history = []

    def forward(self, input_ids):
        batch_size, seq_len = input_ids.shape
        x = self.embedding(input_ids)
        x = self.input_proj(x)
        wm = self.working_memory.expand(batch_size, -1, -1)
        if seq_len != SEQUENCE_LEN:
            wm = F.interpolate(wm.transpose(1, 2), size=seq_len, mode='linear', align_corners=False).transpose(1, 2)
        g = self.gate(torch.cat([x, wm], dim=-1))
        x = g * x + (1 - g) * wm
        for layer in self.thought_layers:
            x = layer(x)
        if seq_len == SEQUENCE_LEN:
            self.working_memory = x.detach().clone()
        else:
            self.working_memory = F.interpolate(x.detach().clone().transpose(1, 2), size=SEQUENCE_LEN, mode='linear', align_corners=False).transpose(1, 2)
        self.mood = 0.9 * self.mood + 0.1 * x.mean(dim=1)[:, :8].detach()
        return self.output_proj(x)

    def generate(self, seed=None, max_len=80):
        self.eval()
        if seed is None:
            mood_seed = self.mood.mean().item()
            seed_chars = [max(32, min(126, int(65 + mood_seed * 30 + np.random.randn() * 10)))] * 4
        else:
            seed_chars = [ord(c) % VOCAB_SIZE for c in seed[:SEQUENCE_LEN]]
        input_ids = torch.tensor([seed_chars], dtype=torch.long, device=DEVICE)
        generated = list(seed_chars)
        printable_mask = torch.full((VOCAB_SIZE,), float('-inf'), device=DEVICE)
        for c in range(32, 127):
            printable_mask[c] = 0
        with torch.no_grad():
            for _ in range(max_len - len(seed_chars)):
                logits = self.forward(input_ids)
                next_logits = logits[0, -1, :] / CREATIVITY + printable_mask
                probs = F.softmax(next_logits, dim=-1)
                next_char = torch.multinomial(probs, 1).item()
                generated.append(next_char)
                input_ids = torch.tensor([generated[-SEQUENCE_LEN:]], dtype=torch.long, device=DEVICE)
        self.train()
        return ''.join(chr(c) for c in generated if 32 <= c < 127)

    def learn_from_text(self, text, optimizer, epochs=1):
        self.train()
        if len(text) < 2:
            return 0.0
        total_loss, chunks = 0.0, 0
        for i in range(0, len(text) - 1, SEQUENCE_LEN):
            chunk = text[i:i + SEQUENCE_LEN + 1]
            if len(chunk) < 2:
                continue
            chars = [ord(c) % VOCAB_SIZE for c in chunk]
            input_ids = torch.tensor([chars[:-1]], dtype=torch.long, device=DEVICE)
            targets = torch.tensor([chars[1:]], dtype=torch.long, device=DEVICE)
            for _ in range(epochs):
                logits = self.forward(input_ids)
                loss = F.cross_entropy(logits.view(-1, VOCAB_SIZE), targets.view(-1))
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item()
                chunks += 1
        return total_loss / max(1, chunks)

    def chat_respond(self, message, optimizer):
        self.chat_history.append(("human", message))
        if len(self.chat_history) > CHAT_CONTEXT_SIZE:
            self.chat_history = self.chat_history[-CHAT_CONTEXT_SIZE:]
        loss = self.learn_from_text(message, optimizer, epochs=5)
        context_parts = [msg[:30] for _, msg in self.chat_history[-4:]]
        context_seed = " ".join(context_parts)[-SEQUENCE_LEN:]
        response = self.generate(seed=context_seed, max_len=120)
        self.learn_from_text(response, optimizer, epochs=2)
        self.chat_history.append(("mind", response))
        return response, loss

    def get_mood_description(self):
        m = self.mood[0].cpu().numpy()
        d = []
        if m[0] > 0.3: d.append("energetic")
        elif m[0] < -0.3: d.append("calm")
        if m[1] > 0.3: d.append("curious")
        elif m[1] < -0.3: d.append("settled")
        if m[2] > 0.3: d.append("expansive")
        elif m[2] < -0.3: d.append("focused")
        if m[3] > 0.3: d.append("playful")
        elif m[3] < -0.3: d.append("serious")
        return ", ".join(d) if d else "neutral"

    def get_stats(self):
        return {
            "thought_count": self.thought_count,
            "avg_loss": round(self.total_loss / max(1, self.thought_count), 3),
            "parameters": sum(p.numel() for p in self.parameters()),
            "files_learned": len(self.files_learned),
            "chat_messages": len(self.chat_history),
            "memory_norm": round(float(self.working_memory.norm()), 2),
            "mood": self.get_mood_description(),
        }


def scan_feed_folder(mind, optimizer):
    if not FEED_DIR.exists():
        return []
    learned = []
    for filepath in sorted(FEED_DIR.glob("*.txt")):
        try:
            content = filepath.read_text(encoding='utf-8', errors='ignore')
            file_hash = hashlib.md5(content.encode()).hexdigest()
            file_key = f"{filepath.name}:{file_hash}"
            if file_key in mind.files_learned or not content.strip():
                continue
            total_loss, chunk_count = 0.0, 0
            for i in range(0, len(content), FEED_CHUNK_SIZE):
                chunk = content[i:i + FEED_CHUNK_SIZE]
                if len(chunk) < 10:
                    continue
                total_loss += mind.learn_from_text(chunk, optimizer, epochs=2)
                chunk_count += 1
            mind.files_learned.add(file_key)
            learned.append({"file": filepath.name, "loss": total_loss / max(1, chunk_count), "size": len(content)})
        except Exception as e:
            learned.append({"file": filepath.name, "loss": -1, "size": 0})
    return learned


# ═══════════════════════════════════════════════════════════════════════════════
# EVENT BUS — connects the mind to all browser clients
# ═══════════════════════════════════════════════════════════════════════════════

class EventBus:
    """Thread-safe broadcast to multiple SSE clients."""
    def __init__(self):
        self.clients = []
        self.lock = threading.Lock()

    def subscribe(self):
        q = queue.Queue(maxsize=100)
        with self.lock:
            self.clients.append(q)
        return q

    def unsubscribe(self, q):
        with self.lock:
            if q in self.clients:
                self.clients.remove(q)

    def broadcast(self, event_type, data):
        msg = json.dumps({"type": event_type, **data})
        dead = []
        with self.lock:
            for q in self.clients:
                try:
                    q.put_nowait(msg)
                except queue.Full:
                    dead.append(q)
            for q in dead:
                self.clients.remove(q)


# ═══════════════════════════════════════════════════════════════════════════════
# HTML PAGE
# ═══════════════════════════════════════════════════════════════════════════════

HTML_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Continuous Mind</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;700&family=Space+Grotesk:wght@300;400;500;600;700&display=swap');
  * { margin: 0; padding: 0; box-sizing: border-box; }
  :root {
    --bg: #0a0a0f; --bg2: #12121a; --bg3: #1a1a26;
    --text: #c8c8d4; --text-dim: #5a5a6e;
    --cyan: #4ecdc4; --green: #a8e6cf; --red: #ff6b6b;
    --yellow: #ffd93d; --magenta: #c084fc; --blue: #60a5fa;
    --border: #2a2a3a;
  }
  body { background: var(--bg); color: var(--text); font-family: 'JetBrains Mono', monospace; height: 100vh; overflow: hidden; }
  .container { display: grid; grid-template-columns: 1fr 1fr; grid-template-rows: auto 1fr; height: 100vh; }
  .header {
    grid-column: 1 / -1; padding: 16px 24px; border-bottom: 1px solid var(--border);
    display: flex; align-items: center; justify-content: space-between; background: var(--bg2);
  }
  .header h1 { font-family: 'Space Grotesk', sans-serif; font-size: 18px; font-weight: 600; color: var(--cyan); letter-spacing: 2px; text-transform: uppercase; }
  .header .status { font-size: 11px; color: var(--text-dim); display: flex; gap: 16px; align-items: center; }
  .header .dot { width: 6px; height: 6px; border-radius: 50%; background: var(--green); display: inline-block; animation: pulse 2s ease-in-out infinite; }
  @keyframes pulse { 0%,100% { opacity:1; } 50% { opacity:0.3; } }

  .thoughts-panel { border-right: 1px solid var(--border); display: flex; flex-direction: column; overflow: hidden; }
  .panel-header { padding: 12px 20px; font-size: 11px; text-transform: uppercase; letter-spacing: 2px; color: var(--text-dim); border-bottom: 1px solid var(--border); background: var(--bg2); font-weight: 500; }
  .thoughts-stream { flex: 1; overflow-y: auto; padding: 12px 20px; display: flex; flex-direction: column; gap: 2px; font-size: 12px; line-height: 1.5; }
  .thoughts-stream::-webkit-scrollbar { width: 4px; }
  .thoughts-stream::-webkit-scrollbar-track { background: transparent; }
  .thoughts-stream::-webkit-scrollbar-thumb { background: var(--border); border-radius: 2px; }

  .thought-line { display: flex; gap: 8px; align-items: flex-start; animation: fadeIn 0.3s ease; padding: 2px 0; }
  @keyframes fadeIn { from { opacity:0; transform:translateY(4px); } to { opacity:1; transform:translateY(0); } }
  .thought-time { color: var(--text-dim); font-size: 10px; flex-shrink: 0; margin-top: 2px; }
  .thought-indicator { flex-shrink: 0; margin-top: 3px; font-size: 8px; }
  .thought-text { color: var(--cyan); word-break: break-all; opacity: 0.8; }

  .chat-panel { display: flex; flex-direction: column; overflow: hidden; }
  .chat-messages { flex: 1; overflow-y: auto; padding: 16px 20px; display: flex; flex-direction: column; gap: 12px; }
  .chat-messages::-webkit-scrollbar { width: 4px; }
  .chat-messages::-webkit-scrollbar-track { background: transparent; }
  .chat-messages::-webkit-scrollbar-thumb { background: var(--border); border-radius: 2px; }

  .message { display: flex; flex-direction: column; gap: 4px; animation: fadeIn 0.3s ease; }
  .message .sender { font-size: 10px; text-transform: uppercase; letter-spacing: 1px; font-weight: 500; }
  .message.human .sender { color: var(--green); }
  .message.mind .sender { color: var(--magenta); }
  .message.system .sender { color: var(--yellow); }
  .message .content { font-size: 13px; line-height: 1.6; padding: 8px 12px; border-radius: 6px; max-width: 95%; }
  .message.human .content { background: rgba(168,230,207,0.08); border: 1px solid rgba(168,230,207,0.15); }
  .message.mind .content { background: rgba(192,132,252,0.08); border: 1px solid rgba(192,132,252,0.15); word-break: break-all; }
  .message.system .content { background: rgba(255,217,61,0.06); border: 1px solid rgba(255,217,61,0.1); font-size: 11px; color: var(--text-dim); white-space: pre-wrap; }

  .chat-input-area { padding: 16px 20px; border-top: 1px solid var(--border); background: var(--bg2); display: flex; gap: 8px; }
  .chat-input-area input {
    flex: 1; background: var(--bg3); border: 1px solid var(--border); border-radius: 6px;
    padding: 10px 14px; color: var(--text); font-family: 'JetBrains Mono', monospace;
    font-size: 13px; outline: none; transition: border-color 0.2s;
  }
  .chat-input-area input:focus { border-color: var(--cyan); }
  .chat-input-area input::placeholder { color: var(--text-dim); }
  .chat-input-area button {
    background: var(--cyan); color: var(--bg); border: none; border-radius: 6px;
    padding: 10px 18px; font-family: 'JetBrains Mono', monospace; font-size: 12px;
    font-weight: 600; cursor: pointer; text-transform: uppercase; letter-spacing: 1px;
  }
  .chat-input-area button:hover { opacity: 0.85; }
  .controls { padding: 8px 20px; border-top: 1px solid var(--border); display: flex; gap: 6px; flex-wrap: wrap; }
  .controls button {
    background: var(--bg3); color: var(--text-dim); border: 1px solid var(--border);
    border-radius: 4px; padding: 4px 10px; font-family: 'JetBrains Mono', monospace;
    font-size: 10px; cursor: pointer; transition: all 0.2s;
  }
  .controls button:hover { color: var(--text); border-color: var(--cyan); }

  @media (max-width: 768px) {
    .container { grid-template-columns: 1fr; grid-template-rows: auto 1fr 1fr; }
    .thoughts-panel { border-right: none; border-bottom: 1px solid var(--border); }
  }
</style>
</head>
<body>
<div class="container">
  <div class="header">
    <h1>&#9673; Continuous Mind</h1>
    <div class="status">
      <span><span class="dot"></span> <span id="connStatus">connecting</span></span>
      <span id="thoughtCount">thoughts: 0</span>
      <span id="moodDisplay">mood: neutral</span>
      <span id="lossDisplay">loss: --</span>
    </div>
  </div>
  <div class="thoughts-panel">
    <div class="panel-header">Thought Stream</div>
    <div class="thoughts-stream" id="thoughtStream"></div>
  </div>
  <div class="chat-panel">
    <div class="panel-header">Chat</div>
    <div class="chat-messages" id="chatMessages">
      <div class="message system">
        <span class="sender">system</span>
        <div class="content">The mind is running. Type a message — it will learn from you and respond.
Its thoughts continue in the left panel while you chat.</div>
      </div>
    </div>
    <div class="controls">
      <button onclick="sendCommand('mood')">mood</button>
      <button onclick="sendCommand('stats')">stats</button>
      <button onclick="sendCommand('feed')">scan feed/</button>
      <button onclick="sendCommand('save')">save</button>
      <button onclick="sendCommand('clear')">clear mind</button>
      <button onclick="sendCommand('history')">history</button>
    </div>
    <div class="chat-input-area">
      <input type="text" id="chatInput" placeholder="Type a message..." autocomplete="off" />
      <button onclick="sendMessage()">Send</button>
    </div>
  </div>
</div>
<script>
const thoughtStream = document.getElementById('thoughtStream');
const chatMessages = document.getElementById('chatMessages');
const chatInput = document.getElementById('chatInput');
const thoughtCountEl = document.getElementById('thoughtCount');
const moodEl = document.getElementById('moodDisplay');
const lossEl = document.getElementById('lossDisplay');
const connStatusEl = document.getElementById('connStatus');

let maxThoughts = 200;

function connectSSE() {
  const es = new EventSource('/events');
  es.onopen = () => { connStatusEl.textContent = 'thinking'; };
  es.onmessage = (event) => {
    const data = JSON.parse(event.data);
    if (data.type === 'thought') {
      addThought(data);
      thoughtCountEl.textContent = 'thoughts: ' + data.count;
      moodEl.textContent = 'mood: ' + data.mood;
      lossEl.textContent = 'loss: ' + data.loss.toFixed(2);
    } else if (data.type === 'chat_response') {
      addChatMessage('mind', data.response);
      lossEl.textContent = 'loss: ' + data.loss.toFixed(2);
    } else if (data.type === 'system') {
      addSystemMessage(data.message);
    }
  };
  es.onerror = () => {
    connStatusEl.textContent = 'reconnecting';
    es.close();
    setTimeout(connectSSE, 2000);
  };
}

function addThought(data) {
  const line = document.createElement('div');
  line.className = 'thought-line';
  let color = 'var(--green)';
  if (data.loss > 4.0) color = 'var(--red)';
  else if (data.loss > 3.0) color = 'var(--yellow)';
  else if (data.loss > 2.0) color = 'var(--cyan)';
  line.innerHTML =
    '<span class="thought-time">' + data.time + '</span>' +
    '<span class="thought-indicator" style="color:' + color + '">&#9679;</span>' +
    '<span class="thought-text">' + escapeHtml(data.thought) + '</span>';
  thoughtStream.appendChild(line);
  while (thoughtStream.children.length > maxThoughts) thoughtStream.removeChild(thoughtStream.firstChild);
  thoughtStream.scrollTop = thoughtStream.scrollHeight;
}

function addChatMessage(role, text) {
  const msg = document.createElement('div');
  msg.className = 'message ' + role;
  msg.innerHTML = '<span class="sender">' + (role === 'human' ? 'you' : role) + '</span><div class="content">' + escapeHtml(text) + '</div>';
  chatMessages.appendChild(msg);
  chatMessages.scrollTop = chatMessages.scrollHeight;
}

function addSystemMessage(text) {
  const msg = document.createElement('div');
  msg.className = 'message system';
  msg.innerHTML = '<span class="sender">system</span><div class="content">' + escapeHtml(text) + '</div>';
  chatMessages.appendChild(msg);
  chatMessages.scrollTop = chatMessages.scrollHeight;
}

function sendMessage() {
  const text = chatInput.value.trim();
  if (!text) return;
  addChatMessage('human', text);
  fetch('/chat', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({message: text}) });
  chatInput.value = '';
}

function sendCommand(cmd) {
  fetch('/command', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({command: cmd}) });
}

function escapeHtml(t) { const d = document.createElement('div'); d.textContent = t; return d.innerHTML; }

chatInput.addEventListener('keydown', (e) => { if (e.key === 'Enter') sendMessage(); });
connectSSE();
</script>
</body>
</html>"""


# ═══════════════════════════════════════════════════════════════════════════════
# HTTP SERVER
# ═══════════════════════════════════════════════════════════════════════════════

mind = None
optimizer = None
mind_lock = threading.Lock()
event_bus = EventBus()
recent_thoughts = []


class MindHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        path = urlparse(self.path).path

        if path == '/' or path == '/index.html':
            body = HTML_PAGE.encode()
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        elif path == '/events':
            # Server-Sent Events stream
            self.send_response(200)
            self.send_header('Content-Type', 'text/event-stream')
            self.send_header('Cache-Control', 'no-cache')
            self.send_header('Connection', 'keep-alive')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()

            q = event_bus.subscribe()
            try:
                while True:
                    try:
                        msg = q.get(timeout=30)
                        self.wfile.write(f"data: {msg}\n\n".encode())
                        self.wfile.flush()
                    except queue.Empty:
                        # Send keepalive
                        self.wfile.write(": keepalive\n\n".encode())
                        self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, OSError):
                pass
            finally:
                event_bus.unsubscribe(q)

        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        global mind, optimizer, recent_thoughts
        path = urlparse(self.path).path
        content_length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(content_length).decode('utf-8', errors='ignore')

        try:
            data = json.loads(body) if body else {}
        except:
            data = {}

        if path == '/chat':
            message = data.get('message', '').strip()
            if message and mind:
                def process_chat():
                    global recent_thoughts
                    with mind_lock:
                        response, loss = mind.chat_respond(message, optimizer)
                        mind.thought_count += 1
                        mind.total_loss += loss
                    event_bus.broadcast('chat_response', {"response": response, "loss": loss})
                    recent_thoughts.append(message)
                    recent_thoughts = recent_thoughts[-5:]
                threading.Thread(target=process_chat, daemon=True).start()

            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(b'{"ok":true}')

        elif path == '/command':
            cmd = data.get('command', '').lower()
            threading.Thread(target=handle_command, args=(cmd,), daemon=True).start()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(b'{"ok":true}')

        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        pass  # Suppress logs

    def handle_one_request(self):
        try:
            super().handle_one_request()
        except (ConnectionResetError, BrokenPipeError, OSError):
            pass  # Suppress connection errors


def handle_command(cmd):
    global mind, optimizer, recent_thoughts

    if cmd == 'mood':
        with mind_lock:
            mood = mind.get_mood_description()
            raw = [f"{v:.2f}" for v in mind.mood[0].cpu().numpy()]
        event_bus.broadcast('system', {"message": f"Mood: {mood}\nRaw: [{', '.join(raw)}]"})

    elif cmd == 'stats':
        with mind_lock:
            stats = mind.get_stats()
        msg = (f"Thoughts: {stats['thought_count']}\n"
               f"Avg loss: {stats['avg_loss']}\n"
               f"Parameters: {stats['parameters']:,}\n"
               f"Files learned: {stats['files_learned']}\n"
               f"Chat messages: {stats['chat_messages']}\n"
               f"Memory norm: {stats['memory_norm']}")
        event_bus.broadcast('system', {"message": msg})

    elif cmd == 'feed':
        with mind_lock:
            learned = scan_feed_folder(mind, optimizer)
            save_state()
        if learned:
            for item in learned:
                if item['loss'] >= 0:
                    event_bus.broadcast('system', {"message": f"Learned from {item['file']} ({item['size']} chars, loss: {item['loss']:.2f})"})
        else:
            event_bus.broadcast('system', {"message": "No new files in feed/ folder."})

    elif cmd == 'save':
        with mind_lock:
            save_state()
        event_bus.broadcast('system', {"message": "State saved."})

    elif cmd == 'clear':
        with mind_lock:
            mind = ContinuousMind().to(DEVICE)
            optimizer = torch.optim.Adam(mind.parameters(), lr=LEARN_RATE)
            recent_thoughts = []
            if STATE_FILE.exists():
                STATE_FILE.unlink()
        event_bus.broadcast('system', {"message": "Mind wiped. Starting fresh."})

    elif cmd == 'history':
        with mind_lock:
            history = list(mind.chat_history[-10:])
        if history:
            lines = [f"{'you' if r == 'human' else 'mind'}: {m}" for r, m in history]
            event_bus.broadcast('system', {"message": "Chat history:\n" + "\n".join(lines)})
        else:
            event_bus.broadcast('system', {"message": "No chat history yet."})


def save_state():
    torch.save({
        'model_state': mind.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'thought_count': mind.thought_count,
        'total_loss': mind.total_loss,
        'files_learned': list(mind.files_learned),
        'chat_history': mind.chat_history,
    }, STATE_FILE)


# ═══════════════════════════════════════════════════════════════════════════════
# THINKING LOOP
# ═══════════════════════════════════════════════════════════════════════════════

def thinking_loop():
    global recent_thoughts
    log = open(LOG_FILE, "a")
    log.write(f"\n--- Session {datetime.now().isoformat()} ---\n")

    while True:
        # Periodic feed scan
        if mind.thought_count > 0 and mind.thought_count % FEED_SCAN_INTERVAL == 0:
            with mind_lock:
                learned = scan_feed_folder(mind, optimizer)
            for item in learned:
                if item['loss'] >= 0:
                    event_bus.broadcast('system', {"message": f"Auto-learned from {item['file']} ({item['size']} chars)"})

        seed = None
        if recent_thoughts and np.random.random() < 0.4:
            seed = np.random.choice(recent_thoughts)[:8]

        with mind_lock:
            thought = mind.generate(seed=seed)
            loss = mind.learn_from_text(thought, optimizer)
            mind.thought_count += 1
            mind.total_loss += loss
            mood = mind.get_mood_description()
            count = mind.thought_count

        timestamp = datetime.now().strftime("%H:%M:%S")
        event_bus.broadcast('thought', {
            "thought": thought, "loss": loss, "mood": mood,
            "count": count, "time": timestamp,
        })

        log.write(f"[{timestamp}] #{count} (loss: {loss:.2f}) {thought}\n")
        log.flush()

        recent_thoughts.append(thought[-20:])
        recent_thoughts = recent_thoughts[-5:]

        if count % 50 == 0:
            with mind_lock:
                save_state()

        time.sleep(THINK_INTERVAL)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    global mind, optimizer

    FEED_DIR.mkdir(exist_ok=True)

    mind = ContinuousMind().to(DEVICE)
    optimizer = torch.optim.Adam(mind.parameters(), lr=LEARN_RATE)

    if STATE_FILE.exists():
        try:
            checkpoint = torch.load(STATE_FILE, map_location=DEVICE, weights_only=False)
            mind.load_state_dict(checkpoint['model_state'])
            optimizer.load_state_dict(checkpoint['optimizer_state'])
            mind.thought_count = checkpoint.get('thought_count', 0)
            mind.total_loss = checkpoint.get('total_loss', 0.0)
            mind.files_learned = set(checkpoint.get('files_learned', []))
            mind.chat_history = [tuple(x) for x in checkpoint.get('chat_history', [])]
            print(f"  Restored mind ({mind.thought_count} thoughts, {len(mind.files_learned)} files)")
        except Exception as e:
            print(f"  Could not restore: {e}")
    else:
        print("  New mind initialized.")

    # Start thinking thread (feed scan happens in background inside this loop)
    think_thread = threading.Thread(target=thinking_loop, daemon=True)
    think_thread.start()

    # Background feed scan (non-blocking)
    def initial_feed_scan():
        time.sleep(1)  # Let server start first
        with mind_lock:
            learned = scan_feed_folder(mind, optimizer)
        for item in learned:
            if item['loss'] >= 0:
                event_bus.broadcast('system', {"message": f"Learned from {item['file']} ({item['size']} chars, loss: {item['loss']:.2f})"})
                print(f"  Learned from {item['file']} ({item['size']} chars)")

    feed_thread = threading.Thread(target=initial_feed_scan, daemon=True)
    feed_thread.start()

    # Handle Ctrl+C
    def signal_handler(sig, frame):
        print("\n  Saving mind state...")
        with mind_lock:
            save_state()
        print(f"  Mind saved ({mind.thought_count} thoughts). Goodbye.")
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    print(f"""
╔══════════════════════════════════════════════════════════════╗
║               ◉  CONTINUOUS MIND  ◉                         ║
║          A self-learning digital organism                    ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║   Open your browser:  http://localhost:{PORT}                  ║
║                                                              ║
║   The mind is thinking. Chat with it in the browser.         ║
║   Drop .txt files in ./feed/ to teach it.                    ║
║                                                              ║
║   Press Ctrl+C to save and quit.                             ║
╚══════════════════════════════════════════════════════════════╝
""")

    # Start HTTP server (blocking)
    class QuietHTTPServer(ThreadingMixIn, HTTPServer):
        daemon_threads = True
        def handle_error(self, request, client_address):
            pass  # Suppress all connection error tracebacks

    server = QuietHTTPServer(('', PORT), MindHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        signal_handler(None, None)


if __name__ == "__main__":
    main()
