/**
 * Chess Game Recording System - Web Operator Dashboard Client
 */

// SVG vector definitions for professional chess pieces
const PIECE_SVGS = {
  P: '<svg viewBox="0 0 45 45"><path d="m22.5 9c-2.21 0-4 1.79-4 4 0 .89.29 1.71.78 2.38C17.33 16.5 16 18.59 16 21c0 2.03.94 3.84 2.41 5.03-3 1.06-7.41 5.55-7.41 13.47h23c0-7.92-4.41-12.41-7.41-13.47 1.47-1.19 2.41-3 2.41-5.03 0-2.41-1.33-4.5-3.28-5.62.49-.67.78-1.49.78-2.38 0-2.21-1.79-4-4-4z" fill="#ffffff" stroke="#222222" stroke-width="1.5"/></svg>',
  N: '<svg viewBox="0 0 45 45"><path d="m22 10c10.5 1 16.5 8 16 29H15c0-9 10-6.5 8-21" fill="#ffffff" stroke="#222222" stroke-width="1.5"/><path d="m24 18c.38 2.91-5.55 7.37-8 9-3 2-2.82 4.34-5 4-1.042-.94 1.41-3.04 0-3-1 0 .19 1.23-1 2-1 0-4.003 1-4-4 0-2 6-12 6-12s1.89-1.9 2-3.5c-.73-.994-.5-2-.5-3 1-1 3 2.5 3 2.5z" fill="#ffffff" stroke="#222222" stroke-width="1.5"/><circle cx="9.5" cy="25.5" r="1" fill="#222222"/></svg>',
  B: '<svg viewBox="0 0 45 45"><g fill="#ffffff" stroke="#222222" stroke-width="1.5"><path d="m9 36c3.39-.97 10.11.43 13.5-2 3.39 2.43 10.11 1.03 13.5 2 0 0 1.65.54 3 2-.68.97-1.65.99-3 .5-3.39-.97-10.11.46-13.5-1-3.39 1.46-10.11.03-13.5 1-1.354.49-2.323.47-3-.5 1.354-1.94 3-2 3-2z"/><path d="m15 32c2.5 2.5 12.5 2.5 15 0 .5-1.5 0-2 0-2 0-2.5-2.5-4-2.5-4 5.5-1.5 6-11.5-5-15.5-11 4-10.5 14-5 15.5 0 0-2.5 1.5-2.5 4 0 0-.5.5 0 2z"/><path d="m25 8a2.5 2.5 0 1 1-5 0 2.5 2.5 0 1 1 5 0z"/><path d="m17.5 26h10M20 23v6m2.5-3.5h.01" fill="none"/></g></svg>',
  R: '<svg viewBox="0 0 45 45"><g fill="#ffffff" stroke="#222222" stroke-width="1.5"><path d="m9 39h27v-3H9v3zm3-3v-4h21v4H12zm2-4v-14h17v14H14zm-3-14v-4h23v4H11zm2-4V9h3v3h5V9h3v3h5V9h3v5H13z"/><path d="m14 29.5v-13h17v13H14z" fill="#ffffff"/></g></svg>',
  Q: '<svg viewBox="0 0 45 45"><g fill="#ffffff" stroke="#222222" stroke-width="1.5"><path d="m9 26c8.5-1.5 21-1.5 27 0l2-12-7 11-4-14-4.5 14-4.5-14-4 14-7-11 2 12z"/><path d="m9 26c0 2 1.5 2 2.5 4 1 1.5 1 1 .5 3.5-1.5 1-1.5 2.5-1.5 2.5-1.5 1.5.5 2.5.5 2.5 6.5 1 16.5 1 23 0 0 0 2-1 .5-2.5 0 0 0-1.5-1.5-2.5-.5-2.5-.5-2 0-3.5 1.5-2 2.5-2 2.5-4l-27 0z"/><circle cx="6" cy="12" r="2"/><circle cx="14" cy="9" r="2"/><circle cx="22.5" cy="8" r="2"/><circle cx="31" cy="9" r="2"/><circle cx="39" cy="12" r="2"/></g></svg>',
  K: '<svg viewBox="0 0 45 45"><g fill="#ffffff" stroke="#222222" stroke-width="1.5"><path d="m22.5 11.63v5.37m-2.68-2.68h5.36M22.5 25s4.5-7.5 3-10.5c0 0-1-2.5-3-2.5s-3 2.5-3 2.5c-1.5 3 3 10.5 3 10.5"/><path d="m11.5 37c5.5 3.5 15.5 3.5 21 0v-7s9-4.5 6-10.5c-4-1-6 2-6 2s-3-7-10-7-10 7-10 7-2-3-6-2c-3 6 6 10.5 6 10.5v7z"/><circle cx="22.5" cy="8" r="1.5"/></g></svg>',
  p: '<svg viewBox="0 0 45 45"><path d="m22.5 9c-2.21 0-4 1.79-4 4 0 .89.29 1.71.78 2.38C17.33 16.5 16 18.59 16 21c0 2.03.94 3.84 2.41 5.03-3 1.06-7.41 5.55-7.41 13.47h23c0-7.92-4.41-12.41-7.41-13.47 1.47-1.19 2.41-3 2.41-5.03 0-2.41-1.33-4.5-3.28-5.62.49-.67.78-1.49.78-2.38 0-2.21-1.79-4-4-4z" fill="#222222" stroke="#ffffff" stroke-width="1.2"/></svg>',
  n: '<svg viewBox="0 0 45 45"><path d="m22 10c10.5 1 16.5 8 16 29H15c0-9 10-6.5 8-21" fill="#222222" stroke="#ffffff" stroke-width="1.2"/><path d="m24 18c.38 2.91-5.55 7.37-8 9-3 2-2.82 4.34-5 4-1.042-.94 1.41-3.04 0-3-1 0 .19 1.23-1 2-1 0-4.003 1-4-4 0-2 6-12 6-12s1.89-1.9 2-3.5c-.73-.994-.5-2-.5-3 1-1 3 2.5 3 2.5z" fill="#222222" stroke="#ffffff" stroke-width="1.2"/><circle cx="9.5" cy="25.5" r="1" fill="#ffffff"/></svg>',
  b: '<svg viewBox="0 0 45 45"><g fill="#222222" stroke="#ffffff" stroke-width="1.2"><path d="m9 36c3.39-.97 10.11.43 13.5-2 3.39 2.43 10.11 1.03 13.5 2 0 0 1.65.54 3 2-.68.97-1.65.99-3 .5-3.39-.97-10.11.46-13.5-1-3.39 1.46-10.11.03-13.5 1-1.354.49-2.323.47-3-.5 1.354-1.94 3-2 3-2z"/><path d="m15 32c2.5 2.5 12.5 2.5 15 0 .5-1.5 0-2 0-2 0-2.5-2.5-4-2.5-4 5.5-1.5 6-11.5-5-15.5-11 4-10.5 14-5 15.5 0 0-2.5 1.5-2.5 4 0 0-.5.5 0 2z"/><path d="m25 8a2.5 2.5 0 1 1-5 0 2.5 2.5 0 1 1 5 0z"/><path d="m17.5 26h10M20 23v6m2.5-3.5h.01" stroke="#ffffff" fill="none"/></g></svg>',
  r: '<svg viewBox="0 0 45 45"><g fill="#222222" stroke="#ffffff" stroke-width="1.2"><path d="m9 39h27v-3H9v3zm3-3v-4h21v4H12zm2-4v-14h17v14H14zm-3-14v-4h23v4H11zm2-4V9h3v3h5V9h3v3h5V9h3v5H13z"/><path d="m14 29.5v-13h17v13H14z" fill="#222222"/></g></svg>',
  q: '<svg viewBox="0 0 45 45"><g fill="#222222" stroke="#ffffff" stroke-width="1.2"><path d="m9 26c8.5-1.5 21-1.5 27 0l2-12-7 11-4-14-4.5 14-4.5-14-4 14-7-11 2 12z"/><path d="m9 26c0 2 1.5 2 2.5 4 1 1.5 1 1 .5 3.5-1.5 1-1.5 2.5-1.5 2.5-1.5 1.5.5 2.5.5 2.5 6.5 1 16.5 1 23 0 0 0 2-1 .5-2.5 0 0 0-1.5-1.5-2.5-.5-2.5-.5-2 0-3.5 1.5-2 2.5-2 2.5-4l-27 0z"/><circle cx="6" cy="12" r="2"/><circle cx="14" cy="9" r="2"/><circle cx="22.5" cy="8" r="2"/><circle cx="31" cy="9" r="2"/><circle cx="39" cy="12" r="2"/></g></svg>',
  k: '<svg viewBox="0 0 45 45"><g fill="#222222" stroke="#ffffff" stroke-width="1.2"><path d="m22.5 11.63v5.37m-2.68-2.68h5.36M22.5 25s4.5-7.5 3-10.5c0 0-1-2.5-3-2.5s-3 2.5-3 2.5c-1.5 3 3 10.5 3 10.5"/><path d="m11.5 37c5.5 3.5 15.5 3.5 21 0v-7s9-4.5 6-10.5c-4-1-6 2-6 2s-3-7-10-7-10 7-10 7-2-3-6-2c-3 6 6 10.5 6 10.5v7z"/><circle cx="22.5" cy="8" r="1.5"/></g></svg>'
};

class ChessWebDashboard {
  constructor() {
    this.ws = null;
    this.reconnectTimer = null;
    this.lastState = null;
    this.isPaused = false;

    // DOM Elements
    this.badgeConn = document.getElementById("badgeConnection");
    this.badgeFsm = document.getElementById("badgeFsm");
    this.badgeFps = document.getElementById("fpsCounter");
    this.btnPause = document.getElementById("btnPause");
    this.btnRollback = document.getElementById("btnRollback");
    this.btnRecalibrate = document.getElementById("btnRecalibrate");
    this.btnDownloadPgn = document.getElementById("btnDownloadPgn");

    this.videoStream = document.getElementById("videoStream");
    this.cornersCanvas = document.getElementById("cornersCanvas");
    this.corners = null;
    this.draggedCornerIdx = -1;
    this.isDragging = false;

    this.boardEl = document.getElementById("chessboard");
    this.evalFillEl = document.getElementById("evalFill");
    this.evalLabelEl = document.getElementById("evalLabel");
    this.movesBodyEl = document.getElementById("movesTableBody");
    this.notationScrollEl = document.getElementById("notationScroll");

    this.metaPlyEl = document.getElementById("metaPly");
    this.metaBoardEl = document.getElementById("metaBoard");
    this.metaHandEl = document.getElementById("metaHand");
    this.metaMotionEl = document.getElementById("metaMotion");

    this.modalOverlay = document.getElementById("uncertainModal");
    this.candidateListEl = document.getElementById("candidateList");
    this.manualUciInput = document.getElementById("manualUciInput");
    this.btnManualCommit = document.getElementById("btnManualCommit");
    this.btnAutoAcceptTop = document.getElementById("btnAutoAcceptTop");

    this.initBoardGrid();
    this.bindEvents();
    this.connectWebSocket();
  }

  // Pre-generate 64 squares for 8x8 grid (rank 8 down to 1, file a to h)
  initBoardGrid() {
    this.boardEl.innerHTML = "";
    const files = ["a", "b", "c", "d", "e", "f", "g", "h"];

    for (let r = 8; r >= 1; r--) {
      for (let f = 0; f < 8; f++) {
        const file = files[f];
        const sq = file + r;
        const isLight = (f + r) % 2 !== 0;

        const sqEl = document.createElement("div");
        sqEl.className = `square ${isLight ? "light" : "dark"}`;
        sqEl.id = `sq_${sq}`;
        sqEl.dataset.square = sq;

        // Coordinate labels
        if (f === 0) {
          const rankLabel = document.createElement("span");
          rankLabel.className = "coord-label coord-rank";
          rankLabel.textContent = r;
          sqEl.appendChild(rankLabel);
        }
        if (r === 1) {
          const fileLabel = document.createElement("span");
          fileLabel.className = "coord-label coord-file";
          fileLabel.textContent = file;
          sqEl.appendChild(fileLabel);
        }

        const pieceContainer = document.createElement("div");
        pieceContainer.className = "piece-symbol";
        sqEl.appendChild(pieceContainer);

        this.boardEl.appendChild(sqEl);
      }
    }
  }

  bindEvents() {
    this.btnPause.addEventListener("click", () => this.postAction("/api/action/pause"));
    this.btnRollback.addEventListener("click", () => this.postAction("/api/action/rollback"));
    this.btnRecalibrate.addEventListener("click", () => {
      this.corners = null;
      if (this.cornersCanvas) {
        const ctx = this.cornersCanvas.getContext("2d");
        ctx.clearRect(0, 0, this.cornersCanvas.width, this.cornersCanvas.height);
      }
      this.postAction("/api/action/recalibrate");
    });
    this.btnDownloadPgn.addEventListener("click", () => {
      window.open("/api/pgn?download=true", "_blank");
    });

    if (this.cornersCanvas) {
      this.cornersCanvas.addEventListener("mousedown", (e) => this.handleCornerMouseDown(e));
      window.addEventListener("mousemove", (e) => this.handleCornerMouseMove(e));
      window.addEventListener("mouseup", (e) => this.handleCornerMouseUp(e));
    }
    window.addEventListener("resize", () => this.renderCornerHandles());

    this.btnAutoAcceptTop.addEventListener("click", () => {
      this.resolveUncertainty({ auto_accept_top: true });
    });

    this.btnManualCommit.addEventListener("click", () => {
      const uci = this.manualUciInput.value.trim().toLowerCase();
      if (uci.length >= 4) {
        this.resolveUncertainty({ move_uci: uci });
      }
    });

    this.manualUciInput.addEventListener("keydown", (e) => {
      if (e.key === "Enter") {
        this.btnManualCommit.click();
      }
    });
  }

  connectWebSocket() {
    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const wsUrl = `${protocol}//${window.location.host}/ws/live`;

    if (this.ws) {
      try { this.ws.close(); } catch (e) {}
    }

    this.ws = new WebSocket(wsUrl);

    this.ws.onopen = () => {
      this.badgeConn.className = "badge badge-connected";
      this.badgeConn.innerHTML = '<span class="badge-dot"></span> LIVE';
      if (this.reconnectTimer) {
        clearTimeout(this.reconnectTimer);
        this.reconnectTimer = null;
      }
    };

    this.ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        this.handleMessage(data);
      } catch (err) {
        console.error("Error parsing WebSocket message:", err);
      }
    };

    this.ws.onclose = () => {
      this.badgeConn.className = "badge badge-disconnected";
      this.badgeConn.innerHTML = '<span class="badge-dot"></span> OFFLINE';
      this.scheduleReconnect();
    };

    this.ws.onerror = (err) => {
      console.warn("WebSocket error:", err);
      this.ws.close();
    };
  }

  scheduleReconnect() {
    if (!this.reconnectTimer) {
      this.reconnectTimer = setTimeout(() => {
        this.reconnectTimer = null;
        this.connectWebSocket();
      }, 2000);
    }
  }

  handleMessage(msg) {
    if (msg.type === "gameState") {
      this.updateState(msg);
    }
  }

  updateState(state) {
    this.lastState = state;

    // Header metrics
    this.badgeFsm.textContent = `FSM: ${state.fsm_state}`;
    this.badgeFps.textContent = `${state.fps} FPS`;

    // Pause button text
    this.isPaused = state.is_paused;
    this.btnPause.textContent = this.isPaused ? "▶ Resume" : "⏸ Pause";
    if (this.isPaused) {
      this.btnPause.classList.add("btn-warning");
    } else {
      this.btnPause.classList.remove("btn-warning");
    }

    // Telemetry bar
    this.metaPlyEl.textContent = `${state.ply} (${Math.floor((state.ply + 1) / 2)})`;
    this.metaBoardEl.textContent = state.board_detected ? "LOCKED" : "SEARCHING...";
    this.metaBoardEl.style.color = state.board_detected ? "var(--color-success)" : "var(--color-warning)";

    // Update corner handles
    if (state.board_corners) {
      this.onBoardCornersReceived(state.board_corners);
    } else if (!state.board_detected && !this.isDragging) {
      this.corners = null;
      if (this.cornersCanvas) {
        const ctx = this.cornersCanvas.getContext("2d");
        ctx.clearRect(0, 0, this.cornersCanvas.width, this.cornersCanvas.height);
      }
    }

    // Update Chessboard FEN
    this.renderFen(state.fen, state.last_move, state.is_check, state.turn);

    // Stockfish Gauge
    this.updateEvalGauge(state.engine_eval);

    // Notation table
    this.updateMoveTable(state.move_history, state.engine_eval);

    // UNCERTAIN Modal
    if (state.active_uncertainty) {
      this.showUncertainModal(state.uncertain_candidates);
    } else {
      this.hideUncertainModal();
    }
  }

  renderFen(fen, lastMove, isCheck, turn) {
    // Clear highlights
    const allSquares = this.boardEl.querySelectorAll(".square");
    allSquares.forEach((sq) => {
      sq.classList.remove("move-source", "move-target", "in-check");
      const sym = sq.querySelector(".piece-symbol");
      if (sym) sym.innerHTML = "";
    });

    if (!fen) return;
    const parts = fen.split(" ");
    const rows = parts[0].split("/");
    let kingSquare = null;

    for (let rIdx = 0; rIdx < 8; rIdx++) {
      const rank = 8 - rIdx;
      const rowStr = rows[rIdx];
      let fileIdx = 0;

      for (let i = 0; i < rowStr.length; i++) {
        const char = rowStr[i];
        if (char >= "1" && char <= "8") {
          fileIdx += parseInt(char, 10);
        } else {
          const file = ["a", "b", "c", "d", "e", "f", "g", "h"][fileIdx];
          const sqId = `sq_${file}${rank}`;
          const sqEl = document.getElementById(sqId);
          if (sqEl) {
            const sym = sqEl.querySelector(".piece-symbol");
            if (sym && PIECE_SVGS[char]) {
              sym.innerHTML = PIECE_SVGS[char];
            }
            if (isCheck) {
              if ((turn === "white" && char === "K") || (turn === "black" && char === "k")) {
                kingSquare = sqEl;
              }
            }
          }
          fileIdx++;
        }
      }
    }

    // Highlight last move
    if (lastMove && lastMove.uci && lastMove.uci.length >= 4) {
      const fromSq = lastMove.uci.slice(0, 2);
      const toSq = lastMove.uci.slice(2, 4);
      const fromEl = document.getElementById(`sq_${fromSq}`);
      const toEl = document.getElementById(`sq_${toSq}`);
      if (fromEl) fromEl.classList.add("move-source");
      if (toEl) toEl.classList.add("move-target");
    }

    // Highlight king in check
    if (kingSquare) {
      kingSquare.classList.add("in-check");
    }
  }

  updateEvalGauge(evalData) {
    if (!evalData) {
      this.evalFillEl.style.height = "50%";
      this.evalLabelEl.textContent = "0.0";
      return;
    }

    if (evalData.mate_in !== null && evalData.mate_in !== undefined) {
      const m = evalData.mate_in;
      const whiteWins = m > 0;
      this.evalFillEl.style.height = whiteWins ? "100%" : "0%";
      this.evalLabelEl.textContent = (m > 0 ? "+M" : "-M") + Math.abs(m);
    } else if (evalData.score_cp !== null && evalData.score_cp !== undefined) {
      const cp = evalData.score_cp;
      // Convert centipawns to win percentage (capped between 5% and 95%)
      const pct = Math.max(5, Math.min(95, 50 + cp / 25));
      this.evalFillEl.style.height = `${pct}%`;
      const sign = cp >= 0 ? "+" : "";
      this.evalLabelEl.textContent = `${sign}${(cp / 100).toFixed(1)}`;
    }
  }

  updateMoveTable(history, latestEval) {
    if (!history) return;

    this.movesBodyEl.innerHTML = "";
    const totalMoves = Math.ceil(history.length / 2);

    for (let m = 0; m < totalMoves; m++) {
      const moveNum = m + 1;
      const whitePly = history[m * 2];
      const blackPly = history[m * 2 + 1];

      const tr = document.createElement("tr");
      if (m === totalMoves - 1) {
        tr.className = "latest-move";
      }

      const tdNum = document.createElement("td");
      tdNum.className = "move-num";
      tdNum.textContent = `${moveNum}.`;

      const tdWhite = document.createElement("td");
      tdWhite.className = "move-white";
      tdWhite.textContent = whitePly ? whitePly.san : "";

      const tdBlack = document.createElement("td");
      tdBlack.className = "move-black";
      tdBlack.textContent = blackPly ? blackPly.san : "";

      const tdEval = document.createElement("td");
      tdEval.className = "move-eval";
      if (m === totalMoves - 1 && latestEval && latestEval.quality) {
        const qClass = `q-${latestEval.quality.toLowerCase()}`;
        tdEval.innerHTML = `<span class="q-badge ${qClass}">${latestEval.quality}</span>`;
      }

      tr.appendChild(tdNum);
      tr.appendChild(tdWhite);
      tr.appendChild(tdBlack);
      tr.appendChild(tdEval);
      this.movesBodyEl.appendChild(tr);
    }

    // Auto-scroll to bottom
    this.notationScrollEl.scrollTop = this.notationScrollEl.scrollHeight;
  }

  showUncertainModal(candidates) {
    this.candidateListEl.innerHTML = "";

    if (candidates && candidates.length > 0) {
      candidates.forEach((cand, idx) => {
        const btn = document.createElement("button");
        btn.className = "candidate-btn";
        btn.innerHTML = `
          <span><strong>#${cand.rank}</strong> ${cand.san} (${cand.uci})</span>
          <span style="color: var(--color-warning);">Score: ${cand.score}</span>
        `;
        btn.addEventListener("click", () => {
          this.resolveUncertainty({ candidate_index: idx });
        });
        this.candidateListEl.appendChild(btn);
      });
      this.btnAutoAcceptTop.style.display = "inline-flex";
    } else {
      this.btnAutoAcceptTop.style.display = "none";
      const emptyMsg = document.createElement("div");
      emptyMsg.textContent = "No automatic candidates available. Please enter UCI move below:";
      this.candidateListEl.appendChild(emptyMsg);
    }

    this.modalOverlay.classList.add("open");
    this.manualUciInput.focus();
  }

  hideUncertainModal() {
    this.modalOverlay.classList.remove("open");
    this.manualUciInput.value = "";
  }

  onBoardCornersReceived(board_corners) {
    if (!this.isDragging && board_corners && board_corners.length === 4) {
      this.corners = board_corners.map(p => [Number(p[0]), Number(p[1])]);
      this.renderCornerHandles();
    }
  }

  getCanvasScale() {
    if (!this.videoStream || !this.cornersCanvas) return { scaleX: 1, scaleY: 1, natW: 1, natH: 1, rect: null };
    const rect = this.videoStream.getBoundingClientRect();
    if (rect.width === 0 || rect.height === 0) return { scaleX: 1, scaleY: 1, natW: 1, natH: 1, rect };
    if (this.cornersCanvas.width !== Math.round(rect.width) || this.cornersCanvas.height !== Math.round(rect.height)) {
      this.cornersCanvas.width = Math.round(rect.width);
      this.cornersCanvas.height = Math.round(rect.height);
    }
    const natW = this.videoStream.naturalWidth || rect.width;
    const natH = this.videoStream.naturalHeight || rect.height;
    return {
      scaleX: rect.width / Math.max(1, natW),
      scaleY: rect.height / Math.max(1, natH),
      natW,
      natH,
      rect
    };
  }

  renderCornerHandles() {
    if (!this.cornersCanvas || !this.videoStream) return;
    const { scaleX, scaleY, rect } = this.getCanvasScale();
    if (!rect) return;
    const ctx = this.cornersCanvas.getContext("2d");
    ctx.clearRect(0, 0, this.cornersCanvas.width, this.cornersCanvas.height);

    if (!this.corners || this.corners.length !== 4) return;

    const pts = this.corners.map(c => [c[0] * scaleX, c[1] * scaleY]);

    // 1. Draw boundary quadrilateral
    ctx.beginPath();
    ctx.moveTo(pts[0][0], pts[0][1]);
    ctx.lineTo(pts[1][0], pts[1][1]);
    ctx.lineTo(pts[2][0], pts[2][1]);
    ctx.lineTo(pts[3][0], pts[3][1]);
    ctx.closePath();
    ctx.strokeStyle = "rgba(234, 179, 8, 0.9)";
    ctx.lineWidth = 2;
    ctx.setLineDash([6, 4]);
    ctx.stroke();
    ctx.setLineDash([]);

    // 2. Draw 4 draggable corner handles
    const labels = ["a8", "h8", "h1", "a1"];
    pts.forEach((p, i) => {
      const isDrag = (i === this.draggedCornerIdx);
      ctx.beginPath();
      ctx.arc(p[0], p[1], isDrag ? 10 : 8, 0, Math.PI * 2);
      ctx.fillStyle = isDrag ? "#f59e0b" : "#06b6d4";
      ctx.fill();
      ctx.strokeStyle = "#ffffff";
      ctx.lineWidth = 2;
      ctx.stroke();

      ctx.fillStyle = "#ffffff";
      ctx.font = "bold 11px Inter, sans-serif";
      ctx.fillText(labels[i] || (i + 1), p[0] + 10, p[1] - 4);
    });
  }

  handleCornerMouseDown(e) {
    if (!this.corners || this.corners.length !== 4) return;
    const { scaleX, scaleY, rect } = this.getCanvasScale();
    if (!rect) return;
    const mouseX = e.clientX - rect.left;
    const mouseY = e.clientY - rect.top;

    for (let i = 0; i < 4; i++) {
      const cx = this.corners[i][0] * scaleX;
      const cy = this.corners[i][1] * scaleY;
      if (Math.hypot(mouseX - cx, mouseY - cy) <= 18) {
        this.draggedCornerIdx = i;
        this.isDragging = true;
        this.cornersCanvas.style.cursor = "grabbing";
        e.preventDefault();
        break;
      }
    }
  }

  handleCornerMouseMove(e) {
    if (!this.corners || this.corners.length !== 4) return;
    const { scaleX, scaleY, natW, natH, rect } = this.getCanvasScale();
    if (!rect) return;
    const mouseX = e.clientX - rect.left;
    const mouseY = e.clientY - rect.top;

    if (this.isDragging && this.draggedCornerIdx !== -1) {
      const clampedNativeX = Math.max(0, Math.min(natW, mouseX / scaleX));
      const clampedNativeY = Math.max(0, Math.min(natH, mouseY / scaleY));
      this.corners[this.draggedCornerIdx] = [clampedNativeX, clampedNativeY];
      this.renderCornerHandles();
    } else {
      let isOver = false;
      for (let i = 0; i < 4; i++) {
        const cx = this.corners[i][0] * scaleX;
        const cy = this.corners[i][1] * scaleY;
        if (Math.hypot(mouseX - cx, mouseY - cy) <= 18) {
          isOver = true;
          break;
        }
      }
      this.cornersCanvas.style.cursor = isOver ? "grab" : "default";
    }
  }

  handleCornerMouseUp(e) {
    if (this.isDragging && this.draggedCornerIdx !== -1) {
      this.isDragging = false;
      this.draggedCornerIdx = -1;
      this.cornersCanvas.style.cursor = "default";
      if (this.corners && this.corners.length === 4) {
        this.postAction("/api/action/set_corners", { corners: this.corners });
      }
      this.renderCornerHandles();
    }
  }
  }

  async postAction(url, body = {}) {
    try {
      const res = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      if (!res.ok) {
        console.error(`Action ${url} failed:`, await res.text());
      }
    } catch (err) {
      console.error(`Error dispatching action to ${url}:`, err);
    }
  }

  async resolveUncertainty(payload) {
    await this.postAction("/api/action/resolve", payload);
    this.hideUncertainModal();
  }
}

// Instantiate dashboard when DOM is ready
document.addEventListener("DOMContentLoaded", () => {
  window.dashboard = new ChessWebDashboard();
});
