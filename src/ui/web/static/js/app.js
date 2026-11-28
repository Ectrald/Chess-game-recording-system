/**
 * Chess Game Recording System - Web Operator Dashboard Client
 */

// SVG vector definitions for chess pieces
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

const INITIAL_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

class ChessWebDashboard {
  constructor() {
    this.ws = null;
    this.reconnectTimer = null;
    this.lastState = null;
    this.isPaused = false;
    this.isRecordingVideo = false;
    this.viewPly = null; // null = live / latest position; integer 0..N = reviewing move

    // DOM Elements
    this.sessionBadge = document.getElementById("sessionBadge");
    this.badgeConn = document.getElementById("badgeConnection");
    this.badgeFsm = document.getElementById("badgeFsm");
    this.badgeFps = document.getElementById("fpsCounter");
    this.badgeRecStatus = document.getElementById("badgeRecStatus");
    this.recTimerLabel = document.getElementById("recTimerLabel");
    this.pauseBanner = document.getElementById("pauseBanner");
    this.statusToast = document.getElementById("statusToast");

    this.videoDeviceSelect = document.getElementById("videoDeviceSelect");
    this.btnRefreshDevices = document.getElementById("btnRefreshDevices");
    this.videoSourceLabel = document.getElementById("videoSourceLabel");

    this.btnPause = document.getElementById("btnPause");
    this.btnCalibrateNow = document.getElementById("btnCalibrateNow");
    this.btnRecalibrate = document.getElementById("btnRecalibrate");
    this.btnRollback = document.getElementById("btnRollback");
    this.btnResetSession = document.getElementById("btnResetSession");
    this.btnNewSession = document.getElementById("btnNewSession");
    this.btnSessionsModal = document.getElementById("btnSessionsModal");
    this.btnRecordVideo = document.getElementById("btnRecordVideo");
    this.btnTranscribeModal = document.getElementById("btnTranscribeModal");
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

    this.btnReplayStart = document.getElementById("btnReplayStart");
    this.btnReplayPrev = document.getElementById("btnReplayPrev");
    this.replayIndicator = document.getElementById("replayIndicator");
    this.btnReplayNext = document.getElementById("btnReplayNext");
    this.btnReplayEnd = document.getElementById("btnReplayEnd");

    this.metaPlyEl = document.getElementById("metaPly");
    this.metaBoardEl = document.getElementById("metaBoard");
    this.metaWhiteEl = document.getElementById("metaWhitePlayer");
    this.metaBlackEl = document.getElementById("metaBlackPlayer");

    // Modals
    this.modalOverlay = document.getElementById("uncertainModal");
    this.modalTitleEl = document.getElementById("uncertainModalTitle");
    this.modalBodyEl = document.getElementById("uncertainModalBody");
    this.candidateListEl = document.getElementById("candidateList");
    this.manualUciInput = document.getElementById("manualUciInput");
    this.btnManualCommit = document.getElementById("btnManualCommit");
    this.btnAutoAcceptTop = document.getElementById("btnAutoAcceptTop");
    this.btnToggleUncertain = document.getElementById("btnToggleUncertain");
    this.isUncertainMinimized = false;
    this.lastUncertainSig = null;

    this.sessionsModal = document.getElementById("sessionsModal");
    this.btnCloseSessionsModal = document.getElementById("btnCloseSessionsModal");
    this.newWhiteInput = document.getElementById("newWhiteInput");
    this.newBlackInput = document.getElementById("newBlackInput");
    this.btnCreateNamedSession = document.getElementById("btnCreateNamedSession");
    this.sessionsListEl = document.getElementById("sessionsList");

    this.transcribeModal = document.getElementById("transcribeModal");
    this.btnCloseTranscribeModal = document.getElementById("btnCloseTranscribeModal");
    this.videoFileInput = document.getElementById("videoFileInput");
    this.transcribeWhiteInput = document.getElementById("transcribeWhiteInput");
    this.transcribeBlackInput = document.getElementById("transcribeBlackInput");
    this.btnUploadAndTranscribe = document.getElementById("btnUploadAndTranscribe");
    this.transcribeProgressBox = document.getElementById("transcribeProgressBox");
    this.transcribeStatusText = document.getElementById("transcribeStatusText");
    this.transcribeProgressFill = document.getElementById("transcribeProgressFill");
    this.transcribeMovesPreview = document.getElementById("transcribeMovesPreview");
    this.videosListEl = document.getElementById("videosList");

    this.initBoardGrid();
    this.bindEvents();
    this.loadVideoDevices();
    this.connectWebSocket();
  }

  initBoardGrid() {
    this.boardEl.innerHTML = "";
    const files = ["a", "b", "c", "d", "e", "f", "g", "h"];

    for (let r = 8; r >= 1; r--) {
      for (let f = 0; f < 8; f++) {
        const file = files[f];
        const sq = file + r;
        const isLight = (f + r) % 2 === 0;

        const sqEl = document.createElement("div");
        sqEl.className = `square ${isLight ? "light" : "dark"}`;
        sqEl.id = `sq_${sq}`;
        sqEl.dataset.square = sq;

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
    this.btnCalibrateNow.addEventListener("click", async () => {
      const res = await this.postAction("/api/action/calibrate");
      if (res && res.success) {
        this.showToast(`Доска откалибрована (${(res.board_orientation || "").toUpperCase()}). Можно нажать «▶ Старт».`);
      } else {
        this.showToast("Не удалось автоматически найти доску на кадре. Попробуйте потянуть углы вручную или поправьте камеру.", true);
      }
    });
    this.btnRollback.addEventListener("click", () => {
      this.viewPly = null;
      this.postAction("/api/action/rollback");
    });
    this.btnRecalibrate.addEventListener("click", () => {
      this.corners = null;
      if (this.cornersCanvas) {
        const ctx = this.cornersCanvas.getContext("2d");
        ctx.clearRect(0, 0, this.cornersCanvas.width, this.cornersCanvas.height);
      }
      this.postAction("/api/action/recalibrate");
    });
    this.btnResetSession.addEventListener("click", async () => {
      if (!confirm("Сбросить текущую партию и вернуть доску в начальную позицию?")) return;
      this.viewPly = null;
      const res = await this.postAction("/api/action/reset_session");
      if (res && res.success) {
        this.showToast("Сессия сброшена в начальную позицию.");
      }
    });
    this.btnNewSession.addEventListener("click", async () => {
      this.viewPly = null;
      const res = await this.postAction("/api/sessions/new", { pause: true });
      if (res && res.success) {
        this.showToast(`Создана новая сессия: ${res.session_id}`);
      }
    });
    this.btnDownloadPgn.addEventListener("click", () => {
      window.open("/api/pgn?download=true", "_blank");
    });

    // Video device selection & recording
    this.btnRefreshDevices.addEventListener("click", () => this.loadVideoDevices());
    this.videoDeviceSelect.addEventListener("change", async () => {
      const src = this.videoDeviceSelect.value;
      if (!src) return;
      const res = await this.postAction("/api/video/source", { source: src });
      if (res && res.success) {
        this.showToast(`Источник переключён на: ${src} (режим паузы для настройки кадра)`);
        this.loadVideoDevices();
      } else {
        this.showToast(`Не удалось открыть видеоустройство ${src}`, true);
        this.loadVideoDevices();
      }
    });

    this.btnRecordVideo.addEventListener("click", async () => {
      if (!this.isRecordingVideo) {
        const res = await this.postAction("/api/recording/start");
        if (res && res.success) {
          this.showToast("Запись видео с камеры началась.");
        } else {
          this.showToast("Не удалось запустить запись видео.", true);
        }
      } else {
        const res = await this.postAction("/api/recording/stop");
        if (res && res.success && res.video) {
          const url = res.video.download_url;
          this.showToastHtml(
            `Видео сохранено (${res.video.frame_count} кадров, ${res.video.duration_sec}s): ` +
            `<a href="${url}" target="_blank">Скачать ${res.video.filename}</a>`
          );
        }
      }
    });

    // Sessions Modal
    this.btnSessionsModal.addEventListener("click", () => {
      this.sessionsModal.classList.add("open");
      this.loadSessionsList();
    });
    this.btnCloseSessionsModal.addEventListener("click", () => {
      this.sessionsModal.classList.remove("open");
    });
    this.btnCreateNamedSession.addEventListener("click", async () => {
      const white = this.newWhiteInput.value.trim() || null;
      const black = this.newBlackInput.value.trim() || null;
      this.viewPly = null;
      const res = await this.postAction("/api/sessions/new", {
        white_player: white,
        black_player: black,
        pause: true,
      });
      if (res && res.success) {
        this.sessionsModal.classList.remove("open");
        this.showToast(`Создана сессия ${res.session_id}`);
      }
    });

    // Video Transcribe Modal
    this.btnTranscribeModal.addEventListener("click", () => {
      this.transcribeModal.classList.add("open");
      this.loadVideosList();
      this.refreshTranscribeStatus();
    });
    this.btnCloseTranscribeModal.addEventListener("click", () => {
      this.transcribeModal.classList.remove("open");
    });
    this.btnUploadAndTranscribe.addEventListener("click", () => this.uploadAndTranscribe());

    // Move Scrubber Navigation
    this.btnReplayStart.addEventListener("click", () => this.setViewPly(0));
    this.btnReplayPrev.addEventListener("click", () => this.stepViewPly(-1));
    this.btnReplayNext.addEventListener("click", () => this.stepViewPly(1));
    this.btnReplayEnd.addEventListener("click", () => this.setViewPly(null));

    window.addEventListener("keydown", (e) => {
      const tag = (document.activeElement && document.activeElement.tagName) || "";
      if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") return;
      if (e.key === "ArrowLeft") this.stepViewPly(-1);
      else if (e.key === "ArrowRight") this.stepViewPly(1);
      else if (e.key === "Home") this.setViewPly(0);
      else if (e.key === "End") this.setViewPly(null);
    });

    // Corner handles
    if (this.cornersCanvas) {
      this.cornersCanvas.addEventListener("mousedown", (e) => this.handleCornerMouseDown(e));
      window.addEventListener("mousemove", (e) => this.handleCornerMouseMove(e));
      window.addEventListener("mouseup", (e) => this.handleCornerMouseUp(e));
    }
    window.addEventListener("resize", () => this.renderCornerHandles());

    // Uncertain resolution
    if (this.btnToggleUncertain) {
      this.btnToggleUncertain.addEventListener("click", () => {
        this.isUncertainMinimized = !this.isUncertainMinimized;
        this.applyUncertainMinimizeState();
      });
    }
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

  showToast(msg, isError = false) {
    this.showToastHtml(this.escapeHtml(msg), isError);
  }

  showToastHtml(html, isError = false) {
    if (!this.statusToast) return;
    this.statusToast.innerHTML = `<span>${html}</span><button class="btn btn-sm" onclick="this.parentElement.style.display='none'">✕</button>`;
    this.statusToast.className = `status-toast ${isError ? "toast-error" : ""}`;
    this.statusToast.style.display = "flex";
  }

  escapeHtml(text) {
    const div = document.createElement("div");
    div.textContent = text;
    return div.innerHTML;
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
    this.ws.onerror = () => {
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
    } else if (msg.type === "transcribeStatus") {
      this.renderTranscribeProgress(msg);
    }
  }

  updateState(state) {
    const prevSid = this.lastState ? this.lastState.session_id : null;
    if (prevSid && state.session_id && prevSid !== state.session_id) {
      this.viewPly = null;
    }
    this.lastState = state;

    if (this.sessionBadge && state.session_id) {
      this.sessionBadge.textContent = state.session_id;
    }
    if (this.metaWhiteEl && state.white_player) this.metaWhiteEl.textContent = state.white_player;
    if (this.metaBlackEl && state.black_player) this.metaBlackEl.textContent = state.black_player;
    if (this.videoSourceLabel && state.video_source) {
      this.videoSourceLabel.textContent = `Source: ${state.video_source}`;
    }

    this.badgeFsm.textContent = `FSM: ${state.fsm_state}`;
    this.badgeFps.textContent = `${state.fps} FPS`;

    // Pause indicator
    this.isPaused = !!state.is_paused;
    this.btnPause.textContent = this.isPaused ? "▶ Старт" : "⏸ Пауза";
    if (this.isPaused) {
      this.btnPause.classList.add("btn-success");
      this.btnPause.classList.remove("btn-warning");
      if (this.pauseBanner) this.pauseBanner.style.display = "block";
    } else {
      this.btnPause.classList.remove("btn-success");
      this.btnPause.classList.remove("btn-warning");
      if (this.pauseBanner) this.pauseBanner.style.display = "none";
    }

    // Video Recording indicator
    const rec = state.video_recording || {};
    this.isRecordingVideo = !!rec.is_recording;
    if (this.isRecordingVideo) {
      this.btnRecordVideo.textContent = "⏹ Стоп видео";
      this.btnRecordVideo.classList.add("btn-danger");
      if (this.badgeRecStatus) {
        this.badgeRecStatus.style.display = "inline-flex";
        const secs = Math.floor(rec.duration_sec || 0);
        const mm = Math.floor(secs / 60);
        const ss = String(secs % 60).padStart(2, "0");
        this.recTimerLabel.textContent = `REC ${mm}:${ss}`;
      }
    } else {
      this.btnRecordVideo.textContent = "⏺ Запись видео";
      this.btnRecordVideo.classList.remove("btn-danger");
      if (this.badgeRecStatus) this.badgeRecStatus.style.display = "none";
    }

    // Telemetry
    this.metaPlyEl.textContent = `${state.ply} (${Math.floor((state.ply + 1) / 2)})`;
    const oriSuffix = state.board_orientation && state.board_orientation !== "unknown"
      ? ` (${state.board_orientation.replace("_", " ").toUpperCase()})`
      : "";
    this.metaBoardEl.textContent = state.board_detected ? `LOCKED${oriSuffix}` : "SEARCHING...";
    this.metaBoardEl.style.color = state.board_detected ? "var(--color-success)" : "var(--color-warning)";

    this.boardOrientation = state.board_orientation || "white_bottom";
    if (state.board_corners) {
      this.onBoardCornersReceived(state.board_corners);
    } else if (!state.board_detected && !this.isDragging) {
      this.corners = null;
      if (this.cornersCanvas) {
        const ctx = this.cornersCanvas.getContext("2d");
        ctx.clearRect(0, 0, this.cornersCanvas.width, this.cornersCanvas.height);
      }
    }

    this.updateEvalGauge(state.engine_eval);
    this.renderBoardAndReplay();

    if (state.active_uncertainty) {
      this.showUncertainModal(state.uncertain_candidates, state.uncertainty_reason, state.uncertain_squares);
    } else {
      this.hideUncertainModal();
    }

    if (state.video_transcription) {
      this.renderTranscribeProgress(state.video_transcription);
    }
  }

  setViewPly(ply) {
    if (!this.lastState) return;
    const history = this.lastState.move_history || [];
    if (ply === null || ply >= history.length) {
      this.viewPly = null;
    } else {
      this.viewPly = Math.max(0, ply);
    }
    this.renderBoardAndReplay();
  }

  stepViewPly(delta) {
    if (!this.lastState) return;
    const history = this.lastState.move_history || [];
    if (history.length === 0) return;
    const current = this.viewPly === null ? history.length : this.viewPly;
    const next = Math.max(0, Math.min(history.length, current + delta));
    if (next >= history.length) {
      this.viewPly = null;
    } else {
      this.viewPly = next;
    }
    this.renderBoardAndReplay();
  }

  renderBoardAndReplay() {
    if (!this.lastState) return;
    const state = this.lastState;
    const history = state.move_history || [];
    const total = history.length;

    if (this.viewPly !== null && this.viewPly > total) {
      this.viewPly = null;
    }

    let displayFen = state.fen;
    let displayMove = state.last_move;
    let isCheck = state.is_check;
    let turn = state.turn;
    let uncertainSquares = state.uncertain_squares;
    let activePly = total;

    if (this.viewPly !== null) {
      activePly = this.viewPly;
      if (this.viewPly === 0) {
        displayFen = state.initial_fen || INITIAL_FEN;
        displayMove = null;
      } else {
        const entry = history[this.viewPly - 1];
        displayFen = (entry && entry.fen_after) ? entry.fen_after : state.fen;
        displayMove = entry;
      }
      isCheck = false;
      uncertainSquares = [];
      turn = (this.viewPly % 2 === 0) ? "white" : "black";
    }

    if (this.replayIndicator) {
      if (this.viewPly === null) {
        this.replayIndicator.textContent = `LIVE • Ход ${total}/${total}`;
        this.replayIndicator.classList.remove("is-reviewing");
      } else {
        this.replayIndicator.textContent = `ПРОСМОТР • ${activePly}/${total}`;
        this.replayIndicator.classList.add("is-reviewing");
      }
    }

    if (this.btnReplayStart) this.btnReplayStart.disabled = (total === 0 || activePly === 0);
    if (this.btnReplayPrev) this.btnReplayPrev.disabled = (total === 0 || activePly === 0);
    if (this.btnReplayNext) this.btnReplayNext.disabled = (this.viewPly === null);
    if (this.btnReplayEnd) this.btnReplayEnd.disabled = (this.viewPly === null);

    this.renderFen(displayFen, displayMove, isCheck, turn, uncertainSquares);
    this.updateMoveTable(history, state.engine_eval, activePly);
  }

  renderFen(fen, lastMove, isCheck, turn, uncertainSquares = []) {
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

    if (lastMove && lastMove.uci && lastMove.uci.length >= 4) {
      const fromSq = lastMove.uci.slice(0, 2);
      const toSq = lastMove.uci.slice(2, 4);
      const fromEl = document.getElementById(`sq_${fromSq}`);
      const toEl = document.getElementById(`sq_${toSq}`);
      if (fromEl) fromEl.classList.add("move-source");
      if (toEl) toEl.classList.add("move-target");
    }

    if (kingSquare) kingSquare.classList.add("in-check");

    if (Array.isArray(uncertainSquares)) {
      uncertainSquares.forEach((sqName) => {
        const el = document.getElementById(`sq_${sqName}`);
        if (el) el.classList.add("in-check");
      });
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
      const pct = Math.max(5, Math.min(95, 50 + cp / 25));
      this.evalFillEl.style.height = `${pct}%`;
      const sign = cp >= 0 ? "+" : "";
      this.evalLabelEl.textContent = `${sign}${(cp / 100).toFixed(1)}`;
    }
  }

  updateMoveTable(history, latestEval, activePly = null) {
    if (!history) return;

    this.movesBodyEl.innerHTML = "";
    const totalMoves = Math.ceil(history.length / 2);

    for (let m = 0; m < totalMoves; m++) {
      const moveNum = m + 1;
      const whitePly = history[m * 2];
      const blackPly = history[m * 2 + 1];

      const tr = document.createElement("tr");
      if (m === totalMoves - 1 && this.viewPly === null) {
        tr.className = "latest-move";
      }

      const tdNum = document.createElement("td");
      tdNum.className = "move-num";
      tdNum.textContent = `${moveNum}.`;

      const tdWhite = document.createElement("td");
      tdWhite.className = "move-white move-cell";
      if (whitePly) {
        tdWhite.textContent = whitePly.san;
        if (activePly === whitePly.ply) tdWhite.classList.add("active-ply");
        tdWhite.addEventListener("click", () => this.setViewPly(whitePly.ply));
      }

      const tdBlack = document.createElement("td");
      tdBlack.className = "move-black move-cell";
      if (blackPly) {
        tdBlack.textContent = blackPly.san;
        if (activePly === blackPly.ply) tdBlack.classList.add("active-ply");
        tdBlack.addEventListener("click", () => this.setViewPly(blackPly.ply));
      }

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

    if (this.viewPly === null) {
      this.notationScrollEl.scrollTop = this.notationScrollEl.scrollHeight;
    }
  }

  async loadVideoDevices() {
    if (!this.videoDeviceSelect) return;
    try {
      const res = await fetch("/api/video/devices");
      if (!res.ok) return;
      const data = await res.json();
      const devices = data.devices || [];
      this.videoDeviceSelect.innerHTML = "";
      if (devices.length === 0) {
        const opt = document.createElement("option");
        opt.value = data.active_source || "0";
        opt.textContent = `Устройство (${opt.value})`;
        this.videoDeviceSelect.appendChild(opt);
      } else {
        devices.forEach((dev) => {
          const opt = document.createElement("option");
          opt.value = dev.source;
          opt.textContent = dev.available ? dev.label : `${dev.label} [недоступно]`;
          if (dev.active) opt.selected = true;
          this.videoDeviceSelect.appendChild(opt);
        });
      }
    } catch (err) {
      console.error("Error fetching video devices:", err);
    }
  }

  async loadSessionsList() {
    if (!this.sessionsListEl) return;
    this.sessionsListEl.innerHTML = '<div style="color: var(--text-dim);">Загрузка списка сессий...</div>';
    try {
      const res = await fetch("/api/sessions");
      const data = await res.json();
      const list = data.sessions || [];
      this.sessionsListEl.innerHTML = "";
      if (list.length === 0) {
        this.sessionsListEl.innerHTML = '<div style="color: var(--text-dim);">Сохранённых сессий пока нет.</div>';
        return;
      }
      list.forEach((s) => {
        const row = document.createElement("div");
        row.className = `session-item ${s.is_current ? "is-current" : ""}`;
        const dateStr = s.created_at ? s.created_at.slice(0, 19).replace("T", " ") : "";
        row.innerHTML = `
          <div>
            <div><strong>${this.escapeHtml(s.session_id)}</strong> ${s.is_current ? "(текущая)" : ""}</div>
            <div class="session-item-meta">${this.escapeHtml(s.white_player)} vs ${this.escapeHtml(s.black_player)} • Полуходов: ${s.total_plies} • ${dateStr}</div>
            ${s.moves_preview ? `<div class="session-item-meta" style="font-family:monospace;">… ${this.escapeHtml(s.moves_preview)}</div>` : ""}
          </div>
          <button class="btn btn-sm ${s.is_current ? "" : "btn-primary"}" ${s.is_current ? "disabled" : ""}>
            ${s.is_current ? "Активна" : "Открыть"}
          </button>
        `;
        const btn = row.querySelector("button");
        if (!s.is_current && btn) {
          btn.addEventListener("click", async () => {
            this.viewPly = null;
            const r = await this.postAction(`/api/sessions/${encodeURIComponent(s.session_id)}/resume`);
            if (r && r.success) {
              this.sessionsModal.classList.remove("open");
              this.showToast(`Загружена сессия ${s.session_id} (${s.total_plies} полуходов).`);
            } else {
              this.showToast(`Не удалось загрузить сессию ${s.session_id}`, true);
            }
          });
        }
        this.sessionsListEl.appendChild(row);
      });
    } catch (err) {
      this.sessionsListEl.innerHTML = '<div style="color: var(--color-danger);">Ошибка загрузки сессий</div>';
    }
  }

  async loadVideosList() {
    if (!this.videosListEl) return;
    try {
      const res = await fetch("/api/videos");
      const data = await res.json();
      const videos = data.videos || [];
      this.videosListEl.innerHTML = "";
      if (videos.length === 0) {
        this.videosListEl.innerHTML = '<div style="color: var(--text-dim); font-size:0.8rem;">Сохранённых видео пока нет.</div>';
        return;
      }
      videos.forEach((v) => {
        const row = document.createElement("div");
        row.className = "session-item";
        row.innerHTML = `
          <div>
            <div><strong>${this.escapeHtml(v.name)}</strong> <span class="q-badge" style="background:#334155;">${this.escapeHtml(v.category)}</span></div>
            <div class="session-item-meta">${v.size_mb} МБ</div>
          </div>
          <div class="btn-group">
            <a class="btn btn-sm" href="${v.download_url}" target="_blank">⬇ Скачать</a>
            <button class="btn btn-sm btn-primary">🎞 Расшифровать</button>
          </div>
        `;
        row.querySelector("button").addEventListener("click", () => this.startTranscription(v.path));
        this.videosListEl.appendChild(row);
      });
    } catch (err) {
      console.error("Failed loading videos:", err);
    }
  }

  async uploadAndTranscribe() {
    const file = this.videoFileInput && this.videoFileInput.files && this.videoFileInput.files[0];
    if (!file) {
      this.showToast("Выберите видеофайл на компьютере.", true);
      return;
    }
    this.btnUploadAndTranscribe.disabled = true;
    this.transcribeProgressBox.style.display = "block";
    this.transcribeStatusText.textContent = `Загрузка файла ${file.name}...`;
    try {
      const uploadRes = await fetch(`/api/video/upload?filename=${encodeURIComponent(file.name)}`, {
        method: "POST",
        body: file,
      });
      const uploadData = await uploadRes.json();
      if (!uploadRes.ok || !uploadData.success) {
        throw new Error(uploadData.detail || "Upload failed");
      }
      await this.startTranscription(uploadData.path);
      this.loadVideosList();
    } catch (err) {
      this.transcribeStatusText.textContent = `Ошибка: ${err.message}`;
    } finally {
      this.btnUploadAndTranscribe.disabled = false;
    }
  }

  async startTranscription(videoPath) {
    const white = (this.transcribeWhiteInput && this.transcribeWhiteInput.value.trim()) || "White Player";
    const black = (this.transcribeBlackInput && this.transcribeBlackInput.value.trim()) || "Black Player";
    this.transcribeProgressBox.style.display = "block";
    this.transcribeStatusText.textContent = "Запуск расшифровки видео...";
    const res = await this.postAction("/api/video/transcribe", {
      video_path: videoPath,
      white_player: white,
      black_player: black,
      target_fps: 6.0,
    });
    if (res && res.success) {
      this.renderTranscribeProgress(res);
    }
  }

  async refreshTranscribeStatus() {
    try {
      const res = await fetch("/api/video/transcribe/status");
      if (res.ok) {
        const data = await res.json();
        this.renderTranscribeProgress(data);
      }
    } catch (e) {}
  }

  renderTranscribeProgress(job) {
    if (!this.transcribeProgressBox || !job || job.status === "idle") return;
    this.transcribeProgressBox.style.display = "block";
    this.transcribeProgressFill.style.width = `${job.progress_pct || 0}%`;

    if (job.status === "running") {
      this.transcribeStatusText.textContent =
        `Расшифровка ${job.video_name}: ${job.progress_pct}% (${job.frames_processed}/${job.total_frames || "?"} кадров), найдено полуходов: ${job.plies_found}`;
    } else if (job.status === "completed") {
      this.transcribeStatusText.textContent =
        `Готово! Сессия ${job.session_id} загружена на доску (${job.plies_found} полуходов).`;
      this.showToast(`Видео «${job.video_name}» расшифровано (${job.plies_found} полуходов). Листайте ходы на правой доске.`);
    } else if (job.status === "failed") {
      this.transcribeStatusText.textContent = `Ошибка расшифровки: ${job.error || "unknown error"}`;
    }

    if (this.transcribeMovesPreview && Array.isArray(job.moves)) {
      this.transcribeMovesPreview.textContent = job.moves.join(" ");
    }
  }

  applyUncertainMinimizeState() {
    if (!this.modalOverlay || !this.btnToggleUncertain) return;
    if (this.isUncertainMinimized) {
      this.modalOverlay.classList.add("minimized");
      this.btnToggleUncertain.textContent = "▴ Развернуть";
    } else {
      this.modalOverlay.classList.remove("minimized");
      this.btnToggleUncertain.textContent = "▾ Свернуть";
    }
  }

  showUncertainModal(candidates, reason = null, uncertainSquares = []) {
    const sig = JSON.stringify({ candidates: candidates || [], reason: reason || "", squares: uncertainSquares || [] });
    if (sig === this.lastUncertainSig && this.modalOverlay.classList.contains("open")) {
      return;
    }
    if (!this.modalOverlay.classList.contains("open")) {
      this.isUncertainMinimized = false;
    }
    this.lastUncertainSig = sig;
    this.applyUncertainMinimizeState();

    this.candidateListEl.innerHTML = "";
    const hasCandidates = candidates && candidates.length > 0;

    if (this.modalTitleEl) {
      this.modalTitleEl.textContent = hasCandidates
        ? "⚠️ Неопределённый ход"
        : "⛔ Нелегальный ход или несовпадение доски";
    }
    if (this.modalBodyEl) {
      if (reason) {
        this.modalBodyEl.textContent = reason;
      } else if (hasCandidates) {
        this.modalBodyEl.textContent =
          "Несколько ходов имеют близкую уверенность. Выберите верный или введите UCI:";
      } else {
        const sqText = Array.isArray(uncertainSquares) && uncertainSquares.length > 0
          ? ` (${uncertainSquares.join(", ")})`
          : "";
        this.modalBodyEl.textContent =
          `Изменение на клетках${sqText} не соответствует ни одному легальному ходу. Верните фигуры на место или введите UCI:`;
      }
    }

    if (hasCandidates) {
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
      const illCount = (this.lastState && this.lastState.illegal_attempts) ? this.lastState.illegal_attempts : 0;
      const illNote = illCount > 0 ? ` Зафиксировано нарушений / нелегальных ходов: ${illCount}.` : "";
      emptyMsg.textContent = `Нет легального хода, объясняющего изменение. Верните фигуры или введите UCI вручную.${illNote}`;
      this.candidateListEl.appendChild(emptyMsg);
    }

    this.modalOverlay.classList.add("open");
  }

  hideUncertainModal() {
    this.modalOverlay.classList.remove("open", "minimized");
    this.isUncertainMinimized = false;
    this.lastUncertainSig = null;
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

    const labelMap = {
      white_bottom: ["a8", "h8", "h1", "a1"],
      black_bottom: ["h1", "a1", "a8", "h8"],
      white_right:  ["h8", "h1", "a1", "a8"],
      white_left:   ["a1", "a8", "h8", "h1"],
    };
    const labels = labelMap[this.boardOrientation] || ["a8", "h8", "h1", "a1"];
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

  handleCornerMouseUp() {
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

  async postAction(url, body = {}) {
    try {
      const res = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) {
        console.error(`Action ${url} failed:`, data);
        return null;
      }
      return data;
    } catch (err) {
      console.error(`Error dispatching action to ${url}:`, err);
      return null;
    }
  }

  async resolveUncertainty(payload) {
    await this.postAction("/api/action/resolve", payload);
    this.hideUncertainModal();
  }
}

document.addEventListener("DOMContentLoaded", () => {
  window.dashboard = new ChessWebDashboard();
});
