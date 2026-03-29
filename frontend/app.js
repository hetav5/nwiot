const modeImageBtn = document.getElementById("modeImageBtn");
const modeIpcamBtn = document.getElementById("modeIpcamBtn");
const imageModePanel = document.getElementById("imageModePanel");
const ipcamModePanel = document.getElementById("ipcamModePanel");

const imageFile = document.getElementById("imageFile");
const analyzeImageBtn = document.getElementById("analyzeImageBtn");

const ipcamUrl = document.getElementById("ipcamUrl");
const captureIpcamBtn = document.getElementById("captureIpcamBtn");
const toggleLiveBtn = document.getElementById("toggleLiveBtn");
const liveInterval = document.getElementById("liveInterval");

const feedbackText = document.getElementById("feedbackText");
const previewImage = document.getElementById("previewImage");
const previewPlaceholder = document.getElementById("previewPlaceholder");

const metricTotal = document.getElementById("metricTotal");
const metricOccupied = document.getElementById("metricOccupied");
const metricFree = document.getElementById("metricFree");

let mode = "image";
let liveTimer = null;
let requestInFlight = false;

function setMode(nextMode) {
  mode = nextMode;
  const imageMode = mode === "image";
  modeImageBtn.classList.toggle("active", imageMode);
  modeIpcamBtn.classList.toggle("active", !imageMode);
  imageModePanel.classList.toggle("hidden", !imageMode);
  ipcamModePanel.classList.toggle("hidden", imageMode);

  if (imageMode && liveTimer) {
    stopLiveMode();
  }
}

function getVal(id) {
  return document.getElementById(id).value.trim();
}

function getOptionsPayload() {
  return {
    modelPath: getVal("modelPath") || "yolo11n.pt",
    backend: getVal("backend") || "ultralytics",
    slotsPath: getVal("slotsPath") || "annotated_parking_coords.txt",
    imgsz: getVal("imgsz") || "416",
    conf: getVal("conf") || "0.25",
    device: getVal("device") || "cpu",
    gridRows: getVal("gridRows") || "0",
    gridCols: getVal("gridCols") || "0",
    gridRoi: getVal("gridRoi") || ""
  };
}

function setBusy(isBusy, text) {
  requestInFlight = isBusy;
  analyzeImageBtn.disabled = isBusy;
  captureIpcamBtn.disabled = isBusy;
  if (!liveTimer) {
    toggleLiveBtn.disabled = isBusy;
  }
  feedbackText.textContent = text;
  feedbackText.classList.remove("error");
}

function setError(text) {
  feedbackText.textContent = text;
  feedbackText.classList.add("error");
}

function applyResult(data) {
  metricTotal.textContent = data.total;
  metricOccupied.textContent = data.occupied;
  metricFree.textContent = data.free;
  previewImage.src = data.image;
  previewPlaceholder.classList.add("hidden");
  previewImage.classList.remove("hidden");
  feedbackText.textContent = `Processed successfully: ${data.free} free / ${data.total} total.`;
  feedbackText.classList.remove("error");
}

async function analyzeUploadedImage() {
  if (requestInFlight) {
    return;
  }

  if (!imageFile.files || imageFile.files.length === 0) {
    setError("Please choose an image first.");
    return;
  }

  setBusy(true, "Processing image...");
  try {
    const body = new FormData();
    body.append("image", imageFile.files[0]);

    const options = getOptionsPayload();
    Object.keys(options).forEach((k) => body.append(k, options[k]));

    const res = await fetch("/api/process-image", {
      method: "POST",
      body
    });
    const data = await res.json();
    if (!res.ok || !data.ok) {
      throw new Error(data.error || "Image processing failed");
    }
    applyResult(data);
  } catch (err) {
    setError(err.message || "Image processing failed");
  } finally {
    setBusy(false, "Ready.");
  }
}

async function analyzeIpcamFrame() {
  if (requestInFlight) {
    return;
  }

  const url = ipcamUrl.value.trim();
  if (!url) {
    setError("Please enter your IP webcam URL.");
    return;
  }

  setBusy(true, "Fetching frame from IP webcam...");
  try {
    const payload = {
      ipUrl: url,
      ...getOptionsPayload()
    };

    const res = await fetch("/api/process-ipcam", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });

    const data = await res.json();
    if (!res.ok || !data.ok) {
      throw new Error(data.error || "IP webcam processing failed");
    }
    applyResult(data);
  } catch (err) {
    setError(err.message || "IP webcam processing failed");
  } finally {
    setBusy(false, "Ready.");
  }
}

function startLiveMode() {
  const sec = Number.parseInt(liveInterval.value, 10);
  const intervalSeconds = Number.isNaN(sec) ? 2 : Math.max(1, sec);

  toggleLiveBtn.textContent = "Stop Live Pull";
  feedbackText.textContent = `Live pull enabled (${intervalSeconds}s).`;
  feedbackText.classList.remove("error");

  liveTimer = setInterval(async () => {
    if (!requestInFlight) {
      await analyzeIpcamFrame();
    }
  }, intervalSeconds * 1000);
}

function stopLiveMode() {
  if (liveTimer) {
    clearInterval(liveTimer);
    liveTimer = null;
  }
  toggleLiveBtn.textContent = "Start Live Pull";
  toggleLiveBtn.disabled = false;
}

function toggleLivePull() {
  if (liveTimer) {
    stopLiveMode();
    feedbackText.textContent = "Live pull stopped.";
    feedbackText.classList.remove("error");
    return;
  }

  if (!ipcamUrl.value.trim()) {
    setError("Please enter your IP webcam URL before starting live pull.");
    return;
  }

  startLiveMode();
}

modeImageBtn.addEventListener("click", () => setMode("image"));
modeIpcamBtn.addEventListener("click", () => setMode("ipcam"));

analyzeImageBtn.addEventListener("click", analyzeUploadedImage);
captureIpcamBtn.addEventListener("click", analyzeIpcamFrame);
toggleLiveBtn.addEventListener("click", toggleLivePull);

window.addEventListener("beforeunload", () => {
  stopLiveMode();
});

previewImage.classList.add("hidden");
setMode("image");

buildCommand();
