import Chart from "chart.js/auto";

const shell = document.querySelector("#viewer-shell");
const emptyState = document.querySelector("#empty-state");
const emptyMessage = document.querySelector("#empty-message");
const summary = document.querySelector("#summary");
const cautionList = document.querySelector("#caution-list");
const uncertaintyTable = document.querySelector("#uncertainty-table");
const architectureList = document.querySelector("#architecture-list");
const architectureChipRow = document.querySelector("#architecture-chip-row");
const neuroChipRow = document.querySelector("#neuro-chip-row");
const performanceChipRow = document.querySelector("#performance-chip-row");
const deviceChip = document.querySelector("#device-chip");
const projectTitle = document.querySelector("#project-title");
const scrollButtons = document.querySelectorAll("[data-scroll-target]");
const focusButtons = document.querySelectorAll("[data-scene-focus]");
const fullscreenButtons = document.querySelectorAll("[data-scene-fullscreen]");

const charts = [];
let neuroSceneStarted = false;
let networkSceneStarted = false;

function detectPerformanceProfile() {
  const reducedMotion = window.matchMedia?.("(prefers-reduced-motion: reduce)")?.matches ?? false;
  const saveData = navigator.connection?.saveData ?? false;
  const lowCpu = (navigator.hardwareConcurrency ?? 8) <= 4;
  const lowMemory = (navigator.deviceMemory ?? 8) <= 4;
  const lowMotion = reducedMotion || saveData || lowCpu || lowMemory;

  return {
    lowMotion,
    reducedMotion,
    saveData,
    lowCpu,
    lowMemory,
    fps: lowMotion ? 18 : 30,
    maxDpr: lowMotion ? 1.1 : 1.5,
    chartAnimation: lowMotion ? false : { duration: 420, easing: "easeOutQuad" },
    lineTension: lowMotion ? 0.16 : 0.25,
    pointRadius: lowMotion ? 2 : 3,
    bubbleRadiusScale: lowMotion ? 16 : 22,
    neuroNodesPerCluster: lowMotion ? 12 : 18,
    neuroConnectionSpan: lowMotion ? 2 : 3,
    neuroRingCount: lowMotion ? 2 : 3,
    networkConnectionModulo: lowMotion ? 4 : 3,
    showPulse: !lowMotion,
    shadowBlur: lowMotion ? 0 : 18,
  };
}

const performanceProfile = detectPerformanceProfile();

function metricCard(label, value) {
  return `
    <article class="metric-card">
      <div class="metric-label">${label}</div>
      <div class="metric-value">${value}</div>
    </article>
  `;
}

function chip(label) {
  return `<span class="chip">${label}</span>`;
}

function formatMetric(value, digits = 3) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) {
    return "n/a";
  }
  return Number(value).toFixed(digits);
}

function destroyCharts() {
  while (charts.length > 0) {
    charts.pop().destroy();
  }
}

function registerChart(chart) {
  charts.push(chart);
  return chart;
}

function renderEmptyState(message) {
  shell.classList.add("is-hidden");
  emptyState.classList.remove("is-hidden");
  emptyMessage.textContent = message;
}

function bindLayoutControls() {
  scrollButtons.forEach((button) => {
    button.addEventListener("click", () => {
      const section = document.getElementById(button.dataset.scrollTarget);
      section?.scrollIntoView({ behavior: "smooth", block: "start" });
    });
  });

  focusButtons.forEach((button) => {
    button.addEventListener("click", () => {
      const target = document.getElementById(button.dataset.sceneFocus);
      document.querySelectorAll(".scene-card").forEach((card) => card.classList.remove("is-focused"));
      target?.classList.add("is-focused");
      target?.scrollIntoView({ behavior: "smooth", block: "center" });
    });
  });

  fullscreenButtons.forEach((button) => {
    button.addEventListener("click", async () => {
      const canvas = document.getElementById(button.dataset.sceneFullscreen);
      if (!canvas?.requestFullscreen) {
        return;
      }
      try {
        await canvas.requestFullscreen();
      } catch (error) {
        console.warn("Fullscreen request failed", error);
      }
    });
  });
}

function showViewer() {
  shell.classList.remove("is-hidden");
  emptyState.classList.add("is-hidden");
}

function createChart(canvasId, config) {
  const canvas = document.getElementById(canvasId);
  if (!canvas) {
    return null;
  }
  const mergedConfig = {
    ...config,
    options: {
      responsive: true,
      maintainAspectRatio: false,
      devicePixelRatio: performanceProfile.maxDpr,
      normalized: true,
      animation: performanceProfile.chartAnimation,
      ...config.options,
    },
  };
  return registerChart(new Chart(canvas, mergedConfig));
}

function renderMetrics(payload) {
  const raw = payload.metrics.raw_test ?? payload.metrics.calibrated_test ?? payload.metrics.mc_dropout_test ?? {};
  const calibrated = payload.metrics.calibrated_test ?? payload.metrics.mc_dropout_test ?? {};
  const mc = payload.metrics.mc_dropout_test ?? {};

  summary.innerHTML = [
    metricCard("Temperature", formatMetric(payload.temperature)),
    metricCard("Best epoch", payload.best_epoch ?? "n/a"),
    metricCard("Raw accuracy", formatMetric(raw.accuracy)),
    metricCard("Calibrated accuracy", formatMetric(calibrated.accuracy)),
    metricCard("Calibrated ECE", formatMetric(calibrated.ece)),
    metricCard("MC accuracy", formatMetric(mc.accuracy)),
    metricCard("MC entropy", formatMetric(mc.mean_entropy)),
    metricCard("MC confidence", formatMetric(mc.mean_confidence)),
  ].join("");
}

function renderCautions(payload) {
  const cautions = payload.cautions?.length
    ? payload.cautions
    : [
        "Research prototype only.",
        "Softmax confidence does not imply clinical certainty.",
        "Model uncertainty should always be displayed.",
      ];
  cautionList.innerHTML = cautions.map((item) => `<li>${item}</li>`).join("");
}

function renderArchitecture(payload) {
  const architecture = payload.architecture ?? {};
  const stages = architecture.stages ?? [];
  architectureList.innerHTML = stages
    .map((stage) => `<li><strong>${stage.name}</strong>: ${stage.nodes} units in group ${stage.group}</li>`)
    .join("");

  architectureChipRow.innerHTML = [
    chip(`Tabular input ${architecture.tabular_input_dim ?? "n/a"}`),
    chip(`Temporal ${architecture.temporal_input_shape?.join(" x ") ?? "n/a"}`),
    chip(`Fusion ${architecture.fusion_hidden_dim ?? "n/a"}`),
    chip(`Dropout ${formatMetric(architecture.dropout, 2)}`),
    chip(`Classes ${architecture.num_classes ?? "n/a"}`),
  ].join("");

  neuroChipRow.innerHTML = [
    chip(`Classes ${payload.label_names.length}`),
    chip(`Temperature ${formatMetric(payload.temperature)}`),
    chip(`Entropy ${formatMetric(payload.metrics.mc_dropout_test?.mean_entropy)}`),
    chip(`ECE ${formatMetric(payload.metrics.calibrated_test?.ece ?? payload.metrics.mc_dropout_test?.ece)}`),
  ].join("");

  performanceChipRow.innerHTML = [
    chip(performanceProfile.lowMotion ? "Auto low-motion mode" : "Full motion mode"),
    chip(performanceProfile.reducedMotion ? "Reduced motion preference" : "Standard motion preference"),
    chip(`Render ${performanceProfile.fps} fps target`),
  ].join("");

  deviceChip.textContent = `Device: ${payload.device ?? "unknown"}`;
  projectTitle.textContent = payload.project ?? projectTitle.textContent;
}

function renderProbabilityCharts(payload) {
  createChart("probability-chart", {
    type: "bar",
    data: {
      labels: payload.label_names,
      datasets: [
        {
          label: "Mean probability",
          data: payload.class_mean_probabilities,
          backgroundColor: ["#145374", "#4f772d", "#b85c38"],
          borderRadius: 14,
        },
      ],
    },
    options: {
      plugins: { legend: { display: false } },
      scales: {
        y: { beginAtZero: true, max: 1, grid: { color: "rgba(16,33,44,0.08)" } },
        x: { grid: { display: false } },
      },
    },
  });

  createChart("variance-chart", {
    type: "bar",
    data: {
      labels: payload.label_names,
      datasets: [
        {
          label: "Mean variance",
          data: payload.class_mean_variances,
          backgroundColor: ["#95d5b2", "#f4a261", "#8ecae6"],
          borderRadius: 14,
        },
      ],
    },
    options: {
      plugins: { legend: { display: false } },
      scales: {
        y: { beginAtZero: true, grid: { color: "rgba(16,33,44,0.08)" } },
        x: { grid: { display: false } },
      },
    },
  });
}

function renderCalibrationChart(payload) {
  const curves = payload.chart_data?.calibration_curves ?? {};
  const pickCenters = (bins) => bins.map((item) => (item.lower + item.upper) * 0.5);
  const pickAccuracies = (bins) => bins.map((item) => item.accuracy);
  const reference = curves.calibrated?.length
    ? curves.calibrated
    : curves.mc_dropout?.length
      ? curves.mc_dropout
      : curves.raw ?? [];

  createChart("calibration-chart", {
    type: "line",
    data: {
      labels: pickCenters(reference),
      datasets: [
        {
          label: "Perfect calibration",
          data: pickCenters(reference),
          borderColor: "#8d99ae",
          borderDash: [6, 6],
          pointRadius: 0,
          tension: 0,
        },
        {
          label: "Raw",
          data: pickAccuracies(curves.raw ?? []),
          borderColor: "#b85c38",
          backgroundColor: "rgba(184,92,56,0.15)",
          pointRadius: performanceProfile.pointRadius,
          tension: performanceProfile.lineTension,
        },
        {
          label: "Calibrated",
          data: pickAccuracies(curves.calibrated ?? []),
          borderColor: "#145374",
          backgroundColor: "rgba(20,83,116,0.15)",
          pointRadius: performanceProfile.pointRadius,
          tension: performanceProfile.lineTension,
        },
        {
          label: "MC Dropout",
          data: pickAccuracies(curves.mc_dropout ?? []),
          borderColor: "#4f772d",
          backgroundColor: "rgba(79,119,45,0.15)",
          pointRadius: performanceProfile.pointRadius,
          tension: performanceProfile.lineTension,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        y: { beginAtZero: true, max: 1, grid: { color: "rgba(16,33,44,0.08)" } },
        x: { title: { display: true, text: "Confidence bin center" }, grid: { display: false } },
      },
    },
  });
}

function renderComparisonChart(payload) {
  const comparison = payload.chart_data?.comparison_metrics;
  if (!comparison) {
    return;
  }
  createChart("comparison-chart", {
    type: "radar",
    data: {
      labels: comparison.labels,
      datasets: [
        {
          label: "Raw",
          data: comparison.raw,
          borderColor: "#b85c38",
          backgroundColor: "rgba(184,92,56,0.14)",
        },
        {
          label: "Calibrated",
          data: comparison.calibrated,
          borderColor: "#145374",
          backgroundColor: "rgba(20,83,116,0.14)",
        },
        {
          label: "MC Dropout",
          data: comparison.mc_dropout,
          borderColor: "#4f772d",
          backgroundColor: "rgba(79,119,45,0.14)",
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        r: {
          beginAtZero: true,
          max: 1,
          pointLabels: { color: "#33414b" },
          grid: { color: "rgba(16,33,44,0.08)" },
          angleLines: { color: "rgba(16,33,44,0.08)" },
        },
      },
    },
  });
}

function renderTrainingCharts(payload) {
  const history = payload.history ?? {};
  const epochs = Array.from({ length: history.train_loss?.length ?? 0 }, (_, index) => `Epoch ${index + 1}`);

  createChart("training-loss-chart", {
    type: "line",
    data: {
      labels: epochs,
      datasets: [
        {
          label: "Train loss",
          data: history.train_loss ?? [],
          borderColor: "#145374",
          backgroundColor: "rgba(20,83,116,0.16)",
          tension: performanceProfile.lineTension,
        },
        {
          label: "Validation loss",
          data: history.validation_loss ?? [],
          borderColor: "#b85c38",
          backgroundColor: "rgba(184,92,56,0.16)",
          tension: performanceProfile.lineTension,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        y: { beginAtZero: true, grid: { color: "rgba(16,33,44,0.08)" } },
        x: { grid: { display: false } },
      },
    },
  });

  createChart("validation-chart", {
    type: "line",
    data: {
      labels: epochs,
      datasets: [
        {
          label: "Validation accuracy",
          data: history.validation_accuracy ?? [],
          borderColor: "#4f772d",
          backgroundColor: "rgba(79,119,45,0.16)",
          tension: performanceProfile.lineTension,
        },
        {
          label: "Validation F1 macro",
          data: history.validation_f1_macro ?? [],
          borderColor: "#d9a441",
          backgroundColor: "rgba(217,164,65,0.18)",
          tension: performanceProfile.lineTension,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        y: { beginAtZero: true, max: 1, grid: { color: "rgba(16,33,44,0.08)" } },
        x: { grid: { display: false } },
      },
    },
  });
}

function renderUncertaintyScatter(payload) {
  const palette = {
    "baseline-risk": "#145374",
    "monitor-closely": "#4f772d",
    "high-risk-flag": "#b85c38",
  };
  const scatterData = payload.chart_data?.uncertainty_scatter ?? [];
  const grouped = new Map();

  scatterData.forEach((point) => {
    const key = point.predicted_label;
    const points = grouped.get(key) ?? [];
    points.push({
      x: point.confidence,
      y: point.entropy,
      r: 5 + point.mutual_information * performanceProfile.bubbleRadiusScale,
    });
    grouped.set(key, points);
  });

  createChart("uncertainty-chart", {
    type: "bubble",
    data: {
      datasets: Array.from(grouped.entries()).map(([label, points]) => ({
        label,
        data: points,
        backgroundColor: `${palette[label] ?? "#145374"}bb`,
        borderColor: palette[label] ?? "#145374",
      })),
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: {
          min: 0,
          max: 1,
          title: { display: true, text: "Confidence" },
          grid: { color: "rgba(16,33,44,0.08)" },
        },
        y: {
          min: 0,
          max: 1.2,
          title: { display: true, text: "Predictive entropy" },
          grid: { color: "rgba(16,33,44,0.08)" },
        },
      },
    },
  });
}

function renderHeatmap(containerId, matrix, labels, color) {
  const container = document.getElementById(containerId);
  if (!container) {
    return;
  }
  if (!matrix?.length) {
    container.innerHTML = "<p>Unavailable for current payload.</p>";
    return;
  }

  const maxValue = Math.max(...matrix.flat(), 1);
  const head = `
    <div class="heatmap-head">
      <div></div>
      ${labels.map((label) => `<div class="heatmap-label">${label}</div>`).join("")}
    </div>
  `;
  const rows = matrix
    .map(
      (row, rowIndex) => `
        <div class="heatmap-row">
          <div class="heatmap-label">${labels[rowIndex]}</div>
          ${row
            .map((value) => {
              const alpha = 0.18 + (value / maxValue) * 0.82;
              return `<div class="heatmap-cell" style="background:${color(alpha)}">${value}</div>`;
            })
            .join("")}
        </div>
      `,
    )
    .join("");

  container.innerHTML = head + rows;
}

function renderUncertaintyTable(payload) {
  uncertaintyTable.innerHTML = payload.uncertain_examples
    .map(
      (item) => `
        <tr>
          <td><span class="pill">#${item.sample_index}</span></td>
          <td>${item.predicted_label}</td>
          <td>${item.true_label}</td>
          <td>${formatMetric(item.confidence)}</td>
          <td>${formatMetric(item.predictive_entropy)}</td>
          <td>${formatMetric(item.mutual_information)}</td>
          <td>${item.mean_probabilities.map((value) => formatMetric(value)).join(" / ")}</td>
        </tr>
      `,
    )
    .join("");
}

function createSceneSurface(canvas) {
  const context = canvas.getContext("2d", { alpha: false });
  const state = {
    context,
    width: 0,
    height: 0,
    active: true,
  };

  function resize() {
    const rect = canvas.getBoundingClientRect();
    const dpr = Math.min(window.devicePixelRatio || 1, performanceProfile.maxDpr);
    state.width = rect.width;
    state.height = rect.height;
    canvas.width = Math.max(1, Math.round(rect.width * dpr));
    canvas.height = Math.max(1, Math.round(rect.height * dpr));
    context.setTransform(dpr, 0, 0, dpr, 0, 0);
  }

  resize();

  if ("ResizeObserver" in window) {
    const observer = new ResizeObserver(() => resize());
    observer.observe(canvas);
  } else {
    window.addEventListener("resize", resize);
  }

  if ("IntersectionObserver" in window) {
    const observer = new IntersectionObserver(
      ([entry]) => {
        state.active = entry?.isIntersecting ?? true;
      },
      { threshold: 0.08 },
    );
    observer.observe(canvas);
  }

  document.addEventListener("visibilitychange", () => {
    state.active = !document.hidden;
  });

  return state;
}

function createPointerState(canvas) {
  const state = { x: 0, y: 0, active: false };
  canvas.addEventListener("pointermove", (event) => {
    const rect = canvas.getBoundingClientRect();
    state.x = (event.clientX - rect.left) / rect.width - 0.5;
    state.y = (event.clientY - rect.top) / rect.height - 0.5;
    state.active = true;
  });
  canvas.addEventListener("pointerleave", () => {
    state.active = false;
  });
  return state;
}

function projectPoint(point, rotation, width, height, depth = 420) {
  const cosY = Math.cos(rotation.y);
  const sinY = Math.sin(rotation.y);
  const cosX = Math.cos(rotation.x);
  const sinX = Math.sin(rotation.x);

  const x1 = point.x * cosY - point.z * sinY;
  const z1 = point.x * sinY + point.z * cosY;
  const y1 = point.y * cosX - z1 * sinX;
  const z2 = point.y * sinX + z1 * cosX;
  const scale = depth / (depth + z2 + 220);

  return {
    x: width / 2 + x1 * scale,
    y: height / 2 + y1 * scale,
    scale,
    z: z2,
  };
}

function startNeuroScene(payload) {
  if (neuroSceneStarted) {
    return;
  }
  neuroSceneStarted = true;

  const canvas = document.querySelector("#neuro-scene");
  const surface = createSceneSurface(canvas);
  const pointer = createPointerState(canvas);
  const palette = ["#4cc9f0", "#90be6d", "#f9844a"];
  const clusterCenters = [
    { x: -110, y: 0, z: -20 },
    { x: 0, y: -10, z: 30 },
    { x: 110, y: 10, z: -10 },
  ];
  const nodes = [];
  let lastTime = 0;

  clusterCenters.forEach((center, clusterIndex) => {
    for (let index = 0; index < performanceProfile.neuroNodesPerCluster; index += 1) {
      nodes.push({
        x: center.x + (Math.random() - 0.5) * 120,
        y: center.y + (Math.random() - 0.5) * 150,
        z: center.z + (Math.random() - 0.5) * 110,
        color: palette[clusterIndex],
        clusterIndex,
        pulseOffset: Math.random() * Math.PI * 2,
      });
    }
  });

  function draw(time) {
    requestAnimationFrame(draw);
    if (!surface.active || time - lastTime < 1000 / performanceProfile.fps) {
      return;
    }
    lastTime = time;

    const { context, width, height } = surface;
    context.clearRect(0, 0, width, height);

    const gradient = context.createRadialGradient(width * 0.5, height * 0.25, 10, width * 0.5, height * 0.5, width * 0.7);
    gradient.addColorStop(0, "rgba(23,78,107,0.18)");
    gradient.addColorStop(1, "rgba(4,12,18,0.92)");
    context.fillStyle = gradient;
    context.fillRect(0, 0, width, height);

    const rotation = {
      x: Math.sin(time * 0.00018) * 0.22 + (pointer.active ? pointer.y * 0.45 : 0),
      y: time * 0.0002 + (pointer.active ? pointer.x * 0.55 : 0),
    };
    const projected = nodes.map((node) => ({
      ...projectPoint(node, rotation, width, height),
      color: node.color,
      clusterIndex: node.clusterIndex,
      pulse: 0.5 + 0.5 * Math.sin(time * 0.003 + node.pulseOffset),
    }));

    projected.sort((a, b) => a.z - b.z);

    for (let index = 0; index < projected.length; index += 1) {
      const current = projected[index];
      for (
        let nextIndex = index + 1;
        nextIndex < Math.min(projected.length, index + performanceProfile.neuroConnectionSpan);
        nextIndex += 1
      ) {
        const next = projected[nextIndex];
        const dx = current.x - next.x;
        const dy = current.y - next.y;
        const distance = Math.sqrt(dx * dx + dy * dy);
        if (current.clusterIndex === next.clusterIndex && distance < 72) {
          context.strokeStyle = `rgba(120, 210, 255, ${0.08 + 0.14 * current.scale})`;
          context.lineWidth = 1;
          context.beginPath();
          context.moveTo(current.x, current.y);
          context.lineTo(next.x, next.y);
          context.stroke();
        }
      }
    }

    projected.forEach((point) => {
      context.beginPath();
      context.fillStyle = point.color;
      context.shadowBlur = performanceProfile.shadowBlur;
      context.shadowColor = point.color;
      context.arc(point.x, point.y, 2 + point.scale * 6 + point.pulse * 1.5, 0, Math.PI * 2);
      context.fill();
    });
    context.shadowBlur = 0;

    context.strokeStyle = "rgba(255,255,255,0.14)";
    context.lineWidth = 1.2;
    for (let ringIndex = 0; ringIndex < performanceProfile.neuroRingCount; ringIndex += 1) {
      const radius = 68 + ringIndex * 36 + Math.sin(time * 0.0013 + ringIndex) * 8;
      context.beginPath();
      context.ellipse(width / 2, height / 2, radius * 1.45, radius * 0.62, 0, 0, Math.PI * 2);
      context.stroke();
    }

    context.fillStyle = "rgba(255,255,255,0.86)";
    context.font = "600 14px Georgia";
    context.fillText("Neuroscientific signal field", 20, 28);
    context.fillStyle = "rgba(197,216,226,0.92)";
    context.font = "12px Georgia";
    context.fillText(`Entropy ${formatMetric(payload.metrics.mc_dropout_test?.mean_entropy)}  |  ECE ${formatMetric(payload.metrics.calibrated_test?.ece ?? payload.metrics.mc_dropout_test?.ece)}`, 20, 48);
  }

  requestAnimationFrame(draw);
}

function buildNetworkNodes(architecture) {
  const stages = architecture.stages?.length
    ? architecture.stages
    : [
        { name: "tabular-input", nodes: 16, group: "input" },
        { name: "tabular-mlp", nodes: 12, group: "tabular" },
        { name: "temporal-conv", nodes: 10, group: "temporal" },
        { name: "fusion", nodes: 8, group: "fusion" },
        { name: "classifier", nodes: 3, group: "output" },
      ];

  return stages.map((stage, stageIndex) => {
    const displayCount = Math.max(3, Math.min(performanceProfile.lowMotion ? 10 : 14, stage.nodes));
    return {
      ...stage,
      nodes3d: Array.from({ length: displayCount }, (_, index) => ({
        x: -180 + stageIndex * 90,
        y: (index - (displayCount - 1) / 2) * 20,
        z: Math.sin(index * 0.8 + stageIndex) * 36,
      })),
    };
  });
}

function startNetworkScene(payload) {
  if (networkSceneStarted) {
    return;
  }
  networkSceneStarted = true;

  const canvas = document.querySelector("#network-scene");
  const surface = createSceneSurface(canvas);
  const pointer = createPointerState(canvas);
  const layers = buildNetworkNodes(payload.architecture ?? {});
  const connections = [];
  let lastTime = 0;

  for (let layerIndex = 0; layerIndex < layers.length - 1; layerIndex += 1) {
    const sourceLayer = layers[layerIndex];
    const targetLayer = layers[layerIndex + 1];
    sourceLayer.nodes3d.forEach((sourceNode, sourceNodeIndex) => {
      targetLayer.nodes3d.forEach((targetNode, targetNodeIndex) => {
        if ((sourceNodeIndex + targetNodeIndex) % performanceProfile.networkConnectionModulo === 0 || targetLayer.nodes3d.length <= 4) {
          connections.push({
            fromLayerIndex: layerIndex,
            toLayerIndex: layerIndex + 1,
            fromIndex: sourceNodeIndex,
            toIndex: targetNodeIndex,
            phase: Math.random(),
          });
        }
      });
    });
  }

  function draw(time) {
    requestAnimationFrame(draw);
    if (!surface.active || time - lastTime < 1000 / performanceProfile.fps) {
      return;
    }
    lastTime = time;

    const { context, width, height } = surface;
    context.clearRect(0, 0, width, height);

    const background = context.createLinearGradient(0, 0, width, height);
    background.addColorStop(0, "rgba(8, 14, 23, 0.98)");
    background.addColorStop(1, "rgba(20, 15, 30, 0.96)");
    context.fillStyle = background;
    context.fillRect(0, 0, width, height);

    const rotation = {
      x: -0.12 + Math.sin(time * 0.0004) * 0.06 + (pointer.active ? pointer.y * 0.35 : 0),
      y: Math.sin(time * 0.0002) * 0.35 + (pointer.active ? pointer.x * 0.65 : 0),
    };
    const projectedLayers = layers.map((layer) => ({
      ...layer,
      projected: layer.nodes3d.map((node, index) =>
        projectPoint(
          {
            x: node.x,
            y: node.y + Math.sin(time * 0.0012 + index) * 2.5,
            z: node.z + Math.cos(time * 0.0009 + index) * 4,
          },
          rotation,
          width,
          height,
          520,
        ),
      ),
    }));

    connections.forEach((connection) => {
      const from = projectedLayers[connection.fromLayerIndex]?.projected[connection.fromIndex];
      const to = projectedLayers[connection.toLayerIndex]?.projected[connection.toIndex];
      if (!from || !to) {
        return;
      }

      context.strokeStyle = "rgba(120, 171, 255, 0.12)";
      context.lineWidth = 1;
      context.beginPath();
      context.moveTo(from.x, from.y);
      context.lineTo(to.x, to.y);
      context.stroke();

      if (performanceProfile.showPulse) {
        const pulseT = (time * 0.00035 + connection.phase) % 1;
        const pulseX = from.x + (to.x - from.x) * pulseT;
        const pulseY = from.y + (to.y - from.y) * pulseT;
        context.beginPath();
        context.fillStyle = "rgba(255, 214, 102, 0.92)";
        context.shadowBlur = 14;
        context.shadowColor = "rgba(255, 214, 102, 0.9)";
        context.arc(pulseX, pulseY, 2.2, 0, Math.PI * 2);
        context.fill();
      }
    });
    context.shadowBlur = 0;

    projectedLayers.forEach((layer, layerIndex) => {
      const hue = [200, 102, 22, 46, 340][layerIndex % 5];
      layer.projected.forEach((node) => {
        context.beginPath();
        context.fillStyle = `hsla(${hue}, 80%, 64%, 0.95)`;
        context.arc(node.x, node.y, 3 + node.scale * 5, 0, Math.PI * 2);
        context.fill();
      });
      const first = layer.projected[0];
      if (first) {
        context.fillStyle = "rgba(230,240,248,0.96)";
        context.font = "600 12px Georgia";
        context.fillText(layer.name, first.x - 18, Math.max(18, first.y - 24));
      }
    });

    context.fillStyle = "rgba(255,255,255,0.86)";
    context.font = "600 14px Georgia";
    context.fillText("Artificial neural network chamber", 20, 28);
    context.fillStyle = "rgba(205,214,230,0.9)";
    context.font = "12px Georgia";
    context.fillText(`Fusion ${payload.architecture?.fusion_hidden_dim ?? "n/a"}  |  Dropout ${formatMetric(payload.architecture?.dropout, 2)}`, 20, 48);
  }

  requestAnimationFrame(draw);
}

function renderData(payload) {
  showViewer();
  destroyCharts();

  renderMetrics(payload);
  renderCautions(payload);
  renderArchitecture(payload);
  renderProbabilityCharts(payload);
  renderCalibrationChart(payload);
  renderComparisonChart(payload);
  renderTrainingCharts(payload);
  renderUncertaintyScatter(payload);
  renderHeatmap("raw-heatmap", payload.chart_data?.confusion_matrices?.raw, payload.label_names, (alpha) => `rgba(184, 92, 56, ${alpha})`);
  renderHeatmap("calibrated-heatmap", payload.chart_data?.confusion_matrices?.calibrated, payload.label_names, (alpha) => `rgba(20, 83, 116, ${alpha})`);
  renderHeatmap("mc-heatmap", payload.chart_data?.confusion_matrices?.mc_dropout, payload.label_names, (alpha) => `rgba(79, 119, 45, ${alpha})`);
  renderUncertaintyTable(payload);
  startNeuroScene(payload);
  startNetworkScene(payload);
}

async function bootstrap() {
  bindLayoutControls();
  try {
    const response = await fetch("./latest_inference.json", { cache: "no-store" });
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }
    const payload = await response.json();
    renderData(payload);
  } catch (error) {
    renderEmptyState(`Could not load local payload: ${error.message}`);
  }
}

bootstrap();
