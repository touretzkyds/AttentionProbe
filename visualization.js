(function () {
  // 1. Inject CSS Styles into <head>
  const style = document.createElement('style');
  style.textContent = `
    :root{
      --bg:#080b11;
      --panel-border:#8b5cf6;
      --dim:#5b6b82;
      --text:#c9d6e8;
      --mono: 'SFMono-Regular', Consolas, 'Courier New', monospace;
      --cyan:#22d3ee;
      --violet:#8b5cf6;
      --orange:#f59e0b;
      --blue:#3b82f6;
      --green:#22c55e;
      --red:#ef4444;
      --magenta:#d946ef;
    }
    *{box-sizing:border-box;}
    html,body{
      margin:0; padding:0; background:var(--bg); color:var(--text);
      font-family:var(--mono);
      display:flex; align-items:center; justify-content:center;
      min-height:100vh;
    }
    .stage{
      position:relative;
      width:1040px; height:760px;
      background:radial-gradient(ellipse at 50% 0%, #0d1220 0%, #080b11 60%);
      overflow:hidden;
      border-radius:6px;
      user-select:none;
    }
    .mode-switch{
      position:absolute; top:34px; left:0; right:0;
      display:flex; justify-content:center; gap:0;
    }
    .mode-btn{
      font-family:var(--mono); font-size:9px; letter-spacing:1.5px;
      color:#8fa3c0; background:#0a0f18; border:1px solid #2a3550;
      padding:6px 12px; cursor:pointer; transition:all .15s ease;
    }
    .mode-btn:first-child{ border-radius:3px 0 0 3px; }
    .mode-btn:last-child{ border-radius:0 3px 3px 0; }
    .mode-btn + .mode-btn{ border-left:none; }
    .mode-btn:hover{ color:#dce6f4; border-color:#5b6b82; }
    .mode-btn.active{
      background:var(--violet); border-color:var(--violet); color:#fff;
    }
    .title{
      position:absolute; top:16px; left:0; right:0;
      text-align:center;
      font-size:10px; letter-spacing:5px; color:#3b4a63;
    }
    .footer{
      position:absolute; bottom:10px; left:0; right:0; text-align:center;
      font-size:9px; letter-spacing:3px; color:#33415a;
    }
    .headCarousel{
      position:absolute; z-index:3;
      display:flex; align-items:center; gap:4px;
      outline:none; border-radius:4px;
    }
    .headCarousel.tl{ top:8px; left:8px; }
    .headCarousel.br{ bottom:40px; right:8px; }
    .headCarousel:focus-visible .hc-matrix{
      box-shadow:0 0 0 3px var(--hc-color, #5b6b82);
    }
    .hc-arrow{
      flex:none; width:18px; height:20px;
      display:flex; align-items:center; justify-content:center;
      font-family:var(--mono); font-size:12px; line-height:1;
      color:#8fa3c0; background:#0a0f18; border:1px solid #2a3550; border-radius:3px;
      cursor:pointer; padding:0;
      transition:border-color .2s ease, color .2s ease;
    }
    .hc-arrow:hover{ border-color:var(--hc-color, #5b6b82); color:#fff; }
    .hc-arrow:focus-visible{ outline:2px solid var(--hc-color, #5b6b82); outline-offset:1px; }
    .hc-matrix{
      width:118px; padding:6px;
      border:2px solid var(--hc-color, #1c2436); border-radius:3px; background:#0a0e17;
      transition:border-color .3s ease, box-shadow .3s ease;
    }
    .cornerMat-title{
      font-size:7px; letter-spacing:1.5px; color:#3b4a63;
      margin-bottom:2px; text-align:center; white-space:nowrap;
    }
    .hc-header{
      font-size:9px; font-weight:600; letter-spacing:1px;
      text-align:center; margin-bottom:4px;
      color:var(--hc-color, #c9d6e8);
      transition:color .3s ease;
      white-space:nowrap; overflow:hidden; text-overflow:ellipsis;
    }
    .hc-subheader{
      font-size:7px; letter-spacing:1px;
      text-align:center; margin-bottom:4px;
      color:#5b6b82;
      white-space:nowrap;
    }
    .cornerMat-grid{ display:grid; gap:1px; }
    .cornerMat-cell{ width:100%; aspect-ratio:1; border-radius:1px; }
    @media (prefers-reduced-motion: reduce){
      .hc-matrix, .hc-header{ transition:none; }
    }
    .panel{
      position:absolute; left:301px; top:172px; width:438px; height:401px;
      border:none; border-radius:2px;
    }
    .token-row{
      position:absolute; left:0; right:0; display:flex; justify-content:space-between;
      padding:0 8px;
    }
    .token-row.top{ top:-13px; }
    .token-row.bottom{ bottom:-13px; }
    .token-btn{
      font-family:var(--mono); font-size:11px; color:#8fa3c0;
      background:#0a0f18; border:1px solid #2a3550; border-radius:3px;
      padding:4px 9px; cursor:pointer; letter-spacing:.5px;
      transition:all .15s ease;
    }
    .token-btn:hover{ border-color:#5b6b82; color:#dce6f4; }
    .token-btn.active{
      background:var(--magenta); border-color:var(--magenta); color:#fff;
      box-shadow:0 0 14px 0 rgba(217,70,239,.6);
    }
    .token-btn.antecedent-marker{ border-bottom:2px solid var(--blue); }
    .names{ position:absolute; width:170px; top:60px; }
    .names.left{ left:-195px; text-align:left; }
    .names.right{ right:-218px; text-align:right; }
    .name-row{
      position:absolute; display:flex; align-items:center; gap:10px; width:170px;
      cursor:pointer; transition:all .2s ease;
      background:transparent;
      border:none;
      box-shadow:none;
      padding:0;
    }
    .names.right .name-row{ flex-direction:row-reverse; }
    .name-row.dim{ opacity:.35; }
    .name-row.active{
      opacity:1;
      transform:scale(1.05);
    }
    .dot{
      width:26px; height:26px; border-radius:50%; flex:none;
      border:2px solid; background:#0a0f18;
      display:flex; align-items:center; justify-content:center;
      font-size:9px; font-weight:bold;
      transition:all .2s ease;
    }
    .name-row.active .dot{
      box-shadow:0 0 10px currentColor;
    }
    .name-meta{ line-height:1.25; }
    .name-meta .nm{ font-size:11px; font-weight:600; letter-spacing:.3px; }
    .name-meta .sc{ font-size:9px; opacity:.65; }
    svg.arcs{ position:absolute; inset:0; width:100%; height:100%; overflow:visible; pointer-events:none; }
    
    /* Allow interaction and line hover highlighting on path arcs */
    svg.arcs path{
      pointer-events: stroke;
      cursor: pointer;
      transition: stroke-width 0.15s ease, opacity 0.15s ease, filter 0.15s ease;
    }
    svg.arcs path:hover{
      opacity: 1 !important;
      stroke-width: calc(var(--base-stroke-width, 2px) + 2.5px) !important;
      filter: drop-shadow(0 0 4px currentColor);
    }

    .tag-layer{ position:absolute; inset:0; pointer-events:none; }
    .tag{
      position:absolute; transform:translateX(-50%);
      display:inline-block; font-size:9px; letter-spacing:.5px; padding:2px 7px;
      border:1px solid; border-radius:3px; opacity:.55; white-space:nowrap;
      transition:opacity .2s ease;
    }
    .tag.active{ opacity:1; }
    .tag.antecedent{ color:var(--blue); border-color:var(--blue); }
    .tag.anaphor{ color:var(--orange); border-color:var(--orange); }
    .tag.headnoun{ color:var(--violet); border-color:var(--violet); }
    .tag.determiner{ color:var(--green); border-color:var(--green); }
    .input-bar{
      position:absolute; bottom:46px; left:0; right:0;
      display:flex; justify-content:center; align-items:center; gap:8px;
    }
    .input-bar input[type=text]{
      font-family:var(--mono); font-size:12px; color:#dce6f4;
      background:#0a0f18; border:1px solid #2a3550; border-radius:3px;
      padding:7px 10px; width:400px; letter-spacing:.3px;
    }
    .input-bar input[type=text]:focus{ outline:none; border-color:var(--violet); }
    .input-bar input[type=text]::placeholder{ color:#3b4a63; }
    .input-bar button{
      font-family:var(--mono); font-size:10px; letter-spacing:1.5px;
      color:#c9d6e8; background:#0f1424; border:1px solid var(--violet);
      border-radius:3px; padding:8px 14px; cursor:pointer; transition:all .15s ease;
    }
    .input-bar button:hover{ background:var(--violet); color:#fff; }
  `;
  document.head.appendChild(style);

  // 2. Build HTML Structure
  const stage = document.createElement('div');
  stage.className = 'stage';
  stage.innerHTML = `
    <svg id="sideArcs" style="position:absolute;inset:0;width:100%;height:100%;pointer-events:none;"></svg>
    <div class="title">TRANSFORMER · MULTI-HEAD ATTENTION</div>

    <div class="mode-switch" id="modeSwitch">
      <button class="mode-btn" type="button" data-mode="left">BOTTOM → LEFT</button>
      <button class="mode-btn" type="button" data-mode="right">TOP → RIGHT</button>
      <button class="mode-btn active" type="button" data-mode="both">BOTH</button>
    </div>

    <div class="headCarousel tl" id="carouselTL" tabindex="0" role="group" aria-roledescription="carousel" aria-label="Attention head selector">
      <button class="hc-arrow hc-prev" type="button" aria-label="Previous head">‹</button>
      <div class="hc-matrix" id="matTL">
        <div class="hc-header" id="matTL-header" aria-live="polite"></div>
        <div class="hc-subheader" id="matTL-subheader"></div>
        <div class="cornerMat-grid" id="matGridTL"></div>
      </div>
      <button class="hc-arrow hc-next" type="button" aria-label="Next head">›</button>
    </div>
    <div class="headCarousel br" id="carouselBR" tabindex="0" role="group" aria-roledescription="carousel" aria-label="Attention head selector">
      <button class="hc-arrow hc-prev" type="button" aria-label="Previous head">‹</button>
      <div class="hc-matrix" id="matBR">
        <div class="hc-header" id="matBR-header" aria-live="polite"></div>
        <div class="hc-subheader" id="matBR-subheader"></div>
        <div class="cornerMat-grid" id="matGridBR"></div>
      </div>
      <button class="hc-arrow hc-next" type="button" aria-label="Next head">›</button>
    </div>

    <div class="panel" id="panel">
      <div class="token-row top" id="topRow"></div>
      <div class="names left" id="namesLeft"></div>
      <div class="names right" id="namesRight"></div>
      <svg class="arcs" id="arcSvg"></svg>
      <div class="token-row bottom" id="bottomRow"></div>
      <div class="tag-layer" id="tagLayer"></div>
    </div>

    <div class="input-bar">
      <input type="text" id="sentenceInput" placeholder="Type a sentence…" value="The man saw his jacket ." />
      <button id="sentenceBtn">VISUALIZE</button>
    </div>
    <div class="footer">CLICK A TOKEN · CLICK A NAME OR USE THE ARROWS TO CHANGE HEAD</div>
  `;
  document.body.appendChild(stage);

  document.addEventListener('click', (event) => {
    const button = event.target.closest('.mode-btn');
  
    if (!button) return;
  
    setViewMode(button.dataset.mode);
  });

  // 3. Named heads
  const NAMED_HEADS = [
    { name: "Alphonse",  layer: 0, head: 15 },
    { name: "Beatrix",   layer: 2, head: 6 },
    { name: "Charlene",  layer: 2, head: 8 },
    { name: "Diego",     layer: 2, head: 9 },
    { name: "Ephraim",     layer: 3, head: 6 },
    { name: "Francesca", layer: 3, head: 9 }
  ];
  const MAX_MATRICES = NAMED_HEADS.length;

  function getMatrixName(idx)  { return NAMED_HEADS[idx % MAX_MATRICES].name; }
  function getMatrixLayer(idx) { return NAMED_HEADS[idx % MAX_MATRICES].layer; }
  function getMatrixHead(idx)  { return NAMED_HEADS[idx % MAX_MATRICES].head; }

  // 4. Application State
  let tokens = [];
  let attentionData = null;
  let numLayers = 0;
  let numHeads = 0;
  let selectedTopIdx = 0;
  let selectedBottomIdx = 0;
  let selectedMatrixIdx = 0;
  let viewMode = 'both';

  const HEAD_COLORS = [
    "#ef4444", "#22c55e", "#3b82f6", "#f59e0b",
    "#8b5cf6", "#14b8a6", "#d946ef", "#22d3ee"
  ];

  function getMatrixColor(idx) {
    return HEAD_COLORS[idx % HEAD_COLORS.length];
  }

  const MIN_WEIGHT_TO_DRAW = 0.02;
  const MIN_STROKE_WIDTH = 0.75;
  const MAX_STROKE_WIDTH = 5.5;

  function weightToStrokeWidth(weight) {
    const clamped = Math.max(0, Math.min(1, weight));
    return MIN_STROKE_WIDTH + clamped * (MAX_STROKE_WIDTH - MIN_STROKE_WIDTH);
  }

  const panelW = 438, panelH = 401, panelX = 301, panelY = 172;
  const pad = 8;
  const usableW = panelW - pad * 2;
  const tokenXtop = [];

  function computeTokenX() {
    tokenXtop.length = 0;
    if (tokens.length <= 1) {
      tokenXtop.push(panelX + panelW / 2);
      return;
    }
    tokens.forEach((t, i) => {
      tokenXtop[i] = panelX + pad + (usableW / (tokens.length - 1)) * i;
    });
  }

  const topY = panelY - 13 + 10;
  const bottomY = panelY + panelH - 13 + 10;
  const nameYtop = panelY + 60;
  const nameRowGap = 43;
  const leftX = panelX - 195 + 13;
  const rightX = panelX + panelW + 218 - 13;

  const topRow = document.getElementById('topRow');
  const bottomRow = document.getElementById('bottomRow');

  function buildRows() {
    topRow.innerHTML = "";
    bottomRow.innerHTML = "";
  
    const lower = tokens.map(t => t.toLowerCase());
  
    tokens.forEach((t, i) => {
      const isTopActive = i === selectedTopIdx;
      const isBottomActive = i === selectedBottomIdx;
      const isAntecedent = lower[i] === 'man';
  
      // TOP ROW
      const bTop = document.createElement('div');
  
      bTop.className =
        'token-btn' +
        (isTopActive ? ' active' : '') +
        (isAntecedent ? ' antecedent-marker' : '');
  
      bTop.textContent = displayToken(t);
      bTop.onclick = () => selectTop(i);
  
      topRow.appendChild(bTop);
  
      // BOTTOM ROW
      const bBot = document.createElement('div');
  
      bBot.className =
        'token-btn' +
        (isBottomActive ? ' active' : '') +
        (isAntecedent ? ' antecedent-marker' : '');
  
      bBot.textContent = displayToken(t);
      bBot.onclick = () => selectBottom(i);
  
      bottomRow.appendChild(bBot);
    });
  }

  const namesLeft = document.getElementById('namesLeft');
  const namesRight = document.getElementById('namesRight');

  function buildNames() {
    namesLeft.innerHTML = "";
    namesRight.innerHTML = "";
    if (!numHeads) return;

    NAMED_HEADS.forEach((nh, idx) => {
      const color = getMatrixColor(idx);
      const isActive = idx === selectedMatrixIdx;
      const y = idx * nameRowGap;

      const rowL = document.createElement('div');
      rowL.className = 'name-row' + (isActive ? ' active' : ' dim');
      rowL.style.top = y + 'px';
      rowL.onclick = () => selectMatrix(idx);
      rowL.innerHTML = `<div class="dot" style="border-color:${color};color:${color}">H${nh.head}</div>
        <div class="name-meta"><div class="nm" style="color:${color}">${nh.name}</div><div class="sc" style="color:${color}">L${nh.layer} · Head ${nh.head}</div></div>`;
      namesLeft.appendChild(rowL);

      const rowR = document.createElement('div');
      rowR.className = 'name-row' + (isActive ? ' active' : ' dim');
      rowR.style.top = y + 'px';
      rowR.onclick = () => selectMatrix(idx);
      rowR.innerHTML = `<div class="dot" style="border-color:${color};color:${color}">H${nh.head}</div>
        <div class="name-meta"><div class="nm" style="color:${color}">${nh.name}</div><div class="sc" style="color:${color}">L${nh.layer} · Head ${nh.head}</div></div>`;
      namesRight.appendChild(rowR);
    });
  }

  const svg = document.getElementById('arcSvg');

  function drawArcs() {
    svg.innerHTML = "";
  
    if (!attentionData || !numHeads) return;
  
    for (let i = 0; i < MAX_MATRICES; i++) {
  
      const layer = getMatrixLayer(i);
      const head = getMatrixHead(i);
      const color = getMatrixColor(i);
  
      const matrix = attentionData[layer]
        ? attentionData[layer][head]
        : null;
  
      if (!matrix) continue;
  
      // Bottom token = query
      // Top token = key
      const weight =
        matrix[selectedBottomIdx]?.[selectedTopIdx] || 0;
  
      if (weight <= MIN_WEIGHT_TO_DRAW) continue;
  
      const strokeWidth = weightToStrokeWidth(weight);
  
      const y1 = nameYtop + i * nameRowGap - panelY;
  
      // LEFT-SIDE ARC: head → selected bottom token
      if (viewMode !== 'right') {
  
        const txBottom = tokenXtop[selectedBottomIdx] - panelX;
        const ty = bottomY - panelY;
        const dotR = 13;
        const x1edge = leftX - panelX + dotR;
  
        const pathL = document.createElementNS(
          'http://www.w3.org/2000/svg',
          'path'
        );
  
        const dL =
          `M ${x1edge} ${y1} C ${x1edge + 150} ${y1}, ` +
          `${txBottom} ${y1 + (ty - y1) * 0.15}, ` +
          `${txBottom} ${ty}`;
  
        pathL.setAttribute('d', dL);
        pathL.setAttribute('fill', 'none');
        pathL.setAttribute('stroke', color);
        pathL.setAttribute('stroke-width', strokeWidth);
        pathL.style.setProperty('--base-stroke-width', `${strokeWidth}px`);
        pathL.setAttribute('opacity', '0.85');
        pathL.setAttribute('stroke-linecap', 'round');
  
        svg.appendChild(pathL);
      }
  
      // RIGHT-SIDE ARC: head → selected top token
      if (viewMode !== 'left') {
  
        const txTop = tokenXtop[selectedTopIdx] - panelX;
        const tyTop = topY - panelY;
        const dotR = 13;
        const x2edge = rightX - panelX - dotR;
  
        const pathR = document.createElementNS(
          'http://www.w3.org/2000/svg',
          'path'
        );
  
        const dR =
          `M ${x2edge} ${y1} C ${x2edge - 150} ${y1}, ` +
          `${txTop} ${y1 + (tyTop - y1) * 0.15}, ` +
          `${txTop} ${tyTop}`;
  
        pathR.setAttribute('d', dR);
        pathR.setAttribute('fill', 'none');
        pathR.setAttribute('stroke', color);
        pathR.setAttribute('stroke-width', strokeWidth);
        pathR.style.setProperty('--base-stroke-width', `${strokeWidth}px`);
        pathR.setAttribute('opacity', '0.85');
        pathR.setAttribute('stroke-linecap', 'round');
  
        svg.appendChild(pathR);
      }
    }
  }

  const tagLayer = document.getElementById('tagLayer');

  function buildTags() {
    tagLayer.innerHTML = "";
    if (tokens.length === 0) return;

    const tagsY = bottomY + 78;
    const lower = tokens.map(t => t.toLowerCase());
    const manIdx = lower.indexOf('man');
    const hisIdx = lower.indexOf('his');
    const jacketIdx = lower.indexOf('jacket');

    const defs = [];
    if (manIdx >= 0) defs.push({ id: 'man', label: 'antecedent', cls: 'antecedent', tok: manIdx, y: tagsY });
    if (jacketIdx >= 0) defs.push({ id: 'jacket', label: 'head noun', cls: 'headnoun', tok: jacketIdx, y: tagsY });
    if (hisIdx >= 0) {
      defs.push({ id: 'his1', label: 'anaphor', cls: 'anaphor', tok: hisIdx, y: tagsY });
      defs.push({ id: 'his2', label: 'determiner', cls: 'determiner', tok: hisIdx, y: tagsY + 20 });
    }

    defs.forEach(d => {
      const el = document.createElement('div');
      el.className = 'tag ' + d.cls;
      el.id = 'tag-' + d.id;
      el.textContent = d.label;
      el.style.left = (tokenXtop[d.tok] - panelX) + 'px';
      el.style.top = (d.y - panelY) + 'px';
      tagLayer.appendChild(el);
    });
  }

  function updateTags() {
    const manEl = document.getElementById('tag-man');
    const his1 = document.getElementById('tag-his1');
    const his2 = document.getElementById('tag-his2');
    const jacketEl = document.getElementById('tag-jacket');
    const cur = (tokens[selectedBottomIdx] || '').toLowerCase();
    if (manEl) manEl.classList.toggle('active', cur === 'man');
    if (his1) his1.classList.toggle('active', cur === 'his');
    if (his2) his2.classList.toggle('active', cur === 'his');
    if (jacketEl) jacketEl.classList.toggle('active', cur === 'jacket');
  }

  const sideSvg = document.getElementById('sideArcs');

  function arrowPoly(tip, dir, size) {
    const s = size || 6;
    let pts;
    if (dir === 'up') {
      pts = [[tip[0] - s * 0.7, tip[1] + s], [tip[0] + s * 0.7, tip[1] + s], [tip[0], tip[1]]];
    } else if (dir === 'down') {
      pts = [[tip[0] - s * 0.7, tip[1] - s], [tip[0] + s * 0.7, tip[1] - s], [tip[0], tip[1]]];
    } else if (dir === 'left') {
      pts = [[tip[0] + s, tip[1] - s * 0.7], [tip[0] + s, tip[1] + s * 0.7], [tip[0], tip[1]]];
    } else {
      pts = [[tip[0] - s, tip[1] - s * 0.7], [tip[0] - s, tip[1] + s * 0.7], [tip[0], tip[1]]];
    }
    const poly = document.createElementNS('http://www.w3.org/2000/svg', 'polygon');
    poly.setAttribute('points', pts.map(p => p.join(',')).join(' '));
    return poly;
  }

  function drawSideBundles() {
    sideSvg.innerHTML = "";
    if (!numHeads) return;

    const stageRect = document.querySelector('.stage').getBoundingClientRect();

    const activeTopBtn = document.querySelector('.token-row.top .token-btn.active');
    const activeBotBtn = document.querySelector('.token-row.bottom .token-btn.active');
    if (!activeTopBtn || !activeBotBtn) return;

    const topRect = activeTopBtn.getBoundingClientRect();
    const topX = topRect.left - stageRect.left + topRect.width / 2;
    const topYPix = topRect.top - stageRect.top;

    const botRect = activeBotBtn.getBoundingClientRect();
    const botX = botRect.left - stageRect.left + botRect.width / 2;
    const botBottom = botRect.bottom - stageRect.top;

    const midY = nameYtop + 2.5 * nameRowGap;

    // --- LEFT SIDE BUNDLES ---
    const leftBaseX = leftX - 55;
    const lFanX = 6, lFanY = 6;
    const runYtop = topYPix - 35;

    for (let i = 0; viewMode !== 'right' && i < MAX_MATRICES; i++) {
      const color = getMatrixColor(i);
      const cy = nameYtop + i * nameRowGap;
      const offsetX = (i - 2.5) * lFanX;
      const fpX = leftBaseX + offsetX;

      const spread = Math.min(topRect.width * 0.7, lFanX * 5 * 2);
      const arrowX = topX - spread / 2 + i * (spread / 5);

      const line = document.createElementNS('http://www.w3.org/2000/svg', 'path');
      line.setAttribute('d', `M ${leftX - 13} ${cy} C ${leftX - 38} ${cy}, ${fpX + 18} ${midY}, ${fpX} ${midY}`);
      line.setAttribute('stroke', color);
      line.setAttribute('stroke-width', '2');
      line.setAttribute('fill', 'none');
      line.setAttribute('opacity', '0.6');
      sideSvg.appendChild(line);

      const vert = document.createElementNS('http://www.w3.org/2000/svg', 'path');
      vert.setAttribute('d', `M ${fpX} ${midY} L ${fpX} ${runYtop}`);
      vert.setAttribute('stroke', color);
      vert.setAttribute('stroke-width', '2');
      vert.setAttribute('fill', 'none');
      vert.setAttribute('opacity', '0.85');
      sideSvg.appendChild(vert);

      const yh = runYtop - (lFanY * 5) / 2 + i * lFanY;
      const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
      path.setAttribute('d', `M ${fpX} ${runYtop} L ${fpX} ${yh} L ${arrowX} ${yh} L ${arrowX} ${topYPix - 8}`);
      path.setAttribute('stroke', color);
      path.setAttribute('stroke-width', '2');
      path.setAttribute('fill', 'none');
      path.setAttribute('opacity', '0.9');
      sideSvg.appendChild(path);

      const arrowBtn = arrowPoly([arrowX, topYPix - 6], 'down');
      arrowBtn.setAttribute('fill', color);
      sideSvg.appendChild(arrowBtn);
    }

    // --- RIGHT SIDE BUNDLES ---
    const rightBaseX = rightX + 55;
    const fanX = 6, fanY = 6;
    const trunkBotY = botBottom + 40;

    for (let i = 0; viewMode !== 'left' && i < MAX_MATRICES; i++) {
      const color = getMatrixColor(i);
      const cy = nameYtop + i * nameRowGap;
      const offsetX = (i - 2.5) * fanX;
      const fpX = rightBaseX + offsetX;

      const spread = Math.min(botRect.width * 0.7, fanX * 5 * 2);
      const arrowX = botX - spread / 2 + i * (spread / 5);

      const line = document.createElementNS('http://www.w3.org/2000/svg', 'path');
      line.setAttribute('d', `M ${rightX + 13} ${cy} C ${rightX + 38} ${cy}, ${fpX - 18} ${midY}, ${fpX} ${midY}`);
      line.setAttribute('stroke', color);
      line.setAttribute('stroke-width', '2');
      line.setAttribute('fill', 'none');
      line.setAttribute('opacity', '0.6');
      sideSvg.appendChild(line);

      const vert = document.createElementNS('http://www.w3.org/2000/svg', 'path');
      vert.setAttribute('d', `M ${fpX} ${midY} L ${fpX} ${trunkBotY}`);
      vert.setAttribute('stroke', color);
      vert.setAttribute('stroke-width', '2');
      vert.setAttribute('fill', 'none');
      vert.setAttribute('opacity', '0.85');
      sideSvg.appendChild(vert);

      const yh = trunkBotY - (fanY * 5) / 2 + i * fanY;
      const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
      path.setAttribute('d', `M ${fpX} ${trunkBotY} L ${fpX} ${yh} L ${arrowX} ${yh} L ${arrowX} ${botBottom + 8}`);
      path.setAttribute('stroke', color);
      path.setAttribute('stroke-width', '2');
      path.setAttribute('fill', 'none');
      path.setAttribute('opacity', '0.9');
      sideSvg.appendChild(path);

      const arrowMatrix = arrowPoly([rightX + 13, cy], 'left');
      arrowMatrix.setAttribute('fill', color);
      sideSvg.appendChild(arrowMatrix);
    }
  }

  const viridisStops = [
    [0.00, [68, 1, 84]], [0.14, [71, 44, 122]], [0.29, [59, 81, 139]],
    [0.43, [44, 113, 142]], [0.57, [33, 144, 141]], [0.71, [39, 173, 129]],
    [0.86, [92, 200, 99]], [1.00, [253, 231, 37]]
  ];

  function viridis(t) {
    t = Math.max(0, Math.min(1, t));
    for (let i = 0; i < viridisStops.length - 1; i++) {
      const [t0, c0] = viridisStops[i], [t1, c1] = viridisStops[i + 1];
      if (t >= t0 && t <= t1) {
        const f = (t - t0) / (t1 - t0);
        const r = Math.round(c0[0] + (c1[0] - c0[0]) * f);
        const g = Math.round(c0[1] + (c1[1] - c0[1]) * f);
        const b = Math.round(c0[2] + (c1[2] - c0[2]) * f);
        return `rgb(${r},${g},${b})`;
      }
    }
    return `rgb(253,231,37)`;
  }

  function renderCornerMatrix(gridId, layer, head) {
    const grid = document.getElementById(gridId);
    if (!grid || !attentionData) return;

    const matrix = (attentionData[layer] && attentionData[layer][head])
      ? attentionData[layer][head]
      : [];

    const n = tokens.length;
    grid.style.gridTemplateColumns = `repeat(${n}, 1fr)`;
    grid.innerHTML = "";

    let minVal = Infinity;
    let maxVal = -Infinity;
    matrix.forEach(row => {
      row.forEach(value => {
        if (value < minVal) minVal = value;
        if (value > maxVal) maxVal = value;
      });
    });
    const range = (maxVal - minVal) || 1;

    matrix.forEach(row => {
      row.forEach(value => {
        const cell = document.createElement("div");
        cell.className = "cornerMat-cell";
        const t = (value - minVal) / range;
        cell.style.background = viridis(t);
        cell.title = value.toFixed(4);
        grid.appendChild(cell);
      });
    });
  }

  function renderMatrices() {
    if (!attentionData) return;

    const layer = getMatrixLayer(selectedMatrixIdx);
    const head = getMatrixHead(selectedMatrixIdx);
    const color = getMatrixColor(selectedMatrixIdx);
    const customName = getMatrixName(selectedMatrixIdx);

    ['TL', 'BR'].forEach(key => {
      const root = document.getElementById('carousel' + key);
      const matrixEl = document.getElementById('mat' + key);
      const headerEl = document.getElementById('mat' + key + '-header');
      const subheaderEl = document.getElementById('mat' + key + '-subheader');

      matrixEl.style.setProperty('--hc-color', color);
      headerEl.textContent = customName;
      subheaderEl.textContent = `L${layer} · Head ${head}`;
      root.setAttribute('aria-label', `Layer ${layer}, Head ${head}, ${customName}`);

      renderCornerMatrix('matGrid' + key, layer, head);
    });

    buildNames();
    drawArcs();
    drawSideBundles();
  }

  function selectMatrix(idx) {
    selectedMatrixIdx = ((idx % MAX_MATRICES) + MAX_MATRICES) % MAX_MATRICES;
    renderMatrices();
  }

  function stepMatrix(delta) {
    selectMatrix(selectedMatrixIdx + delta);
  }

  function setViewMode(mode) {
    if (!['left', 'right', 'both'].includes(mode)) return;
  
    viewMode = mode;
  
    document.querySelectorAll('.mode-btn').forEach(button => {
      button.classList.toggle('active', button.dataset.mode === mode);
    });
  
    drawArcs();
    drawSideBundles();
  }

  function initCarousels() {
    ['TL', 'BR'].forEach(key => {
      const root = document.getElementById('carousel' + key);
      root.querySelectorAll('.hc-arrow').forEach(btn => {
        const delta = btn.classList.contains('hc-prev') ? -1 : 1;
        btn.addEventListener('click', () => {
          stepMatrix(delta);
          root.focus();
        });
      });

      root.addEventListener('keydown', (e) => {
        if (e.key === 'ArrowLeft') { stepMatrix(-1); e.preventDefault(); }
        else if (e.key === 'ArrowRight') { stepMatrix(1); e.preventDefault(); }
      });
    });
  }

  function selectTop(idx) {
    selectedTopIdx = idx;
  
    buildRows();
    drawArcs();
    updateTags();
    drawSideBundles();
  }
  
  function selectBottom(idx) {
    selectedBottomIdx = idx;
  
    buildRows();
    drawArcs();
    updateTags();
    drawSideBundles();
  }

  function cleanT5Token(token) {
    return token.replace(/^▁/, "").replace(/^Ġ/, "");
  }

  function displayToken(token) {
    if (token === "</s>") return "</s>";
    const cleaned = cleanT5Token(token);
    return cleaned || token;
  }

  async function runSentence(raw, preferWord) {
    if (!raw.trim()) return;

    const button = document.getElementById("sentenceBtn");

    try {
      button.disabled = true;
      button.textContent = "LOADING...";

      const response = await fetch("http://localhost:8000/attention", {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({
          sentence: raw
        })
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`Server returned status ${response.status}`);
      }

      const data = await response.json();

      tokens = data.tokens;
      attentionData = data.attentions;
      numLayers = data.num_layers;
      numHeads = data.num_heads;

      computeTokenX();

      let idx = -1;

      if (preferWord) {
        idx = tokens.findIndex(
          t => cleanT5Token(t).toLowerCase() === preferWord.toLowerCase()
        );
      }

      if (idx < 0) {
        idx = tokens.findIndex(t => t !== "</s>");
      }

      if (idx < 0) {
        idx = 0;
      }

      selectedTopIdx = idx;
      selectedBottomIdx = idx;
      selectedMatrixIdx = 0;

      buildRows();
      buildTags();
      updateTags();
      renderMatrices();

    } catch (error) {
      console.error("API Fetch Error:", error);
    } finally {
      button.disabled = false;
      button.textContent = "VISUALIZE";
    }
  }

  document.getElementById('sentenceBtn').addEventListener('click', () => {
    const val = document.getElementById('sentenceInput').value;
    runSentence(val);
  });

  initCarousels();
  runSentence("The man saw his jacket .", "his");
})();