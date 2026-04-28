import React, { useState, useMemo, useEffect, useRef } from 'react';

// =================================================================
// ASCE 41 NONLINEAR HINGE PARAMETERS (Tier 2/3 — pushover)
// =================================================================
//
// Connection-type ductility ratios — how many multiples of yield rotation
// the hinge can sustain before each performance level is reached.
//   mu_LS = θ_LS / θy   (Life Safety plastic-rotation acceptance)
//   mu_CP = θ_CP / θy   (Collapse Prevention — total rotation capacity)
// θy itself is computed from the analysis (relative rotation at first hinge).
// Values approximate ASCE 41-17 Table 9-7.1 modeling parameters scaled to
// non-dimensional θy of typical W-section beams.
const ASCE41_CONNECTIONS = {
  'WUF-W (ductile FR)': { mu_LS: 8, mu_CP: 12, note: 'Welded unreinforced flange / bolted web — modern detailing' },
  'RBS (reduced beam)': { mu_LS: 12, mu_CP: 18, note: 'Reduced beam section ("dogbone") — most ductile' },
  'Pre-Northridge FR':  { mu_LS: 2.5, mu_CP: 4, note: 'Pre-1994 fully restrained — brittle weld-access-hole behavior' },
  'Generic ductile':    { mu_LS: 6, mu_CP: 10, note: "Laursen lecture default — μ_struct ≈ 10" },
};
const ASCE41_DEFAULT_KEY = 'Generic ductile';

// =================================================================
// LINEAR ALGEBRA
// =================================================================

const zeros1 = (n) => new Array(n).fill(0);
const zeros2 = (n, m) => Array.from({ length: n }, () => new Array(m).fill(0));

function gaussSolve(A_in, b_in) {
  const n = b_in.length;
  if (n === 0) return [];
  const M = A_in.map((r) => [...r]);
  const v = [...b_in];
  let maxDiag = 0;
  for (let i = 0; i < n; i++) maxDiag = Math.max(maxDiag, Math.abs(M[i][i]));
  const tol = Math.max(maxDiag * 1e-11, 1e-13);

  for (let i = 0; i < n; i++) {
    let pivotRow = i;
    let pivotVal = Math.abs(M[i][i]);
    for (let k = i + 1; k < n; k++) {
      if (Math.abs(M[k][i]) > pivotVal) {
        pivotVal = Math.abs(M[k][i]);
        pivotRow = k;
      }
    }
    if (pivotVal < tol) return null;
    if (pivotRow !== i) {
      [M[i], M[pivotRow]] = [M[pivotRow], M[i]];
      [v[i], v[pivotRow]] = [v[pivotRow], v[i]];
    }
    for (let k = i + 1; k < n; k++) {
      if (M[k][i] === 0) continue;
      const f = M[k][i] / M[i][i];
      for (let j = i; j < n; j++) M[k][j] -= f * M[i][j];
      v[k] -= f * v[i];
    }
  }

  const x = zeros1(n);
  for (let i = n - 1; i >= 0; i--) {
    let s = v[i];
    for (let j = i + 1; j < n; j++) s -= M[i][j] * x[j];
    x[i] = s / M[i][i];
  }
  return x;
}

// =================================================================
// FRAME ELEMENT STIFFNESS
// =================================================================

function localStiffness(L, EA, EI, hi, hj) {
  const k = zeros2(6, 6);
  const ka = EA / L;
  k[0][0] = k[3][3] = ka;
  k[0][3] = k[3][0] = -ka;

  if (!hi && !hj) {
    const c1 = (12 * EI) / (L * L * L);
    const c2 = (6 * EI) / (L * L);
    const c3 = (4 * EI) / L;
    const c4 = (2 * EI) / L;
    k[1][1] = c1; k[1][2] = c2; k[1][4] = -c1; k[1][5] = c2;
    k[2][1] = c2; k[2][2] = c3; k[2][4] = -c2; k[2][5] = c4;
    k[4][1] = -c1; k[4][2] = -c2; k[4][4] = c1; k[4][5] = -c2;
    k[5][1] = c2; k[5][2] = c4; k[5][4] = -c2; k[5][5] = c3;
  } else if (hi && !hj) {
    const c1 = (3 * EI) / (L * L * L);
    const c2 = (3 * EI) / (L * L);
    const c3 = (3 * EI) / L;
    k[1][1] = c1; k[1][4] = -c1; k[1][5] = c2;
    k[4][1] = -c1; k[4][4] = c1; k[4][5] = -c2;
    k[5][1] = c2; k[5][4] = -c2; k[5][5] = c3;
  } else if (!hi && hj) {
    const c1 = (3 * EI) / (L * L * L);
    const c2 = (3 * EI) / (L * L);
    const c3 = (3 * EI) / L;
    k[1][1] = c1; k[1][2] = c2; k[1][4] = -c1;
    k[2][1] = c2; k[2][2] = c3; k[2][4] = -c2;
    k[4][1] = -c1; k[4][2] = -c2; k[4][4] = c1;
  }
  return k;
}

function transformToGlobal(kL, c, s) {
  // K_global = T^T · k_local · T
  // T blocks: [[c,s,0],[-s,c,0],[0,0,1]] for each node (twice on diagonal)
  // Build T·k_local·T explicitly, exploiting block structure
  const T = zeros2(6, 6);
  T[0][0] = c; T[0][1] = s; T[1][0] = -s; T[1][1] = c; T[2][2] = 1;
  T[3][3] = c; T[3][4] = s; T[4][3] = -s; T[4][4] = c; T[5][5] = 1;
  // First compute kL · T (6x6)
  const kT = zeros2(6, 6);
  for (let i = 0; i < 6; i++)
    for (let j = 0; j < 6; j++) {
      let sum = 0;
      for (let p = 0; p < 6; p++) sum += kL[i][p] * T[p][j];
      kT[i][j] = sum;
    }
  // Then T^T · kT
  const Kg = zeros2(6, 6);
  for (let i = 0; i < 6; i++)
    for (let j = 0; j < 6; j++) {
      let sum = 0;
      for (let p = 0; p < 6; p++) sum += T[p][i] * kT[p][j];
      Kg[i][j] = sum;
    }
  return Kg;
}

// =================================================================
// PROBLEM DEFINITIONS
// =================================================================

function makeP1A() {
  const Mp = 1, EI = 1, EA = 1e4;
  const nodes = [
    { x: 0, y: 0, rest: [1, 1, 0], type: 'pin', label: 'Pin' },
    { x: 0.75, y: 0, rest: [0, 0, 0], type: 'beam-pt', label: 'P (0.75L)' },
    { x: 1.5, y: 0, rest: [0, 1, 0], type: 'roller', label: 'Roller (1.5L)' },
    { x: 2.0, y: 0, rest: [0, 0, 0], type: 'beam-pt', label: '2P (2.0L)' },
    { x: 2.5, y: 0, rest: [0, 1, 0], type: 'roller', label: 'Roller (2.5L)' },
  ];
  const elements = [
    { i: 0, j: 1, EA, EI, Mp, type: 'beam', label: 'L1' },
    { i: 1, j: 2, EA, EI, Mp, type: 'beam', label: 'L2' },
    { i: 2, j: 3, EA, EI, Mp, type: 'beam', label: 'R1' },
    { i: 3, j: 4, EA, EI, Mp, type: 'beam', label: 'R2' },
  ];
  const loadCases = [
    {
      name: 'Combined (P + 2P)',
      loads: [
        { node: 1, fx: 0, fy: -1, m: 0 },
        { node: 3, fx: 0, fy: -2, m: 0 },
      ],
    },
  ];
  return {
    name: 'P1A',
    title: 'Two-Span Continuous Beam',
    nodes, elements, loadCases,
    kinematicPu: 3.0,
    units: 'M_p / L',
    deflectScale: 2.5,
    bmdScale: 0.22,
    viewBox: { xMin: -0.4, xMax: 2.9, yMin: -0.85, yMax: 0.6 },
    multiCase: false,
    indeterminacyInit: 1,
    loadArrowScale: 0.5,
  };
}

function makeP1B() {
  const Mpb = 100, Mpc = 150;
  const EIb = 1, EIc = 1.5, EA = 5e4;
  const nodes = [
    { x: 0, y: 0, rest: [1, 1, 1], type: 'fixed', label: 'A' },
    { x: 0, y: 14, rest: [0, 0, 0], type: 'joint', label: 'B' },
    { x: 12, y: 14, rest: [0, 0, 0], type: 'beam-pt', label: 'E' },
    { x: 24, y: 14, rest: [0, 0, 0], type: 'joint', label: 'C' },
    { x: 24, y: -6, rest: [1, 1, 1], type: 'fixed', label: 'D' },
  ];
  const elements = [
    { i: 0, j: 1, EA, EI: EIc, Mp: Mpc, type: 'col', label: 'AB' },
    { i: 1, j: 2, EA, EI: EIb, Mp: Mpb, type: 'beam', label: 'BE' },
    { i: 2, j: 3, EA, EI: EIb, Mp: Mpb, type: 'beam', label: 'EC' },
    { i: 4, j: 3, EA, EI: EIc, Mp: Mpc, type: 'col', label: 'DC' },
  ];
  const loadCases = [
    { name: 'Horizontal P at B', loads: [{ node: 1, fx: 1, fy: 0, m: 0 }] },
    { name: 'Vertical 0.2P at E', loads: [{ node: 2, fx: 0, fy: -0.2, m: 0 }] },
  ];
  return {
    name: 'P1B',
    title: 'Asymmetric Portal Frame',
    nodes, elements, loadCases,
    kinematicPu: (17 / 56) * 100,
    units: 'k (Mp_beam = 100 k-ft)',
    deflectScale: 0.0003,
    bmdScale: 0.018,
    viewBox: { xMin: -5, xMax: 30, yMin: -10, yMax: 20 },
    multiCase: true,
    indeterminacyInit: 3,
    loadArrowScale: 5,
  };
}

function makeP2() {
  const Mp_outer = 2, Mp_middle = 3, Mp_beam = 1;
  const EI_outer = 2, EI_middle = 3, EI_beam = 1;
  const EA = 1e4;
  const L = 1;
  // Canonical M1+M2+M3 mechanism nodes: joints at 0, 2L, 4L; midspans at L, 3L.
  // (Coarser discretization here gives Pu_inc = 10·Mp/L exactly. Finer
  // discretization finds a slightly lower-energy variant with off-center beam
  // hinges; those tools may show ~9.88 instead. The tradeoff: this layout
  // matches the textbook M1+M2+M3 hinge pattern.)
  const nodes = [
    { x: 0, y: 0, rest: [1, 1, 1], type: 'fixed', label: 'LB' },                  // 0
    { x: 0, y: L, rest: [0, 0, 0], type: 'joint', label: 'LJ' },                  // 1
    { x: L, y: L, rest: [0, 0, 0], type: 'beam-pt', label: 'mid-L' },             // 2
    { x: 2 * L, y: L, rest: [0, 0, 0], type: 'joint', label: 'MJ' },              // 3
    { x: 2 * L, y: 0, rest: [1, 1, 1], type: 'fixed', label: 'MB' },              // 4
    { x: 3 * L, y: L, rest: [0, 0, 0], type: 'beam-pt', label: 'mid-R' },         // 5
    { x: 4 * L, y: L, rest: [0, 0, 0], type: 'joint', label: 'RJ' },              // 6
    { x: 4 * L, y: 0, rest: [1, 1, 1], type: 'fixed', label: 'RB' },              // 7
  ];
  const elements = [
    { i: 0, j: 1, EA, EI: EI_outer, Mp: Mp_outer, type: 'col', label: 'L-col' },
    { i: 4, j: 3, EA, EI: EI_middle, Mp: Mp_middle, type: 'col', label: 'M-col' },
    { i: 7, j: 6, EA, EI: EI_outer, Mp: Mp_outer, type: 'col', label: 'R-col' },
    { i: 1, j: 2, EA, EI: EI_beam, Mp: Mp_beam, type: 'beam', label: '' },
    { i: 2, j: 3, EA, EI: EI_beam, Mp: Mp_beam, type: 'beam', label: '' },
    { i: 3, j: 5, EA, EI: EI_beam, Mp: Mp_beam, type: 'beam', label: '' },
    { i: 5, j: 6, EA, EI: EI_beam, Mp: Mp_beam, type: 'beam', label: '' },
  ];
  const loadCases = [
    { name: 'Horizontal P at LJ', loads: [{ node: 1, fx: 1, fy: 0, m: 0 }] },
    { name: 'Vertical P/4 at mid-L', loads: [{ node: 2, fx: 0, fy: -0.25, m: 0 }] },
    {
      // UDL on right bay [2L, 4L], total = P/2; lumped to MJ, mid-R, RJ
      // tributary widths: MJ=L/2, mid-R=L, RJ=L/2 → loads P/8, P/4, P/8
      name: 'UDL on right bay (P/2 total)',
      loads: [
        { node: 3, fx: 0, fy: -1 / 8, m: 0 },
        { node: 5, fx: 0, fy: -1 / 4, m: 0 },
        { node: 6, fx: 0, fy: -1 / 8, m: 0 },
      ],
    },
  ];
  return {
    name: 'P2',
    title: 'Two-Bay Portal',
    nodes, elements, loadCases,
    kinematicPu: 10.0,
    units: 'M_p / L',
    deflectScale: 1.0,
    bmdScale: 0.14,
    viewBox: { xMin: -0.4, xMax: 4.5, yMin: -0.4, yMax: 1.7 },
    multiCase: true,
    indeterminacyInit: 6,
    loadArrowScale: 0.6,
    controlDof: 3,                         // node 1 (LJ) lateral DOF — pushover Δ
    pushoverEnabled: true,
    pushoverNote: 'Two-bay portal · lateral P at LJ · Δ measured at LJ',
  };
}

function makeP3() {
  const EA = 1e6;
  const Mp_roof = 225, Mp_3rd = 554, Mp_2nd = 933, Mp_col = 1479;
  const EI_roof = Mp_roof, EI_3rd = Mp_3rd, EI_2nd = Mp_2nd, EI_col = Mp_col;
  const heights = [0, 16, 30, 44];
  const xs = [0, 30, 60, 90];

  const nodes = [];
  for (let lvl = 0; lvl < 4; lvl++) {
    for (let col = 0; col < 4; col++) {
      const isFixed = lvl === 0;
      nodes.push({
        x: xs[col],
        y: heights[lvl],
        rest: isFixed ? [1, 1, 1] : [0, 0, 0],
        type: isFixed ? 'fixed' : 'joint',
        label:
          lvl === 0
            ? 'Base ' + (col + 1)
            : ['', '2nd', '3rd', 'Roof'][lvl] + ' ' + (col + 1),
      });
    }
  }
  const idx = (lvl, col) => lvl * 4 + col;
  const elements = [];
  for (let lvl = 0; lvl < 3; lvl++) {
    for (let col = 0; col < 4; col++) {
      elements.push({
        i: idx(lvl, col),
        j: idx(lvl + 1, col),
        EA, EI: EI_col, Mp: Mp_col,
        type: 'col',
        label: 'col' + (col + 1) + '-S' + (lvl + 1),
      });
    }
  }
  const beamMps = [Mp_2nd, Mp_3rd, Mp_roof];
  const beamEIs = [EI_2nd, EI_3rd, EI_roof];
  const beamLabels = ['2nd', '3rd', 'Roof'];
  for (let lvl = 1; lvl <= 3; lvl++) {
    for (let bay = 0; bay < 3; bay++) {
      elements.push({
        i: idx(lvl, bay),
        j: idx(lvl, bay + 1),
        EA, EI: beamEIs[lvl - 1], Mp: beamMps[lvl - 1],
        type: 'beam',
        label: beamLabels[lvl - 1] + '-bay' + (bay + 1),
      });
    }
  }
  const loadCases = [
    {
      name: 'Lateral V (0.49/0.33/0.18)',
      loads: [
        { node: idx(3, 0), fx: 0.49, fy: 0, m: 0 },
        { node: idx(2, 0), fx: 0.33, fy: 0, m: 0 },
        { node: idx(1, 0), fx: 0.18, fy: 0, m: 0 },
      ],
    },
  ];
  return {
    name: 'P3',
    title: 'Three-Story SMRF (SCWB)',
    nodes, elements, loadCases,
    kinematicPu: 471,
    units: 'k',
    deflectScale: 0.025,
    bmdScale: 0.0065,
    viewBox: { xMin: -10, xMax: 105, yMin: -6, yMax: 56 },
    multiCase: false,
    indeterminacyInit: 27,
    loadArrowScale: 30,
  };
}

const PROBLEMS = {
  P1A: makeP1A(),
  P1B: makeP1B(),
  P2: makeP2(),
  P3: makeP3(),
};

// =================================================================
// STRUCTURE ASSEMBLY & INCREMENTAL ANALYSIS
// =================================================================

function buildDofMap(nodes) {
  // Returns { dofMap: array of [u_dof, v_dof, theta_dof] per node, nFree, nFixed,
  //           freeDofs: list of (node, comp) for each free dof }
  // Free DOFs are numbered first, then fixed DOFs.
  const n = nodes.length;
  const dofMap = Array.from({ length: n }, () => [-1, -1, -1]);
  let freeIdx = 0;
  for (let i = 0; i < n; i++) {
    for (let c = 0; c < 3; c++) {
      if (!nodes[i].rest[c]) {
        dofMap[i][c] = freeIdx++;
      }
    }
  }
  return { dofMap, nFree: freeIdx };
}

function assembleK(structure, hingeState, dofMap, nFree) {
  const K = zeros2(nFree, nFree);
  const elements = structure.elements;
  const nodes = structure.nodes;
  for (let e = 0; e < elements.length; e++) {
    const el = elements[e];
    const ni = nodes[el.i], nj = nodes[el.j];
    const dx = nj.x - ni.x, dy = nj.y - ni.y;
    const L = Math.hypot(dx, dy);
    const c = dx / L, s = dy / L;
    const hi = hingeState[e][0];
    const hj = hingeState[e][1];
    const kL = localStiffness(L, el.EA, el.EI, hi, hj);
    const kG = transformToGlobal(kL, c, s);
    const dofs = [
      ...dofMap[el.i],
      ...dofMap[el.j],
    ];
    for (let r = 0; r < 6; r++) {
      const R = dofs[r];
      if (R < 0) continue;
      for (let cc = 0; cc < 6; cc++) {
        const C = dofs[cc];
        if (C < 0) continue;
        K[R][C] += kG[r][cc];
      }
    }
  }
  return K;
}

function assembleF(structure, loadVec, dofMap, nFree) {
  // loadVec: array same length as structure.loadCases. Each element is the scaling for that load case.
  const F = zeros1(nFree);
  for (let lc = 0; lc < structure.loadCases.length; lc++) {
    const scale = loadVec[lc];
    if (scale === 0) continue;
    for (const ld of structure.loadCases[lc].loads) {
      const dofs = dofMap[ld.node];
      const fxyz = [ld.fx, ld.fy, ld.m];
      for (let c = 0; c < 3; c++) {
        if (dofs[c] >= 0) F[dofs[c]] += scale * fxyz[c];
      }
    }
  }
  return F;
}

function fullDispFromFree(d_free, dofMap, nNodes) {
  // Returns 3*nNodes array of [u,v,theta] for each node.
  const d = zeros1(3 * nNodes);
  for (let i = 0; i < nNodes; i++) {
    for (let c = 0; c < 3; c++) {
      if (dofMap[i][c] >= 0) d[3 * i + c] = d_free[dofMap[i][c]];
    }
  }
  return d;
}

function elementMoments(structure, e, hingeState, dispFull) {
  // Returns [M_at_i, M_at_j] in sagging+ convention for element e.
  const el = structure.elements[e];
  const nodes = structure.nodes;
  const ni = nodes[el.i], nj = nodes[el.j];
  const dx = nj.x - ni.x, dy = nj.y - ni.y;
  const L = Math.hypot(dx, dy);
  const c = dx / L, s = dy / L;
  const hi = hingeState[e][0];
  const hj = hingeState[e][1];
  const kL = localStiffness(L, el.EA, el.EI, hi, hj);
  // Get global d for this element's 6 dofs
  const dG = [
    dispFull[3 * el.i],
    dispFull[3 * el.i + 1],
    dispFull[3 * el.i + 2],
    dispFull[3 * el.j],
    dispFull[3 * el.j + 1],
    dispFull[3 * el.j + 2],
  ];
  // Convert to local: dL = T · dG
  const dL = [
    c * dG[0] + s * dG[1],
    -s * dG[0] + c * dG[1],
    dG[2],
    c * dG[3] + s * dG[4],
    -s * dG[3] + c * dG[4],
    dG[5],
  ];
  // f_local = kL · dL
  const fL = zeros1(6);
  for (let r = 0; r < 6; r++) {
    let sum = 0;
    for (let cc = 0; cc < 6; cc++) sum += kL[r][cc] * dL[cc];
    fL[r] = sum;
  }
  // M_internal_sagging at i = -fL[2], at j = +fL[5]
  return [-fL[2], +fL[5]];
}

function runIncrementalAnalysis(structure, loadCaseSelection) {
  // loadCaseSelection: array of length structure.loadCases.length; entry i is the multiplier (typically 1 or 0).
  // Returns { Pu, hinges, history (snapshots), unitMomentHistory }
  const { dofMap, nFree } = buildDofMap(structure.nodes);
  const nElem = structure.elements.length;
  const hingeState = Array.from({ length: nElem }, () => [false, false]);
  const M_total = Array.from({ length: nElem }, () => [0, 0]); // sagging+
  const d_total = zeros1(3 * structure.nodes.length);
  let P_total = 0;
  const hinges = [];
  const snapshots = [
    {
      P: 0,
      d: [...d_total],
      M: M_total.map((m) => [...m]),
      hinges: [],
      indet: structure.indeterminacyInit,
    },
  ];

  const TOL_M = 1e-6;
  const MAX_HINGES = 60;

  for (let iter = 0; iter < MAX_HINGES; iter++) {
    const K = assembleK(structure, hingeState, dofMap, nFree);
    const F = assembleF(structure, loadCaseSelection, dofMap, nFree);
    const d_free = gaussSolve(K, F);
    if (d_free === null) break; // mechanism: K singular

    const d_full = fullDispFromFree(d_free, dofMap, structure.nodes.length);

    // Compute ΔM at each potential hinge for unit increment
    const dM = Array.from({ length: nElem }, () => [0, 0]);
    for (let e = 0; e < nElem; e++) {
      const m = elementMoments(structure, e, hingeState, d_full);
      dM[e][0] = hingeState[e][0] ? 0 : m[0];
      dM[e][1] = hingeState[e][1] ? 0 : m[1];
    }

    // Find smallest positive λ such that some unhinged loc reaches ±Mp.
    let lambdaMin = Infinity;
    let nextHinge = null;
    for (let e = 0; e < nElem; e++) {
      const Mp = structure.elements[e].Mp;
      for (let end = 0; end < 2; end++) {
        if (hingeState[e][end]) continue;
        const Mc = M_total[e][end];
        const dm = dM[e][end];
        if (Math.abs(dm) < 1e-12) continue;
        // λ for +Mp:
        const lambdaPlus = (Mp - Mc) / dm;
        const lambdaMinus = (-Mp - Mc) / dm;
        for (const [lambda, signed] of [
          [lambdaPlus, +1],
          [lambdaMinus, -1],
        ]) {
          if (lambda > 1e-9 && lambda < lambdaMin) {
            lambdaMin = lambda;
            nextHinge = { elem: e, end, sign: signed };
          }
        }
      }
    }

    if (nextHinge === null || !isFinite(lambdaMin)) break;

    // Update state
    P_total += lambdaMin;
    for (let e = 0; e < nElem; e++) {
      for (let end = 0; end < 2; end++) {
        if (!hingeState[e][end]) M_total[e][end] += lambdaMin * dM[e][end];
      }
    }
    for (let i = 0; i < d_total.length; i++) d_total[i] += lambdaMin * d_full[i];

    // Snap target moment to exact ±Mp to avoid drift
    M_total[nextHinge.elem][nextHinge.end] =
      nextHinge.sign * structure.elements[nextHinge.elem].Mp;

    // Insert hinge in element
    hingeState[nextHinge.elem][nextHinge.end] = true;
    hinges.push({ ...nextHinge, P: P_total });

    snapshots.push({
      P: P_total,
      d: [...d_total],
      M: M_total.map((m) => [...m]),
      hinges: hinges.map((h) => ({ ...h })),
      indet: structure.indeterminacyInit - hinges.length,
    });
  }

  return {
    Pu: P_total,
    hinges,
    snapshots,
    nFree,
    dofMap,
    isComplete: true,
  };
}

// Interpolate state at arbitrary P between snapshots.
function interpolateState(snapshots, P) {
  if (P <= 0) return snapshots[0];
  // Find segment: snapshots[k].P <= P <= snapshots[k+1].P
  for (let k = 0; k < snapshots.length - 1; k++) {
    if (P <= snapshots[k + 1].P + 1e-12) {
      const Pa = snapshots[k].P;
      const Pb = snapshots[k + 1].P;
      const t = Pb > Pa ? (P - Pa) / (Pb - Pa) : 0;
      // Linear interpolation of d and M
      const da = snapshots[k].d;
      const db = snapshots[k + 1].d;
      const d = da.map((v, i) => v + t * (db[i] - v));
      const Ma = snapshots[k].M;
      const Mb = snapshots[k + 1].M;
      const M = Ma.map((row, e) => row.map((v, end) => v + t * (Mb[e][end] - v)));
      return {
        P,
        d,
        M,
        hinges: snapshots[k].hinges,
        indet: snapshots[k].indet,
        nextHingeIdx: k < snapshots.length - 1 ? k : null,
      };
    }
  }
  // Beyond mechanism: extrapolate from last segment (visualization only)
  const last = snapshots[snapshots.length - 1];
  const prev = snapshots[snapshots.length - 2] || snapshots[0];
  if (last.P > prev.P) {
    const t = (P - last.P) / (last.P - prev.P);
    const d = last.d.map((v, i) => v + t * (v - prev.d[i]));
    const M = last.M.map((row) => [...row]);
    return { P, d, M, hinges: last.hinges, indet: last.indet, nextHingeIdx: null, beyond: true };
  }
  return { ...last, P, beyond: true, nextHingeIdx: null };
}


// =================================================================
// PUSHOVER POST-PROCESSING (Tier 1/2 — additive, no solver changes)
// =================================================================

// End rotation of element e at end (0=i, 1=j) relative to the chord, in
// the small-rotation approximation. This is the rotation that, multiplied
// by EI/L, gives the elastic moment at that end. After a hinge forms, the
// relative-rotation increment IS the plastic rotation θ_p.
function endRelRotation(structure, e, end, dispFull) {
  const el = structure.elements[e];
  const ni = structure.nodes[el.i], nj = structure.nodes[el.j];
  const dx = nj.x - ni.x, dy = nj.y - ni.y;
  const L = Math.hypot(dx, dy);
  const c = dx / L, s = dy / L;
  const ux_i = dispFull[3 * el.i],     uy_i = dispFull[3 * el.i + 1];
  const ux_j = dispFull[3 * el.j],     uy_j = dispFull[3 * el.j + 1];
  const vli = -ux_i * s + uy_i * c;    // local transverse displacement at i
  const vlj = -ux_j * s + uy_j * c;    // local transverse displacement at j
  const chord = (vlj - vli) / L;       // small-rotation chord rotation
  const nodeIdx = end === 0 ? el.i : el.j;
  const theta_global = dispFull[3 * nodeIdx + 2];
  return theta_global - chord;
}

// For one specific hinge, return |θ_p| at every snapshot. Zero before
// the hinge has formed; afterward it tracks the magnitude of relative
// rotation accumulated since the moment of formation.
function hingePlasticRotation(structure, snapshots, elemIdx, endIdx) {
  let formedAt = -1;
  for (let k = 1; k < snapshots.length; k++) {
    const has = snapshots[k].hinges.some(h => h.elem === elemIdx && h.end === endIdx);
    const had = snapshots[k - 1].hinges.some(h => h.elem === elemIdx && h.end === endIdx);
    if (has && !had) { formedAt = k; break; }
  }
  if (formedAt < 0) return snapshots.map(() => 0);
  const refRel = endRelRotation(structure, elemIdx, endIdx, snapshots[formedAt].d);
  return snapshots.map((s, k) => {
    if (k <= formedAt) return 0;
    const rel = endRelRotation(structure, elemIdx, endIdx, s.d);
    return Math.abs(rel - refRel);
  });
}

// (P, Δ) curve points sampled at every snapshot.
function pushoverPoints(structure, snapshots) {
  if (structure.controlDof === undefined) return [];
  return snapshots.map(s => ({ P: s.P, delta: s.d[structure.controlDof] }));
}

// Maximum |θ_p| across all hinges that have formed by snapshot k.
function maxThetaPBySnapshot(structure, snapshots, hinges) {
  const perHinge = hinges.map(h => hingePlasticRotation(structure, snapshots, h.elem, h.end));
  return snapshots.map((_, k) => {
    let m = 0;
    for (const arr of perHinge) if (arr[k] > m) m = arr[k];
    return m;
  });
}

// Yield rotation θy = relative rotation at the first hinge's location,
// at the snapshot where it formed. Used as the unit of ductility for
// performance-level thresholds.
function yieldRotation(structure, snapshots) {
  if (snapshots.length < 2 || snapshots[1].hinges.length === 0) return 1e-6;
  const h = snapshots[1].hinges[0];
  const r = endRelRotation(structure, h.elem, h.end, snapshots[1].d);
  return Math.max(Math.abs(r), 1e-6);
}

// Find P, Δ at the moment max(|θ_p|) first crosses `threshold`. Linear
// interpolation between the two bracketing snapshots. null if never reached.
function findThresholdCrossing(snapshots, maxTheta, threshold, controlDof) {
  for (let k = 1; k < snapshots.length; k++) {
    if (maxTheta[k] >= threshold) {
      const t1 = maxTheta[k - 1], t2 = maxTheta[k];
      const frac = t2 > t1 ? (threshold - t1) / (t2 - t1) : 0;
      const Pa = snapshots[k - 1].P, Pb = snapshots[k].P;
      const da = controlDof !== undefined ? snapshots[k - 1].d[controlDof] : 0;
      const db = controlDof !== undefined ? snapshots[k].d[controlDof] : 0;
      return { P: Pa + frac * (Pb - Pa), delta: da + frac * (db - da), k };
    }
  }
  return null;
}

// D/C ratios at every potential hinge (every element end) for a given
// snapshot. Used by the Walkthrough panel to surface "next hinge to yield".
function dcRatios(structure, snapshot) {
  const out = [];
  for (let e = 0; e < structure.elements.length; e++) {
    const el = structure.elements[e];
    for (let end = 0; end < 2; end++) {
      const M = snapshot.M[e][end];
      const isHinged = snapshot.hinges.some(h => h.elem === e && h.end === end);
      out.push({
        elem: e, end, M, Mp: el.Mp,
        dc: Math.abs(M) / el.Mp,
        residual: el.Mp - Math.abs(M),
        isHinged,
        nodeLabel: structure.nodes[end === 0 ? el.i : el.j].label,
        elLabel: el.label || `e${e}`,
        elType: el.type,
      });
    }
  }
  out.sort((a, b) => b.dc - a.dc);
  return out;
}

// =================================================================
// VISUALIZATION HELPERS
// =================================================================

function elementGeom(structure, e) {
  const el = structure.elements[e];
  const ni = structure.nodes[el.i], nj = structure.nodes[el.j];
  const dx = nj.x - ni.x, dy = nj.y - ni.y;
  const L = Math.hypot(dx, dy);
  const c = dx / L, s = dy / L;
  // Perpendicular CW from local-x: (s, -c) — for sagging+ "tension side"
  return { L, c, s, ni, nj, perp: [s, -c] };
}

// Cubic Hermite shape function for transverse local v(xi) where xi = x/L.
function hermitePoints(L, vli, thi, vlj, thj, n) {
  const pts = [];
  for (let k = 0; k <= n; k++) {
    const xi = k / n;
    const N1 = 1 - 3 * xi * xi + 2 * xi * xi * xi;
    const N2 = L * (xi - 2 * xi * xi + xi * xi * xi);
    const N3 = 3 * xi * xi - 2 * xi * xi * xi;
    const N4 = L * (-xi * xi + xi * xi * xi);
    pts.push({ xi, vL: N1 * vli + N2 * thi + N3 * vlj + N4 * thj });
  }
  return pts;
}

function deflectedPath(structure, e, dispFull, scale) {
  const { L, c, s, ni } = elementGeom(structure, e);
  const i = structure.elements[e].i;
  const j = structure.elements[e].j;
  // Global displacements at both ends
  const ui = dispFull[3 * i], vi = dispFull[3 * i + 1], thi = dispFull[3 * i + 2];
  const uj = dispFull[3 * j], vj = dispFull[3 * j + 1], thj = dispFull[3 * j + 2];
  // Rotate to local: axial uL and transverse vL at each end
  const uli = c * ui + s * vi;
  const vli = -s * ui + c * vi;
  const ulj = c * uj + s * vj;
  const vlj = -s * uj + c * vj;
  // Cubic Hermite for transverse, linear for axial
  const pts = hermitePoints(L, vli, thi, vlj, thj, 12);
  return pts.map((p) => {
    const uL = (1 - p.xi) * uli + p.xi * ulj;
    const xL = p.xi * L;
    // ni + xL·(c,s) along axis, + scale·uL·(c,s) axial deformation, + scale·vL·(-s,c) transverse
    return {
      x: ni.x + xL * c + scale * uL * c + scale * p.vL * -s,
      y: ni.y + xL * s + scale * uL * s + scale * p.vL * c,
    };
  });
}

// Deflected position of a single node — for placing markers, hinges, load arrows
// on the deformed configuration so they stay attached to their joints.
function deflectedNode(node, nodeIdx, dispFull, scale) {
  return {
    x: node.x + scale * dispFull[3 * nodeIdx],
    y: node.y + scale * dispFull[3 * nodeIdx + 1],
  };
}

function bmdPolygonPoints(structure, e, M_at, scale) {
  const { L, c, s, ni, nj, perp } = elementGeom(structure, e);
  const Mi = M_at[0], Mj = M_at[1];
  // Polygon: ni → ni+perp·scale·Mi → nj+perp·scale·Mj → nj → close
  const a = { x: ni.x, y: ni.y };
  const b = { x: ni.x + perp[0] * scale * Mi, y: ni.y + perp[1] * scale * Mi };
  const cP = { x: nj.x + perp[0] * scale * Mj, y: nj.y + perp[1] * scale * Mj };
  const d = { x: nj.x, y: nj.y };
  // If sign change, split at zero crossing
  if (Mi * Mj < 0) {
    const f = Mi / (Mi - Mj);
    const z = { x: ni.x + f * (nj.x - ni.x), y: ni.y + f * (nj.y - ni.y) };
    return [
      { points: [a, b, z], sign: Math.sign(Mi) },
      { points: [z, cP, d], sign: Math.sign(Mj) },
    ];
  }
  if (Math.abs(Mi) < 1e-9 && Math.abs(Mj) < 1e-9) {
    return [];
  }
  return [
    { points: [a, b, cP, d], sign: Math.sign(Mi !== 0 ? Mi : Mj) },
  ];
}

// =================================================================
// REACT — RENDERING COMPONENTS
// =================================================================

function svgTransform(pt, vb, w, h, padding) {
  const sx = (w - 2 * padding) / (vb.xMax - vb.xMin);
  const sy = (h - 2 * padding) / (vb.yMax - vb.yMin);
  const scale = Math.min(sx, sy);
  const tx = padding + ((vb.xMax - vb.xMin) * sx - (vb.xMax - vb.xMin) * scale) / 2;
  return {
    x: tx + (pt.x - vb.xMin) * scale,
    y: padding + (vb.yMax - pt.y) * scale,
  };
}

function makeXform(vb, w, h, padding) {
  const sx = (w - 2 * padding) / (vb.xMax - vb.xMin);
  const sy = (h - 2 * padding) / (vb.yMax - vb.yMin);
  const scale = Math.min(sx, sy);
  const usedW = (vb.xMax - vb.xMin) * scale;
  const usedH = (vb.yMax - vb.yMin) * scale;
  const tx = (w - usedW) / 2;
  const ty = (h - usedH) / 2;
  return {
    scale,
    p: (pt) => ({
      x: tx + (pt.x - vb.xMin) * scale,
      y: ty + (vb.yMax - pt.y) * scale,
    }),
  };
}

function StructureView({ problem, state, scale, snapshots, currentP, Pu, deflectScaleAdj }) {
  const w = 540, h = 320;
  const padding = 24;
  const xform = makeXform(problem.viewBox, w, h, padding);

  const dispFull = state.d;
  const M_at = state.M;
  const hinges = state.hinges;
  const nextIdx = state.nextHingeIdx;

  const dScale = problem.deflectScale * deflectScaleAdj;

  // Determine visible hinges
  const visibleHinges = hinges; // all formed before now

  // Find next hinge that *will form* (open circle): the hinge that forms at the upcoming snapshot's P, when current P is between previous snapshot.P and that next.P.
  let formingHinge = null;
  if (nextIdx !== null && snapshots[nextIdx + 1]) {
    const nextSnap = snapshots[nextIdx + 1];
    if (currentP < nextSnap.P - 1e-9 && nextSnap.hinges.length > visibleHinges.length) {
      formingHinge = nextSnap.hinges[nextSnap.hinges.length - 1];
    }
  }

  return (
    <svg viewBox={`0 0 ${w} ${h}`} className="w-full h-full" style={{ background: 'transparent' }}>
      {/* graph paper grid */}
      <defs>
        <pattern id={`grid-${problem.name}`} width="20" height="20" patternUnits="userSpaceOnUse">
          <path d="M 20 0 L 0 0 0 20" fill="none" stroke="#e8e4d8" strokeWidth="0.5" />
        </pattern>
        <pattern id={`gridmajor-${problem.name}`} width="100" height="100" patternUnits="userSpaceOnUse">
          <rect width="100" height="100" fill={`url(#grid-${problem.name})`} />
          <path d="M 100 0 L 0 0 0 100" fill="none" stroke="#dbd6c5" strokeWidth="0.7" />
        </pattern>
      </defs>
      <rect width={w} height={h} fill={`url(#gridmajor-${problem.name})`} />

      {/* Original structure (faint) */}
      {problem.elements.map((el, e) => {
        const a = xform.p(problem.nodes[el.i]);
        const b = xform.p(problem.nodes[el.j]);
        return (
          <line
            key={`o${e}`}
            x1={a.x} y1={a.y} x2={b.x} y2={b.y}
            stroke="#cdc9b8"
            strokeWidth={el.type === 'col' ? 2 : 1.5}
            strokeDasharray="3,3"
          />
        );
      })}

      {/* Deflected shape */}
      {problem.elements.map((el, e) => {
        const pts = deflectedPath(problem, e, dispFull, dScale);
        const dStr = pts
          .map((p, k) => {
            const sp = xform.p(p);
            return (k === 0 ? 'M' : 'L') + sp.x + ',' + sp.y;
          })
          .join(' ');
        return (
          <path
            key={`d${e}`}
            d={dStr}
            fill="none"
            stroke="#1a1a1a"
            strokeWidth={el.type === 'col' ? 2.4 : 1.8}
            strokeLinejoin="round"
            strokeLinecap="round"
          />
        );
      })}

      {/* Restraints */}
      {problem.nodes.map((n, i) => {
        const p = xform.p(n);
        if (n.type === 'fixed') {
          return (
            <g key={`r${i}`} transform={`translate(${p.x},${p.y})`}>
              <rect x="-12" y="0" width="24" height="6" fill="#888" />
              <line x1="-12" y1="6" x2="12" y2="6" stroke="#444" strokeWidth="1" />
              {[...Array(5)].map((_, k) => (
                <line
                  key={k}
                  x1={-10 + 5 * k}
                  y1={6}
                  x2={-14 + 5 * k}
                  y2={11}
                  stroke="#666"
                  strokeWidth="0.8"
                />
              ))}
            </g>
          );
        }
        if (n.type === 'pin') {
          return (
            <g key={`r${i}`} transform={`translate(${p.x},${p.y})`}>
              <polygon points="0,0 -8,12 8,12" fill="white" stroke="#333" strokeWidth="1.2" />
              <line x1="-12" y1="12" x2="12" y2="12" stroke="#333" strokeWidth="1.2" />
              {[...Array(5)].map((_, k) => (
                <line
                  key={k}
                  x1={-10 + 5 * k}
                  y1={12}
                  x2={-14 + 5 * k}
                  y2={17}
                  stroke="#666"
                  strokeWidth="0.8"
                />
              ))}
            </g>
          );
        }
        if (n.type === 'roller') {
          return (
            <g key={`r${i}`} transform={`translate(${p.x},${p.y})`}>
              <polygon points="0,0 -8,10 8,10" fill="white" stroke="#333" strokeWidth="1.2" />
              <circle cx="-4" cy="13" r="2.5" fill="white" stroke="#333" strokeWidth="1" />
              <circle cx="4" cy="13" r="2.5" fill="white" stroke="#333" strokeWidth="1" />
              <line x1="-12" y1="17" x2="12" y2="17" stroke="#333" strokeWidth="1.2" />
            </g>
          );
        }
        return null;
      })}

      {/* Loads (scaled by current P) — arrow tip on deflected joint */}
      {(() => {
        const arrowScale = problem.loadArrowScale * (currentP / Pu);
        const arrows = [];
        let lkey = 0;
        for (let lc = 0; lc < problem.loadCases.length; lc++) {
          const sel = scale[lc];
          if (sel === 0) continue;
          for (const ld of problem.loadCases[lc].loads) {
            const node = problem.nodes[ld.node];
            const nDef = deflectedNode(node, ld.node, dispFull, dScale);
            const np = xform.p(nDef);
            const fmag = Math.hypot(ld.fx, ld.fy);
            if (fmag < 1e-9) continue;
            const vx_struct = ld.fx * arrowScale;
            const vy_struct = ld.fy * arrowScale;
            const dx = vx_struct * xform.scale;
            const dy = -vy_struct * xform.scale;
            const fromX = np.x - dx;
            const fromY = np.y - dy;
            const len = Math.hypot(dx, dy);
            if (len < 2) continue;
            const ux = dx / len, uy = dy / len;
            arrows.push(
              <g key={`L${lkey++}`}>
                <line x1={fromX} y1={fromY} x2={np.x} y2={np.y} stroke="#1f4d8a" strokeWidth="1.6" />
                <polygon
                  points={`${np.x},${np.y} ${np.x - 7 * ux + 3.5 * uy},${np.y - 7 * uy - 3.5 * ux} ${np.x - 7 * ux - 3.5 * uy},${np.y - 7 * uy + 3.5 * ux}`}
                  fill="#1f4d8a"
                />
              </g>
            );
          }
        }
        return arrows;
      })()}

      {/* Hinges (formed: filled red; forming: open red) — follow deflected joint */}
      {visibleHinges.map((h, k) => {
        const el = problem.elements[h.elem];
        const nIdx = h.end === 0 ? el.i : el.j;
        const oIdx = h.end === 0 ? el.j : el.i;
        const node = problem.nodes[nIdx];
        const otherNode = problem.nodes[oIdx];
        // Use deflected node positions in structure coords, then xform to screen
        const nDef = deflectedNode(node, nIdx, dispFull, dScale);
        const oDef = deflectedNode(otherNode, oIdx, dispFull, dScale);
        const np = xform.p(nDef);
        const op = xform.p(oDef);
        const sdx = op.x - np.x, sdy = op.y - np.y;
        const sLen = Math.hypot(sdx, sdy);
        const PIX = Math.min(11, sLen * 0.18);
        const sp = { x: np.x + (sdx / sLen) * PIX, y: np.y + (sdy / sLen) * PIX };
        return (
          <circle
            key={`h${k}`}
            cx={sp.x}
            cy={sp.y}
            r="4.5"
            fill="#c43e3e"
            stroke="#7a1f1f"
            strokeWidth="1"
          />
        );
      })}
      {formingHinge && (() => {
        const el = problem.elements[formingHinge.elem];
        const nIdx = formingHinge.end === 0 ? el.i : el.j;
        const oIdx = formingHinge.end === 0 ? el.j : el.i;
        const node = problem.nodes[nIdx];
        const otherNode = problem.nodes[oIdx];
        const nDef = deflectedNode(node, nIdx, dispFull, dScale);
        const oDef = deflectedNode(otherNode, oIdx, dispFull, dScale);
        const np = xform.p(nDef);
        const op = xform.p(oDef);
        const sdx = op.x - np.x, sdy = op.y - np.y;
        const sLen = Math.hypot(sdx, sdy);
        const PIX = Math.min(11, sLen * 0.18);
        const sp = { x: np.x + (sdx / sLen) * PIX, y: np.y + (sdy / sLen) * PIX };
        return (
          <circle
            cx={sp.x}
            cy={sp.y}
            r="6"
            fill="white"
            stroke="#c43e3e"
            strokeWidth="1.7"
            strokeDasharray="2.5,1.5"
          />
        );
      })()}

      {/* Node labels (small, only for key joints) — track deflected joint */}
      {problem.nodes.map((n, i) => {
        if (!n.label || n.type === 'beam-pt') return null;
        // Fixed supports stay at their original coords (the support itself doesn't move)
        const useDef = n.type !== 'fixed' && n.type !== 'pin' && n.type !== 'roller';
        const pt = useDef ? deflectedNode(n, i, dispFull, dScale) : n;
        const p = xform.p(pt);
        let dx = 8, dy = -6;
        if (n.type === 'fixed') { dy = 22; dx = 0; }
        return (
          <text
            key={`lab${i}`}
            x={p.x + dx}
            y={p.y + dy}
            fontSize="10"
            fontFamily="ui-monospace, Menlo, monospace"
            fill="#666"
            textAnchor={dx < 0 ? 'end' : dx === 0 ? 'middle' : 'start'}
          >
            {n.label}
          </text>
        );
      })}
    </svg>
  );
}

function BMDView({ problem, state, scale, currentP, Pu }) {
  const w = 480, h = 320;
  const padding = 24;
  const xform = makeXform(problem.viewBox, w, h, padding);

  const M_at = state.M;
  const hinges = state.hinges;
  const bmdScale = problem.bmdScale;

  return (
    <svg viewBox={`0 0 ${w} ${h}`} className="w-full h-full">
      <defs>
        <pattern id={`gridb-${problem.name}`} width="20" height="20" patternUnits="userSpaceOnUse">
          <path d="M 20 0 L 0 0 0 20" fill="none" stroke="#e8e4d8" strokeWidth="0.5" />
        </pattern>
      </defs>
      <rect width={w} height={h} fill={`url(#gridb-${problem.name})`} />

      {/* Reference structure (very light) */}
      {problem.elements.map((el, e) => {
        const a = xform.p(problem.nodes[el.i]);
        const b = xform.p(problem.nodes[el.j]);
        return (
          <line
            key={`o${e}`}
            x1={a.x} y1={a.y} x2={b.x} y2={b.y}
            stroke="#bcb8a8"
            strokeWidth={el.type === 'col' ? 1.5 : 1.2}
          />
        );
      })}

      {/* BMD polygons */}
      {problem.elements.map((el, e) => {
        const polys = bmdPolygonPoints(problem, e, M_at[e], bmdScale);
        return polys.map((poly, k) => {
          const ptsStr = poly.points
            .map((p) => {
              const sp = xform.p(p);
              return `${sp.x},${sp.y}`;
            })
            .join(' ');
          const fill = poly.sign > 0 ? 'rgba(75, 130, 195, 0.35)' : 'rgba(195, 130, 75, 0.35)';
          const stroke = poly.sign > 0 ? '#2c5d9b' : '#9b5d2c';
          return (
            <polygon
              key={`b${e}-${k}`}
              points={ptsStr}
              fill={fill}
              stroke={stroke}
              strokeWidth="0.9"
            />
          );
        });
      })}

      {/* Hinges on BMD (mark with circle) */}
      {hinges.map((h, k) => {
        const el = problem.elements[h.elem];
        const nodeI = problem.nodes[el.i], nodeJ = problem.nodes[el.j];
        const node = h.end === 0 ? nodeI : nodeJ;
        const otherNode = h.end === 0 ? nodeJ : nodeI;
        const dx = otherNode.x - node.x, dy = otherNode.y - node.y;
        const L = Math.hypot(dx, dy);
        const M_at_hinge = M_at[h.elem][h.end];
        const c = dx / L, s = dy / L;
        const perp = [s, -c];
        // Position at the joint plus the BMD perpendicular offset
        // (BMD polygon ends at structure-coord ni + perp*scale*Mi, so place mark there)
        // Walk 8 px along element from joint, then add full BMD perp offset
        const npScreen = xform.p(node);
        const opScreen = xform.p(otherNode);
        const sdx = opScreen.x - npScreen.x, sdy = opScreen.y - npScreen.y;
        const sLen = Math.hypot(sdx, sdy);
        const walk = Math.min(8, sLen * 0.12);
        const baseX = npScreen.x + (sdx / sLen) * walk;
        const baseY = npScreen.y + (sdy / sLen) * walk;
        // Add BMD perpendicular offset, scaled to screen
        const perpScreenX = perp[0] * bmdScale * M_at_hinge * xform.scale;
        const perpScreenY = -perp[1] * bmdScale * M_at_hinge * xform.scale;
        const sp = { x: baseX + perpScreenX, y: baseY + perpScreenY };
        return (
          <g key={`bh${k}`}>
            <circle cx={sp.x} cy={sp.y} r="3" fill="#c43e3e" />
            <text
              x={sp.x + (perp[0] >= 0 ? 6 : -6)}
              y={sp.y - 2}
              fontSize="9"
              fontFamily="ui-monospace, monospace"
              fill="#444"
              textAnchor={perp[0] >= 0 ? 'start' : 'end'}
            >
              {M_at_hinge >= 0 ? '+' : ''}{Math.abs(M_at_hinge) < 100 ? M_at_hinge.toFixed(2) : M_at_hinge.toFixed(0)}
            </text>
          </g>
        );
      })}
    </svg>
  );
}

// =================================================================
// MAIN APP
// =================================================================


// =================================================================
// PUSHOVER CURVE PANEL (Tier 1 + Tier 2 + Tier 3)
// =================================================================
function PushoverPanel({ problem, analysis, currentP, P_norm, setP_norm, connection }) {
  const points = useMemo(
    () => pushoverPoints(problem, analysis.snapshots),
    [problem, analysis.snapshots]
  );
  const maxTheta = useMemo(
    () => maxThetaPBySnapshot(problem, analysis.snapshots, analysis.hinges),
    [problem, analysis.snapshots, analysis.hinges]
  );
  const theta_y = useMemo(
    () => yieldRotation(problem, analysis.snapshots),
    [problem, analysis.snapshots]
  );

  const conn = ASCE41_CONNECTIONS[connection] || ASCE41_CONNECTIONS[ASCE41_DEFAULT_KEY];
  const theta_LS = theta_y * conn.mu_LS;
  const theta_CP = theta_y * conn.mu_CP;

  // First-yield (Δy) is the displacement at snapshot 1 (first hinge formation).
  const yieldPt = points.length > 1 ? points[1] : null;
  // Performance crossings on the curve (in P, Δ space).
  const cross_LS = findThresholdCrossing(analysis.snapshots, maxTheta, theta_LS, problem.controlDof);
  const cross_CP = findThresholdCrossing(analysis.snapshots, maxTheta, theta_CP, problem.controlDof);

  // Δu = where structure first crosses CP (collapse prevention failure).
  const ult = cross_CP || (points.length > 0 ? points[points.length - 1] : null);
  const mu_struct = yieldPt && ult && yieldPt.delta > 1e-9 ? ult.delta / yieldPt.delta : null;

  // SVG geometry
  const W = 620, H = 320;
  const padL = 56, padR = 16, padT = 14, padB = 36;
  const plotW = W - padL - padR, plotH = H - padT - padB;

  // Determine plot extents — extend slightly past CP failure if known.
  const lastP = points.length ? points[points.length - 1].P : 1;
  const lastD = points.length ? points[points.length - 1].delta : 1;
  const xMax = Math.max(
    (ult ? ult.delta : lastD) * 1.15,
    lastD * 1.05,
    1e-6
  );
  const yMax = Math.max(lastP * 1.10, 1e-6);

  const xToPx = x => padL + (x / xMax) * plotW;
  const yToPx = y => padT + plotH - (y / yMax) * plotH;

  // Live (P, Δ) point from current slider position — interpolate.
  const liveDelta = (() => {
    if (!problem.controlDof === undefined) return 0;
    const st = interpolateState(analysis.snapshots, currentP);
    return st.d[problem.controlDof] || 0;
  })();

  const polyline = points.map(p => `${xToPx(p.delta)},${yToPx(p.P)}`).join(' ');

  // Performance-band x-coordinates
  const xIO_lo = yieldPt ? xToPx(yieldPt.delta) : padL;
  const xLS_lo = cross_LS ? xToPx(cross_LS.delta) : null;
  const xCP_lo = cross_CP ? xToPx(cross_CP.delta) : null;
  const xCP_hi = padL + plotW;

  return (
    <div className="border border-stone-300 bg-white rounded-sm relative">
      <div
        className="absolute top-1.5 left-2 text-[10px] uppercase tracking-widest text-stone-500"
        style={{ fontFamily: 'ui-monospace, monospace' }}
      >
        Pushover · capacity curve
      </div>
      <svg viewBox={`0 0 ${W} ${H}`} width="100%" preserveAspectRatio="xMidYMid meet">
        {/* Performance bands (under the curve) */}
        {yieldPt && xLS_lo && (
          <rect x={xIO_lo} y={padT} width={xLS_lo - xIO_lo} height={plotH}
                fill="#7fb98c" fillOpacity="0.10" />
        )}
        {xLS_lo && xCP_lo && (
          <rect x={xLS_lo} y={padT} width={xCP_lo - xLS_lo} height={plotH}
                fill="#e2c14d" fillOpacity="0.13" />
        )}
        {xCP_lo && (
          <rect x={xCP_lo} y={padT} width={xCP_hi - xCP_lo} height={plotH}
                fill="#c43e3e" fillOpacity="0.10" />
        )}

        {/* Axes */}
        <line x1={padL} y1={padT + plotH} x2={padL + plotW} y2={padT + plotH} stroke="#666" />
        <line x1={padL} y1={padT} x2={padL} y2={padT + plotH} stroke="#666" />

        {/* X-axis ticks */}
        {Array.from({ length: 5 }, (_, i) => {
          const x = (xMax * (i + 1)) / 5;
          return (
            <g key={`xt${i}`}>
              <line x1={xToPx(x)} y1={padT + plotH} x2={xToPx(x)} y2={padT + plotH + 4} stroke="#666" />
              <text x={xToPx(x)} y={padT + plotH + 16} textAnchor="middle"
                    fontSize="10" fill="#555" fontFamily="ui-monospace, monospace">
                {x.toFixed(x < 1 ? 3 : 2)}
              </text>
            </g>
          );
        })}
        <text x={padL + plotW / 2} y={H - 6} textAnchor="middle" fontSize="11" fill="#333"
              fontFamily="ui-monospace, monospace">
          Δ (control displacement, length-units)
        </text>

        {/* Y-axis ticks */}
        {Array.from({ length: 5 }, (_, i) => {
          const y = (yMax * (i + 1)) / 5;
          return (
            <g key={`yt${i}`}>
              <line x1={padL - 4} y1={yToPx(y)} x2={padL} y2={yToPx(y)} stroke="#666" />
              <text x={padL - 6} y={yToPx(y) + 3} textAnchor="end"
                    fontSize="10" fill="#555" fontFamily="ui-monospace, monospace">
                {y.toFixed(y < 1 ? 3 : y < 10 ? 2 : 1)}
              </text>
            </g>
          );
        })}
        <text x={14} y={padT + plotH / 2} textAnchor="middle" fontSize="11" fill="#333"
              fontFamily="ui-monospace, monospace"
              transform={`rotate(-90 14 ${padT + plotH / 2})`}>
          P ({problem.units})
        </text>

        {/* Capacity curve */}
        <polyline points={polyline} fill="none" stroke="#1a1a1a" strokeWidth="1.8" />

        {/* Event markers H1, H2, ... */}
        {points.slice(1).map((p, k) => (
          <g key={k}>
            <circle cx={xToPx(p.delta)} cy={yToPx(p.P)} r="3.5" fill="#fff" stroke="#1a1a1a" strokeWidth="1.4" />
            <text x={xToPx(p.delta) + 5} y={yToPx(p.P) - 5}
                  fontSize="9.5" fill="#1a1a1a" fontFamily="ui-monospace, monospace" fontWeight="600">
              H{k + 1}
            </text>
          </g>
        ))}

        {/* Δy marker (vertical dashed) */}
        {yieldPt && (
          <g>
            <line x1={xToPx(yieldPt.delta)} y1={padT} x2={xToPx(yieldPt.delta)} y2={padT + plotH}
                  stroke="#1f7a3a" strokeWidth="1" strokeDasharray="3,3" opacity="0.6" />
            <text x={xToPx(yieldPt.delta) + 3} y={padT + 11}
                  fontSize="9.5" fill="#1f7a3a" fontFamily="ui-monospace, monospace" fontWeight="600">
              Δy
            </text>
          </g>
        )}
        {/* Δu marker (CP failure) */}
        {cross_CP && (
          <g>
            <line x1={xToPx(cross_CP.delta)} y1={padT} x2={xToPx(cross_CP.delta)} y2={padT + plotH}
                  stroke="#c43e3e" strokeWidth="1.2" strokeDasharray="3,3" opacity="0.7" />
            <text x={xToPx(cross_CP.delta) + 3} y={padT + 23}
                  fontSize="9.5" fill="#c43e3e" fontFamily="ui-monospace, monospace" fontWeight="600">
              Δu (CP)
            </text>
            <circle cx={xToPx(cross_CP.delta)} cy={yToPx(cross_CP.P)} r="4"
                    fill="#c43e3e" stroke="#fff" strokeWidth="1.2" />
          </g>
        )}

        {/* Live (P, Δ) marker from slider */}
        {points.length > 0 && (
          <g>
            <circle cx={xToPx(liveDelta)} cy={yToPx(currentP)} r="5"
                    fill="#1a1a1a" stroke="#fff" strokeWidth="1.5" />
          </g>
        )}
      </svg>

      {/* Readout strip */}
      <div className="px-3 py-2 border-t border-stone-200 grid grid-cols-2 sm:grid-cols-4 gap-2 text-xs"
           style={{ fontFamily: 'ui-monospace, monospace' }}>
        <div>
          <div className="text-[9px] uppercase tracking-widest text-stone-500">Δy</div>
          <div className="tabular-nums font-semibold">{yieldPt ? yieldPt.delta.toExponential(2) : '—'}</div>
        </div>
        <div>
          <div className="text-[9px] uppercase tracking-widest text-stone-500">Δu (CP)</div>
          <div className="tabular-nums font-semibold" style={{ color: cross_CP ? '#c43e3e' : '#666' }}>
            {cross_CP ? cross_CP.delta.toExponential(2) : 'beyond range'}
          </div>
        </div>
        <div>
          <div className="text-[9px] uppercase tracking-widest text-stone-500">μ = Δu / Δy</div>
          <div className="tabular-nums font-semibold">{mu_struct ? mu_struct.toFixed(2) : '—'}</div>
        </div>
        <div>
          <div className="text-[9px] uppercase tracking-widest text-stone-500">θy</div>
          <div className="tabular-nums">{theta_y.toExponential(2)} rad</div>
        </div>
      </div>
    </div>
  );
}

// =================================================================
// HAND-PROCEDURE WALKTHROUGH PANEL (Tier 5)
// =================================================================
// Step-by-step narration matching Laursen's lecture: at every event,
// show ΔP and Δδ for that step, the hinge that formed, and the D/C
// ratios at the remaining potential hinge locations so the viewer can
// see WHICH hinge will yield next and why.
function WalkthroughPanel({ problem, analysis, currentP }) {
  const snaps = analysis.snapshots;
  const points = useMemo(() => pushoverPoints(problem, snaps), [problem, snaps]);

  // Identify "current step" — the latest snapshot reached by the slider.
  const currentStepIdx = (() => {
    let idx = 0;
    for (let k = 0; k < snaps.length; k++) if (snaps[k].P <= currentP + 1e-9) idx = k;
    return idx;
  })();

  return (
    <div className="border border-stone-300 bg-white rounded-sm relative">
      <div
        className="absolute top-1.5 left-2 text-[10px] uppercase tracking-widest text-stone-500"
        style={{ fontFamily: 'ui-monospace, monospace' }}
      >
        Hand procedure · Laursen-style walkthrough
      </div>
      <div className="pt-7 pb-2 px-3 max-h-[320px] overflow-y-auto"
           style={{ fontFamily: 'ui-monospace, monospace' }}>
        {snaps.map((s, k) => {
          const isCurrent = k === currentStepIdx;
          const isReached = currentP >= s.P - 1e-9;
          const prev = k > 0 ? snaps[k - 1] : null;
          const dP = prev ? s.P - prev.P : 0;
          const dDelta = (prev && problem.controlDof !== undefined)
            ? s.d[problem.controlDof] - prev.d[problem.controlDof]
            : 0;
          const dRatios = dcRatios(problem, s);
          const formedHinge = (prev && s.hinges.length > prev.hinges.length)
            ? s.hinges[s.hinges.length - 1] : null;
          const formedEl = formedHinge ? problem.elements[formedHinge.elem] : null;
          const formedNode = formedHinge
            ? problem.nodes[formedHinge.end === 0 ? formedEl.i : formedEl.j]
            : null;
          const nextHinge = (k < snaps.length - 1)
            ? snaps[k + 1].hinges[snaps[k + 1].hinges.length - 1]
            : null;
          const nextEl = nextHinge ? problem.elements[nextHinge.elem] : null;
          const nextNode = nextHinge
            ? problem.nodes[nextHinge.end === 0 ? nextEl.i : nextEl.j]
            : null;

          return (
            <div
              key={k}
              className="border-l-2 pl-3 mb-3 pb-1"
              style={{
                borderColor: isCurrent ? '#1a1a1a' : (isReached ? '#92897a' : '#d9d4ca'),
                opacity: isReached ? 1 : 0.5,
              }}
            >
              <div className="flex items-baseline justify-between gap-3 flex-wrap">
                <div className="text-sm">
                  <span className="font-semibold">Step {k}</span>
                  {k === 0 && <span className="text-stone-500 ml-2">— linear elastic, all locations intact</span>}
                  {k > 0 && formedHinge && (
                    <span className="text-stone-700 ml-2">
                      hinge formed at <span className="font-semibold">{formedEl.label || `e${formedHinge.elem}`}</span>
                      {' '}({formedEl.type}, end {formedNode.label || (formedHinge.end === 0 ? 'i' : 'j')},{' '}
                      <span style={{ color: formedHinge.sign > 0 ? '#1f7a3a' : '#9b5d2c' }}>
                        {formedHinge.sign > 0 ? '+Mp' : '−Mp'}
                      </span>)
                    </span>
                  )}
                </div>
                <div className="text-[11px] tabular-nums text-stone-600">
                  {prev && <span>ΔP = <span className="font-semibold">{dP.toFixed(4)}</span> · </span>}
                  P = <span className="font-semibold">{s.P.toFixed(4)}</span>
                  {problem.controlDof !== undefined && (
                    <span> · Δ = <span className="font-semibold">{points[k].delta.toExponential(2)}</span></span>
                  )}
                </div>
              </div>

              {/* D/C ratios — top 4 most-critical unhinged locations */}
              {k < snaps.length - 1 && (
                <div className="mt-1.5 text-[11px]">
                  <div className="text-[9px] uppercase tracking-widest text-stone-500 mb-0.5">
                    D/C ratios at this state · highest = next to yield
                  </div>
                  <table className="text-[11px] tabular-nums w-full">
                    <thead className="text-stone-500">
                      <tr>
                        <th className="text-left font-normal pr-2">Location</th>
                        <th className="text-right font-normal pr-2">|M| / Mp</th>
                        <th className="text-right font-normal pr-2">|M|</th>
                        <th className="text-right font-normal">Residual to Mp</th>
                      </tr>
                    </thead>
                    <tbody>
                      {dRatios.filter(r => !r.isHinged).slice(0, 4).map((r, i) => {
                        const isNext = nextHinge && r.elem === nextHinge.elem && r.end === nextHinge.end;
                        return (
                          <tr key={i} style={{
                            color: isNext ? '#c43e3e' : '#444',
                            fontWeight: isNext ? 600 : 400,
                          }}>
                            <td className="pr-2">
                              {r.elLabel} ({r.elType}, {r.nodeLabel}){isNext && ' ← yields next'}
                            </td>
                            <td className="text-right pr-2">{r.dc.toFixed(3)}</td>
                            <td className="text-right pr-2">{Math.abs(r.M).toFixed(3)}</td>
                            <td className="text-right">{r.residual.toFixed(3)}</td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
              )}

              {k === snaps.length - 1 && (
                <div className="mt-1 text-[11px] text-stone-700 italic">
                  Mechanism reached — K is singular, additional load increment cannot be carried.
                  Pu = <span className="font-semibold">{s.P.toFixed(4)}</span> {problem.units}.
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}

const TABS = ['P1A', 'P1B', 'P2', 'P3'];

function App() {
  const [tab, setTab] = useState('P1A');
  const [mode, setMode] = useState('proportional');
  const [individualCaseIdx, setIndividualCaseIdx] = useState(0);
  const [P_norm, setP_norm] = useState(0.0);
  const [deflectScaleAdj, setDeflectScaleAdj] = useState(1.0);
  const [logExpanded, setLogExpanded] = useState(true);
  const [connection, setConnection] = useState(ASCE41_DEFAULT_KEY);

  const problem = PROBLEMS[tab];

  // Build load case selection vector
  const loadSelection = useMemo(() => {
    const sel = new Array(problem.loadCases.length).fill(0);
    if (mode === 'proportional' || !problem.multiCase) {
      for (let i = 0; i < sel.length; i++) sel[i] = 1;
    } else {
      sel[Math.min(individualCaseIdx, sel.length - 1)] = 1;
    }
    return sel;
  }, [tab, mode, individualCaseIdx, problem.multiCase, problem.loadCases.length]);

  // Run analysis
  const analysis = useMemo(() => {
    return runIncrementalAnalysis(problem, loadSelection);
  }, [problem, loadSelection]);

  const Pu = analysis.Pu;
  const currentP = P_norm * Pu;
  const state = useMemo(() => interpolateState(analysis.snapshots, currentP), [analysis, currentP]);

  // Reset slider on tab change
  useEffect(() => {
    setP_norm(0);
  }, [tab, mode, individualCaseIdx]);

  // Hinge count, indeterminacy
  const hingesFormed = state.hinges.length;
  const totalHinges = analysis.snapshots.length - 1;
  const indet = state.indet;

  // Snapshot percentages for slider tick marks
  const tickPositions = useMemo(() => {
    if (!Pu || Pu === 0) return [];
    return analysis.snapshots.map((s) => Math.min(s.P / Pu, 1.2));
  }, [analysis, Pu]);

  // Determine the next forming hinge for status display
  const inSegmentNext =
    state.nextHingeIdx !== null && analysis.snapshots[state.nextHingeIdx + 1]
      ? analysis.snapshots[state.nextHingeIdx + 1]
      : null;

  return (
    <div
      className="w-full"
      style={{
        fontFamily: '"Inter Tight", "Helvetica Neue", system-ui, sans-serif',
        background: '#f7f4ec',
        minHeight: '100vh',
        color: '#1a1a1a',
      }}
    >
      <div className="max-w-7xl mx-auto px-5 py-4">
        {/* Header */}
        <header className="border-b border-stone-300 pb-3 mb-3">
          <div className="flex items-baseline gap-3 flex-wrap">
            <h1
              className="text-2xl tracking-tight"
              style={{
                fontFamily: '"Iowan Old Style", "Apple Garamond", "Baskerville", "Times New Roman", serif',
                fontWeight: 600,
                letterSpacing: '-0.01em',
              }}
            >
              Incremental Plastic Collapse Analyzer
            </h1>
            <span
              className="text-xs uppercase tracking-widest text-stone-500"
              style={{ fontFamily: 'ui-monospace, monospace' }}
            >
              ARCE 483 · HW#4 companion
            </span>
          </div>
          <p className="text-sm text-stone-600 mt-1">
            Direct stiffness, hinges form one at a time as load steps up. K singular ⇒ mechanism.
          </p>
        </header>

        {/* Tabs + Mode toggle */}
        <div className="flex flex-wrap items-center gap-2 mb-3">
          <div className="flex border border-stone-400 rounded-sm overflow-hidden">
            {TABS.map((t) => (
              <button
                key={t}
                onClick={() => setTab(t)}
                className={`px-3 py-1.5 text-sm transition-colors`}
                style={{
                  background: tab === t ? '#1a1a1a' : 'transparent',
                  color: tab === t ? '#f7f4ec' : '#1a1a1a',
                  fontFamily: 'ui-monospace, monospace',
                  borderRight: t !== 'P3' ? '1px solid #92897a' : 'none',
                  fontWeight: tab === t ? 600 : 400,
                }}
              >
                {t}
              </button>
            ))}
          </div>
          <span className="text-stone-500 text-sm">{problem.title}</span>

          {problem.pushoverEnabled && (
            <div className="flex items-center gap-2">
              <span className="text-xs uppercase tracking-widest text-stone-500">Connection</span>
              <select
                className="text-xs px-2 py-1 border border-stone-400 rounded-sm bg-white"
                value={connection}
                onChange={(e) => setConnection(e.target.value)}
                style={{ fontFamily: 'ui-monospace, monospace' }}
                title={ASCE41_CONNECTIONS[connection]?.note}
              >
                {Object.keys(ASCE41_CONNECTIONS).map((k) => (
                  <option key={k} value={k}>{k}</option>
                ))}
              </select>
            </div>
          )}

          {problem.multiCase && (
            <div className="ml-auto flex items-center gap-2">
              <span className="text-xs uppercase tracking-widest text-stone-500">Mode</span>
              <div className="flex border border-stone-400 rounded-sm overflow-hidden">
                <button
                  className="px-2.5 py-1 text-xs"
                  style={{
                    background: mode === 'proportional' ? '#1a1a1a' : 'transparent',
                    color: mode === 'proportional' ? '#f7f4ec' : '#1a1a1a',
                    borderRight: '1px solid #92897a',
                    fontFamily: 'ui-monospace, monospace',
                  }}
                  onClick={() => setMode('proportional')}
                >
                  Proportional
                </button>
                <button
                  className="px-2.5 py-1 text-xs"
                  style={{
                    background: mode === 'individual' ? '#1a1a1a' : 'transparent',
                    color: mode === 'individual' ? '#f7f4ec' : '#1a1a1a',
                    fontFamily: 'ui-monospace, monospace',
                  }}
                  onClick={() => setMode('individual')}
                >
                  Individual
                </button>
              </div>
              {mode === 'individual' && (
                <select
                  className="text-xs px-2 py-1 border border-stone-400 rounded-sm bg-white"
                  value={individualCaseIdx}
                  onChange={(e) => setIndividualCaseIdx(parseInt(e.target.value))}
                  style={{ fontFamily: 'ui-monospace, monospace' }}
                >
                  {problem.loadCases.map((lc, i) => (
                    <option key={i} value={i}>
                      {lc.name}
                    </option>
                  ))}
                </select>
              )}
            </div>
          )}
        </div>

        {/* Status row */}
        <div
          className="grid grid-cols-2 sm:grid-cols-4 gap-3 mb-3 px-3 py-2 border border-stone-300 bg-white rounded-sm"
          style={{ fontFamily: 'ui-monospace, monospace' }}
        >
          <div>
            <div className="text-[10px] uppercase tracking-widest text-stone-500">Load</div>
            <div className="text-base font-semibold tabular-nums">
              {currentP.toFixed(currentP < 10 ? 3 : currentP < 100 ? 2 : 1)}{' '}
              <span className="text-xs text-stone-500">{problem.units}</span>
            </div>
          </div>
          <div>
            <div className="text-[10px] uppercase tracking-widest text-stone-500">P / Pu</div>
            <div className="text-base font-semibold tabular-nums">{P_norm.toFixed(3)}</div>
          </div>
          <div>
            <div className="text-[10px] uppercase tracking-widest text-stone-500">Hinges</div>
            <div className="text-base font-semibold tabular-nums">
              {hingesFormed}{' '}
              <span className="text-xs text-stone-500">/ {totalHinges}</span>
            </div>
          </div>
          <div>
            <div className="text-[10px] uppercase tracking-widest text-stone-500">Indet.</div>
            <div className="text-base font-semibold tabular-nums">
              {indet}
              <span className="text-xs text-stone-500"> / {problem.indeterminacyInit}</span>
            </div>
          </div>
        </div>

        {/* Main visualization area */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-3 mb-3">
          <div className="border border-stone-300 bg-white rounded-sm relative">
            <div
              className="absolute top-1.5 left-2 text-[10px] uppercase tracking-widest text-stone-500"
              style={{ fontFamily: 'ui-monospace, monospace' }}
            >
              Structure · deflected ×{(() => {
                const v = problem.deflectScale * deflectScaleAdj;
                if (v >= 1) return v.toFixed(2);
                if (v >= 0.01) return v.toFixed(3);
                if (v >= 0.0001) return v.toFixed(5);
                return v.toExponential(1);
              })()}
            </div>
            <StructureView
              problem={problem}
              state={state}
              scale={loadSelection}
              snapshots={analysis.snapshots}
              currentP={currentP}
              Pu={Pu}
              deflectScaleAdj={deflectScaleAdj}
            />
          </div>
          <div className="border border-stone-300 bg-white rounded-sm relative">
            <div
              className="absolute top-1.5 left-2 text-[10px] uppercase tracking-widest text-stone-500"
              style={{ fontFamily: 'ui-monospace, monospace' }}
            >
              BMD · tension side
            </div>
            <BMDView
              problem={problem}
              state={state}
              scale={loadSelection}
              currentP={currentP}
              Pu={Pu}
            />
          </div>
        </div>

        {/* Pushover + Walkthrough — Tier 1+2+3+5, gated on problem.pushoverEnabled */}
        {problem.pushoverEnabled && (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-3 mb-3">
            <PushoverPanel
              problem={problem}
              analysis={analysis}
              currentP={currentP}
              P_norm={P_norm}
              setP_norm={setP_norm}
              connection={connection}
            />
            <WalkthroughPanel
              problem={problem}
              analysis={analysis}
              currentP={currentP}
            />
          </div>
        )}

        {/* Slider */}
        <div className="border border-stone-300 bg-white rounded-sm p-3 mb-3">
          <div className="flex items-baseline justify-between mb-1.5">
            <span
              className="text-[10px] uppercase tracking-widest text-stone-500"
              style={{ fontFamily: 'ui-monospace, monospace' }}
            >
              Load · 0 → 1.2 Pu
            </span>
            <span
              className="text-xs text-stone-600 tabular-nums"
              style={{ fontFamily: 'ui-monospace, monospace' }}
            >
              {(P_norm * 100).toFixed(1)}% of Pu
            </span>
          </div>
          <div className="relative">
            <input
              type="range"
              min="0"
              max="1.2"
              step="0.001"
              value={P_norm}
              onChange={(e) => setP_norm(parseFloat(e.target.value))}
              className="w-full h-2 cursor-pointer"
              style={{ accentColor: '#1a1a1a' }}
            />
            {/* Tick marks for hinge events */}
            <div className="relative h-3 mt-1">
              {tickPositions.slice(1).map((pos, k) => {
                const left = (pos / 1.2) * 100;
                if (pos > 1.2) return null;
                return (
                  <div
                    key={k}
                    className="absolute"
                    style={{
                      left: `${left}%`,
                      top: 0,
                      transform: 'translateX(-50%)',
                    }}
                    title={`Hinge ${k + 1}: P/Pu = ${pos.toFixed(3)}`}
                  >
                    <div
                      style={{
                        width: 1,
                        height: 10,
                        background: '#c43e3e',
                      }}
                    />
                  </div>
                );
              })}
              {/* 1.0 marker */}
              <div
                className="absolute"
                style={{
                  left: `${(1.0 / 1.2) * 100}%`,
                  top: 0,
                  transform: 'translateX(-50%)',
                }}
              >
                <div
                  style={{
                    width: 1.5,
                    height: 12,
                    background: '#1a1a1a',
                  }}
                />
                <div
                  className="absolute text-[9px] text-stone-700"
                  style={{
                    top: 12,
                    left: '50%',
                    transform: 'translateX(-50%)',
                    fontFamily: 'ui-monospace, monospace',
                  }}
                >
                  Pu
                </div>
              </div>
            </div>
          </div>
          {/* Snap-to controls */}
          <div className="flex flex-wrap items-center gap-1 mt-3">
            <span
              className="text-[10px] uppercase tracking-widest text-stone-500 mr-1"
              style={{ fontFamily: 'ui-monospace, monospace' }}
            >
              Snap →
            </span>
            <button
              className="text-[11px] px-2 py-0.5 border border-stone-300 hover:bg-stone-100"
              style={{ fontFamily: 'ui-monospace, monospace' }}
              onClick={() => setP_norm(0)}
            >
              0
            </button>
            {analysis.snapshots.slice(1).map((s, k) => (
              <button
                key={k}
                className="text-[11px] px-2 py-0.5 border border-stone-300 hover:bg-stone-100 tabular-nums"
                style={{ fontFamily: 'ui-monospace, monospace' }}
                onClick={() => setP_norm(Math.min(s.P / Pu, 1.2))}
                title={`Hinge ${k + 1}: ${s.P.toFixed(4)}`}
              >
                H{k + 1}
              </button>
            ))}
            <button
              className="text-[11px] px-2 py-0.5 border border-stone-300 hover:bg-stone-100"
              style={{ fontFamily: 'ui-monospace, monospace' }}
              onClick={() => setP_norm(1.0)}
            >
              Pu
            </button>
            <span className="ml-auto text-[10px] text-stone-500" style={{ fontFamily: 'ui-monospace, monospace' }}>
              defl ×
              <input
                type="number"
                value={deflectScaleAdj}
                onChange={(e) => setDeflectScaleAdj(parseFloat(e.target.value) || 1)}
                step="0.5"
                min="0.1"
                className="w-14 ml-1 px-1 py-0.5 border border-stone-300 tabular-nums"
                style={{ fontFamily: 'ui-monospace, monospace' }}
              />
            </span>
          </div>
        </div>

        {/* Hinge log + Pu comparison */}
        <div className="border border-stone-300 bg-white rounded-sm">
          <div
            className="flex items-baseline justify-between px-3 py-2 border-b border-stone-200 cursor-pointer"
            onClick={() => setLogExpanded((x) => !x)}
          >
            <span
              className="text-[10px] uppercase tracking-widest text-stone-500"
              style={{ fontFamily: 'ui-monospace, monospace' }}
            >
              Hinge formation log
            </span>
            <div
              className="flex items-baseline gap-3 text-xs tabular-nums"
              style={{ fontFamily: 'ui-monospace, monospace' }}
            >
              <span>
                Pu (incremental) ={' '}
                <span className="font-semibold">
                  {Pu.toFixed(Pu < 10 ? 4 : Pu < 100 ? 2 : 1)}
                </span>{' '}
                <span className="text-stone-500">{problem.units}</span>
              </span>
              {(mode === 'proportional' || !problem.multiCase) && (
                <span>
                  vs Pu (kinematic) ={' '}
                  <span className="font-semibold">
                    {problem.kinematicPu.toFixed(problem.kinematicPu < 10 ? 4 : problem.kinematicPu < 100 ? 2 : 1)}
                  </span>{' '}
                  <span
                    className="text-xs"
                    style={{
                      color:
                        Math.abs(Pu - problem.kinematicPu) / problem.kinematicPu < 0.005
                          ? '#1f7a3a'
                          : '#9b5d2c',
                    }}
                  >
                    {Math.abs(Pu - problem.kinematicPu) / problem.kinematicPu < 0.005
                      ? '✓'
                      : `Δ ${(((Pu - problem.kinematicPu) / problem.kinematicPu) * 100).toFixed(2)}%`}
                  </span>
                </span>
              )}
              <span className="text-stone-400">{logExpanded ? '▾' : '▸'}</span>
            </div>
          </div>
          {logExpanded && (
            <div className="max-h-72 overflow-auto">
              <table
                className="w-full text-xs tabular-nums"
                style={{ fontFamily: 'ui-monospace, monospace' }}
              >
                <thead className="bg-stone-100 text-stone-600">
                  <tr>
                    <th className="px-2 py-1 text-left">#</th>
                    <th className="px-2 py-1 text-right">P</th>
                    <th className="px-2 py-1 text-right">P / Pu</th>
                    <th className="px-2 py-1 text-left">Element</th>
                    <th className="px-2 py-1 text-left">End</th>
                    <th className="px-2 py-1 text-right">Sign</th>
                    <th className="px-2 py-1 text-right">|M|</th>
                  </tr>
                </thead>
                <tbody>
                  {analysis.hinges.length === 0 && (
                    <tr>
                      <td colSpan="7" className="px-2 py-3 text-stone-500 text-center">
                        No hinges yet. Increase load.
                      </td>
                    </tr>
                  )}
                  {analysis.hinges.map((h, k) => {
                    const el = problem.elements[h.elem];
                    const endNode = problem.nodes[h.end === 0 ? el.i : el.j];
                    const isPast = currentP >= h.P - 1e-9;
                    return (
                      <tr
                        key={k}
                        className="border-t border-stone-100"
                        style={{
                          color: isPast ? '#1a1a1a' : '#a0a0a0',
                          background: isPast ? 'transparent' : '#fafaf6',
                        }}
                      >
                        <td className="px-2 py-1">{k + 1}</td>
                        <td className="px-2 py-1 text-right">
                          {h.P.toFixed(h.P < 10 ? 4 : h.P < 100 ? 2 : 1)}
                        </td>
                        <td className="px-2 py-1 text-right">{(h.P / Pu).toFixed(3)}</td>
                        <td className="px-2 py-1">
                          {el.label || `e${h.elem}`}{' '}
                          <span className="text-stone-500">({el.type})</span>
                        </td>
                        <td className="px-2 py-1">
                          {endNode.label || (h.end === 0 ? 'i' : 'j')}
                        </td>
                        <td className="px-2 py-1 text-right">
                          {h.sign > 0 ? '+Mp' : '−Mp'}
                        </td>
                        <td className="px-2 py-1 text-right">{el.Mp.toFixed(2)}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          )}
        </div>

        {/* Footer */}
        <p className="text-[10px] text-stone-500 mt-3 text-center" style={{ fontFamily: 'ui-monospace, monospace' }}>
          Direct-stiffness FEM in JS · cubic Hermite deformed shapes · K singular detected by Gauss pivot &lt; rel tol
        </p>
      </div>
    </div>
  );
}

export default App;
