/*
 *  example_pose.js
 *  purecv
 *
 *  This file is part of purecv - WebARKit.
 *
 *  purecv is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU Lesser General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  purecv is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU Lesser General Public License for more details.
 *
 *  You should have received a copy of the GNU Lesser General Public License
 *  along with purecv.  If not, see <http://www.gnu.org/licenses/>.
 *
 *  As a special exception, the copyright holders of this library give you
 *  permission to link this library with independent modules to produce an
 *  executable, regardless of the license terms of these independent modules, and to
 *  copy and distribute the resulting executable under terms of your choice,
 *  provided that you also meet, for each linked independent module, the terms and
 *  conditions of the license of that module. An independent module is a module
 *  which is neither derived from nor based on this library. If you modify this
 *  library, you may extend this exception to your version of the library, but you
 *  are not obligated to do so. If you do not wish to do so, delete this exception
 *  statement from your version.
 *
 *  Copyright 2026 WebARKit.
 *
 *  Author(s): Walter Perdan @kalwalt https://github.com/kalwalt
 *
 */

import { initWasm } from './cv_demo_utils.js';

// ---------------------------------------------------------------------------
//  Constants
// ---------------------------------------------------------------------------

const CANVAS_SIZE = 500;
const MARKER_HALF = 5;       // half-size of the 3D marker (units) — 10×10 marker
const GRAB_RADIUS = 14;      // hit-test radius for corner dragging (px)

// Simulated camera intrinsics for a 500×500 "image"
const FX = 500, FY = 500, CX = 250, CY = 250;

// 3D object points: a planar marker lying on Z=0
// Two extra points are added to help the solver.
const OBJECT_POINTS_3D = [
    { x: -MARKER_HALF, y: -MARKER_HALF, z: 0 },  // top-left
    { x: MARKER_HALF, y: -MARKER_HALF, z: 0 },  // top-right
    { x: MARKER_HALF, y: MARKER_HALF, z: 0 },  // bottom-right
    { x: -MARKER_HALF, y: MARKER_HALF, z: 0 },  // bottom-left
    { x: 0, y: 0, z: 0 },   // centre
    { x: 0, y: MARKER_HALF, z: 0 },   // bottom-centre
];

// Default image-point positions — computed from a slightly tilted frontal
// view (camera at z≈25) so solvePnP produces a clean initial pose.
function defaultCorners() {
    return [
        { x: 155, y: 155 },  // TL
        { x: 345, y: 150 },  // TR
        { x: 350, y: 350 },  // BR
        { x: 150, y: 345 },  // BL
    ];
}

// Derive the 2 auxiliary image points from the 4 corners (centre + bottom-centre)
function auxPointsFromCorners(c) {
    return [
        {
            x: (c[0].x + c[1].x + c[2].x + c[3].x) / 4,
            y: (c[0].y + c[1].y + c[2].y + c[3].y) / 4
        },
        {
            x: (c[2].x + c[3].x) / 2,
            y: (c[2].y + c[3].y) / 2
        },
    ];
}

// ---------------------------------------------------------------------------
//  State
// ---------------------------------------------------------------------------

let cv = null;
let corners = defaultCorners();
let dragging = -1;  // index of corner being dragged, or -1

const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');

// ---------------------------------------------------------------------------
//  Drawing helpers
// ---------------------------------------------------------------------------

function drawScene() {
    ctx.clearRect(0, 0, CANVAS_SIZE, CANVAS_SIZE);

    // Grid background
    ctx.strokeStyle = 'rgba(255,255,255,0.04)';
    ctx.lineWidth = 1;
    for (let i = 0; i <= CANVAS_SIZE; i += 25) {
        ctx.beginPath(); ctx.moveTo(i, 0); ctx.lineTo(i, CANVAS_SIZE); ctx.stroke();
        ctx.beginPath(); ctx.moveTo(0, i); ctx.lineTo(CANVAS_SIZE, i); ctx.stroke();
    }

    // Semi-transparent filled quad
    ctx.fillStyle = 'rgba(79, 172, 254, 0.12)';
    ctx.beginPath();
    ctx.moveTo(corners[0].x, corners[0].y);
    for (let i = 1; i < 4; i++) ctx.lineTo(corners[i].x, corners[i].y);
    ctx.closePath();
    ctx.fill();

    // Quad outline
    ctx.strokeStyle = '#4facfe';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(corners[0].x, corners[0].y);
    for (let i = 1; i < 4; i++) ctx.lineTo(corners[i].x, corners[i].y);
    ctx.closePath();
    ctx.stroke();

    // Corner dots with labels
    const labels = ['TL', 'TR', 'BR', 'BL'];
    const colors = ['#22c55e', '#f59e0b', '#ef4444', '#a855f7'];
    corners.forEach((pt, i) => {
        ctx.fillStyle = colors[i];
        ctx.beginPath();
        ctx.arc(pt.x, pt.y, 8, 0, 2 * Math.PI);
        ctx.fill();

        ctx.fillStyle = '#fff';
        ctx.font = 'bold 11px Inter, system-ui, sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText(labels[i], pt.x, pt.y - 14);
    });

    // Crosshair at image centre
    ctx.strokeStyle = 'rgba(255,255,255,0.15)';
    ctx.lineWidth = 1;
    ctx.setLineDash([4, 4]);
    ctx.beginPath(); ctx.moveTo(CX, 0); ctx.lineTo(CX, CANVAS_SIZE); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(0, CY); ctx.lineTo(CANVAS_SIZE, CY); ctx.stroke();
    ctx.setLineDash([]);
}

// ---------------------------------------------------------------------------
//  Pose computation
// ---------------------------------------------------------------------------

function computePose() {
    if (!cv) return;

    const aux = auxPointsFromCorners(corners);
    const allImagePts = [...corners, ...aux]; // 6 points

    // Build Point3fVector
    const objPts = new cv.Point3fVector();
    OBJECT_POINTS_3D.forEach(p => objPts.push(p.x, p.y, p.z));

    // Build Point2fVector
    const imgPts = new cv.Point2fVector();
    allImagePts.forEach(p => imgPts.push(p.x, p.y));

    // Camera matrix (f64, 3×3)
    const camMat = cv.Mat.fromF64Data(3, 3, 1, new Float64Array([
        FX, 0, CX,
        0, FY, CY,
        0, 0, 1,
    ]));

    // Output rvec / tvec (3×1 f64)
    const rvec = cv.Mat.fromF64Data(3, 1, 1, new Float64Array([0, 0, 0]));
    const tvec = cv.Mat.fromF64Data(3, 1, 1, new Float64Array([0, 0, 0]));

    let pnpOk = false;
    try {
        // solvePnP(objPts, imgPts, camMat, distCoeffs, rvec, tvec, useGuess, flags)
        // flags=0 → Iterative
        pnpOk = cv.solvePnP(objPts, imgPts, camMat, undefined, rvec, tvec, false, 0);
    } catch (e) {
        console.warn('solvePnP failed:', e);
    }

    let rmatData = null;
    if (pnpOk) {
        // rodrigues: rvec → rotation matrix
        const rmat = cv.Mat.fromF64Data(3, 3, 1, new Float64Array(9));
        try {
            cv.rodrigues(rvec, rmat);
            rmatData = rmat.dataF64();
        } catch (e) {
            console.warn('rodrigues failed:', e);
        }
        rmat.free();
    }

    // findHomography (using the 4 corner correspondences only)
    const srcH = new cv.Point2fVector();
    const dstH = new cv.Point2fVector();
    // Source = default (un-dragged) marker corners
    const canon = [
        { x: 155, y: 155 }, { x: 345, y: 150 },
        { x: 350, y: 350 }, { x: 150, y: 345 },
    ];
    canon.forEach(p => srcH.push(p.x, p.y));
    corners.forEach(p => dstH.push(p.x, p.y));

    let homoData = null;
    try {
        // method=0 (None/DLT), threshold=3.0
        const result = cv.findHomography(srcH, dstH, 0, 3.0);
        if (result && result.homography) {
            homoData = result.homography.dataF64();
            result.homography.free();
        }
    } catch (e) {
        console.warn('findHomography failed:', e);
    }

    // Update the UI
    updateResultsUI(pnpOk, rvec, tvec, rmatData, homoData);

    // Cleanup WASM objects
    objPts.free();
    imgPts.free();
    camMat.free();
    rvec.free();
    tvec.free();
    srcH.free();
    dstH.free();
}

// ---------------------------------------------------------------------------
//  UI updates
// ---------------------------------------------------------------------------

function fmt(v) { return v.toFixed(4); }

function updateResultsUI(pnpOk, rvec, tvec, rmatData, homoData) {
    const badge = document.getElementById('status-badge');
    if (pnpOk) {
        badge.textContent = 'OK';
        badge.className = 'badge badge-ok';
    } else {
        badge.textContent = 'FAIL';
        badge.className = 'badge badge-fail';
    }

    if (pnpOk) {
        const rv = rvec.dataF64();
        const tv = tvec.dataF64();
        document.getElementById('rvec-display').textContent =
            `[${fmt(rv[0])},  ${fmt(rv[1])},  ${fmt(rv[2])}]`;
        document.getElementById('tvec-display').textContent =
            `[${fmt(tv[0])},  ${fmt(tv[1])},  ${fmt(tv[2])}]`;
    } else {
        document.getElementById('rvec-display').textContent = '— solve failed —';
        document.getElementById('tvec-display').textContent = '— solve failed —';
    }

    // Rotation matrix
    const rmatEl = document.getElementById('rmat-display');
    if (rmatData) {
        rmatEl.innerHTML = Array.from(rmatData).slice(0, 9).map(v =>
            `<span class="matrix-cell">${fmt(v)}</span>`
        ).join('');
    } else {
        rmatEl.innerHTML = Array(9).fill('<span class="matrix-cell">—</span>').join('');
    }

    // Homography
    const homoEl = document.getElementById('homo-display');
    if (homoData) {
        homoEl.innerHTML = Array.from(homoData).slice(0, 9).map(v =>
            `<span class="matrix-cell">${fmt(v)}</span>`
        ).join('');
    } else {
        homoEl.innerHTML = Array(9).fill('<span class="matrix-cell">—</span>').join('');
    }
}

// ---------------------------------------------------------------------------
//  Interaction: drag corners
// ---------------------------------------------------------------------------

function getMousePos(e) {
    const rect = canvas.getBoundingClientRect();
    const scaleX = CANVAS_SIZE / rect.width;
    const scaleY = CANVAS_SIZE / rect.height;
    return {
        x: (e.clientX - rect.left) * scaleX,
        y: (e.clientY - rect.top) * scaleY,
    };
}

canvas.addEventListener('pointerdown', (e) => {
    const pos = getMousePos(e);
    for (let i = 0; i < corners.length; i++) {
        const dx = pos.x - corners[i].x;
        const dy = pos.y - corners[i].y;
        if (Math.sqrt(dx * dx + dy * dy) < GRAB_RADIUS) {
            dragging = i;
            canvas.setPointerCapture(e.pointerId);
            break;
        }
    }
});

canvas.addEventListener('pointermove', (e) => {
    if (dragging < 0) return;
    const pos = getMousePos(e);
    corners[dragging].x = Math.max(10, Math.min(CANVAS_SIZE - 10, pos.x));
    corners[dragging].y = Math.max(10, Math.min(CANVAS_SIZE - 10, pos.y));
    drawScene();
    computePose();
});

canvas.addEventListener('pointerup', () => { dragging = -1; });
canvas.addEventListener('pointercancel', () => { dragging = -1; });

document.getElementById('reset-btn').addEventListener('click', () => {
    corners = defaultCorners();
    drawScene();
    computePose();
});

// ---------------------------------------------------------------------------
//  Bootstrap
// ---------------------------------------------------------------------------

async function start() {
    try {
        cv = await initWasm();
        document.getElementById('loader').classList.add('hidden');
        drawScene();
        computePose();
    } catch (err) {
        console.error("WASM initialisation failed:", err);
        document.getElementById('loader').innerHTML =
            `<p style="color:red">Error loading WASM: ${err.message}</p>`;
    }
}

start();
