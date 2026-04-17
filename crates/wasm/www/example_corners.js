import { initWasm, loadImage, getScaledDimensions, canvasToMat, drawPoints } from './cv_demo_utils.js';

let sourceImage = null;
let cv = null;
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');

// UI Elements
const sliders = {
    maxCorners: document.getElementById('max-corners'),
    quality: document.getElementById('quality'),
    distance: document.getElementById('distance')
};
const displays = {
    maxCorners: document.getElementById('val-max-corners'),
    quality: document.getElementById('val-quality'),
    distance: document.getElementById('val-distance')
};

async function start() {
    try {
        cv = await initWasm();
        document.getElementById('loader').classList.add('hidden');

        // Initial default image
        sourceImage = await loadImage('https://raw.githubusercontent.com/opencv/opencv/master/samples/data/butterfly.jpg');
        processImage();
    } catch (err) {
        console.error("WASM Initialization failed:", err);
        document.getElementById('loader').innerHTML = `<p style="color:red">Error loading WASM: ${err.message}</p>`;
    }
}

function processImage() {
    if (!sourceImage || !cv) return;

    const { width, height } = getScaledDimensions(sourceImage, 1024);
    canvas.width = width;
    canvas.height = height;
    ctx.drawImage(sourceImage, 0, 0, width, height);

    // 1. Convert canvas to Mat
    const mat = canvasToMat(cv, canvas, ctx);

    // 2. Convert to Grayscale (required for many imgproc ops)
    const gray = cv.cvtColor(mat, cv.COLOR_RGBA2GRAY()); 
    const grayF32 = gray.convertTo("f32");

    // 3. Detect corners
    const maxCorners = parseInt(sliders.maxCorners.value);
    const qualityLevel = parseFloat(sliders.quality.value);
    const minDistance = parseFloat(sliders.distance.value);

    // goodFeaturesToTrack(src, maxCorners, qualityLevel, minDistance, blockSize, useHarris, harrisK)
    const corners = cv.goodFeaturesToTrack(grayF32, maxCorners, qualityLevel, minDistance, 3, false, 0.04);

    // 4. Visualization
    drawPoints(ctx, corners, '#00FF00', 4);

    // Cleanup
    mat.free();
    gray.free();
    grayF32.free();
}

// Event Listeners
Object.keys(sliders).forEach(key => {
    sliders[key].addEventListener('input', () => {
        displays[key].innerText = sliders[key].value;
        processImage();
    });
});

const fileInput = document.getElementById('file-input');
const dropZone = document.getElementById('drop-zone');

dropZone.onclick = () => fileInput.click();
fileInput.onchange = async (e) => {
    if (e.target.files.length > 0) {
        const url = URL.createObjectURL(e.target.files[0]);
        sourceImage = await loadImage(url);
        processImage();
    }
};

start();
