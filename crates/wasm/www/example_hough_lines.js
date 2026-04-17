import { initWasm, getScaledDimensions, canvasToMat, drawHoughLines, drawLineSegments } from './cv_demo_utils.js';

let cv = null;
let sourceImage = null;
let mode = 'standard'; // 'standard' or 'prob'

const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const fileInput = document.getElementById('file-input');
const dropZone = document.getElementById('drop-zone');
const loader = document.getElementById('loader');

const sliders = {
    thresh: document.getElementById('thresh'),
    rho: document.getElementById('rho'),
    minLen: document.getElementById('min-len'),
    canny: document.getElementById('canny')
};

const labels = {
    thresh: document.getElementById('val-thresh'),
    rho: document.getElementById('val-rho'),
    minLen: document.getElementById('val-min-len'),
    canny: document.getElementById('val-canny')
};

// Initialize WASM
async function start() {
    try {
        cv = await initWasm();
        loader.classList.add('hidden');
        console.log("WASM Initialized");
    } catch (e) {
        console.error("WASM Initialization failed:", e);
        alert("Failed to initialize WASM. Check console for details.");
    }
}

function processImage() {
    if (!sourceImage || !cv) return;

    const { width, height } = getScaledDimensions(sourceImage, 800);
    canvas.width = width;
    canvas.height = height;
    ctx.drawImage(sourceImage, 0, 0, width, height);

    const mat = canvasToMat(cv, canvas, ctx);
    
    // 1. Gray
    const gray = cv.cvtColor(mat, cv.COLOR_RGBA2GRAY()); 
    
    // 2. Blur (requires f32)
    const grayF32 = gray.convertTo("f32");
    const blurred = cv.gaussianBlur(grayF32, 3, 3, 0, 0, cv.BORDER_REFLECT_101());
    
    // 3. Canny (requires f32 input, returns u8)
    const cannyHigh = parseInt(sliders.canny.value);
    const edges = cv.canny(blurred, cannyHigh / 2, cannyHigh, 3, false);

    // 4. Hough Params
    const threshold = parseInt(sliders.thresh.value);
    const rho = parseFloat(sliders.rho.value);
    const theta = Math.PI / 180;

    // 5. Detect and Draw
    if (mode === 'prob') {
        const minLineLen = parseInt(sliders.minLen.value);
        const segments = cv.houghLinesP(edges, rho, theta, threshold, minLineLen, 10);
        drawLineSegments(ctx, segments, '#84fab0', 2);
    } else {
        const lines = cv.houghLines(edges, rho, theta, threshold);
        drawHoughLines(ctx, lines, width, height, '#84fab0', 2);
    }

    // Cleanup
    mat.free();
    gray.free();
    grayF32.free();
    blurred.free();
    edges.free();
}

// Algo switching
document.getElementById('btn-standard').onclick = function() {
    mode = 'standard';
    this.classList.add('active');
    document.getElementById('btn-prob').classList.remove('active');
    document.getElementById('standard-params').classList.remove('hidden');
    document.getElementById('prob-params').classList.add('hidden');
    processImage();
};

document.getElementById('btn-prob').onclick = function() {
    mode = 'prob';
    this.classList.add('active');
    document.getElementById('btn-standard').classList.remove('active');
    document.getElementById('prob-params').classList.remove('hidden');
    document.getElementById('standard-params').classList.add('hidden');
    processImage();
};

// Event Listeners
Object.keys(sliders).forEach(key => {
    sliders[key].oninput = function() {
        labels[key].innerText = this.value;
        processImage();
    };
});

fileInput.onchange = (e) => {
    const file = e.target.files[0];
    if (file) {
        const reader = new FileReader();
        reader.onload = (event) => {
            const img = new Image();
            img.onload = () => {
                sourceImage = img;
                processImage();
            };
            img.src = event.target.result;
        };
        reader.readAsDataURL(file);
    }
};

dropZone.onclick = () => fileInput.click();
dropZone.ondragover = (e) => {
    e.preventDefault();
    dropZone.style.borderColor = '#84fab0';
};
dropZone.ondragleave = () => {
    dropZone.style.borderColor = '#334155';
};
dropZone.ondrop = (e) => {
    e.preventDefault();
    dropZone.style.borderColor = '#334155';
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith('image/')) {
        const reader = new FileReader();
        reader.onload = (event) => {
            const img = new Image();
            img.onload = () => {
                sourceImage = img;
                processImage();
            };
            img.src = event.target.result;
        };
        reader.readAsDataURL(file);
    }
};

start();
