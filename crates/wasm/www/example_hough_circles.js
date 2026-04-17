import { initWasm, loadImage, getScaledDimensions, canvasToMat, drawCircles } from './cv_demo_utils.js';

let sourceImage = null;
let cv = null;
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');

const sliders = {
    minDist: document.getElementById('min-dist'),
    p1: document.getElementById('p1'),
    p2: document.getElementById('p2'),
    minR: document.getElementById('min-r'),
    maxR: document.getElementById('max-r')
};

async function start() {
    try {
        cv = await initWasm();
        document.getElementById('loader').classList.add('hidden');

        sourceImage = await loadImage('https://raw.githubusercontent.com/opencv/opencv/master/samples/data/smarties.png');
        processImage();
    } catch (err) {
        console.error("WASM Initialization failed:", err);
        document.getElementById('loader').innerHTML = `<p style="color:red">Error loading WASM: ${err.message}</p>`;
    }
}

function processImage() {
    if (!sourceImage || !cv) return;

    const { width, height } = getScaledDimensions(sourceImage, 800);
    canvas.width = width;
    canvas.height = height;
    ctx.drawImage(sourceImage, 0, 0, width, height);

    const mat = canvasToMat(cv, canvas, ctx);
    
    // 1. Gray (RGBA u8 -> Gray u8)
    const gray = cv.cvtColor(mat, cv.COLOR_RGBA2GRAY()); 
    
    // 2. Conversion to f32 for medianBlur
    const grayF32 = gray.convertTo("f32");
    
    // 3. Blur (Median blur is good for circles)
    const blurredF32 = cv.medianBlur(grayF32, 5); 
    
    // 4. Conversion back to u8 for houghCircles
    const blurredU8 = blurredF32.convertTo("u8");

    // 5. Hough Circles
    const dp = 1.2;
    const minDist = parseFloat(sliders.minDist.value);
    const p1 = parseFloat(sliders.p1.value);
    const p2 = parseFloat(sliders.p2.value);
    const minR = parseInt(sliders.minR.value);
    const maxR = parseInt(sliders.maxR.value);

    const circles = cv.houghCircles(blurredU8, dp, minDist, p1, p2, minR, maxR);

    drawCircles(ctx, circles, '#FF3E3E', 3);

    mat.free();
    gray.free();
    grayF32.free();
    blurredF32.free();
    blurredU8.free();
}

// UI Listeners
Object.values(sliders).forEach(s => {
    s.oninput = () => {
        const display = document.getElementById('val-' + s.id);
        if (display) display.innerText = s.value;
        processImage();
    };
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
