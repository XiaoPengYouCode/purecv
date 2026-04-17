import { initWasm, loadImage, getScaledDimensions, canvasToMat, matToCanvas } from './cv_demo_utils.js';

let sourceImage = null;
let cv = null;
const stack = document.getElementById('pyramid-stack');
const levelsSlider = document.getElementById('levels');
const levelsDisplay = document.getElementById('val-levels');

async function start() {
    try {
        cv = await initWasm();
        document.getElementById('loader').classList.add('hidden');

        sourceImage = await loadImage('https://raw.githubusercontent.com/opencv/opencv/master/samples/data/butterfly.jpg');
        processImage();
    } catch (err) {
        console.error("WASM Initialization failed:", err);
        document.getElementById('loader').innerHTML = `<p style="color:red">Error loading WASM: ${err.message}</p>`;
    }
}

function processImage() {
    if (!sourceImage || !cv) return;

    stack.innerHTML = '';

    const { width, height } = getScaledDimensions(sourceImage, 512);
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = width;
    tempCanvas.height = height;
    const tempCtx = tempCanvas.getContext('2d');
    tempCtx.drawImage(sourceImage, 0, 0, width, height);

    const mat = canvasToMat(cv, tempCanvas, tempCtx);
    const maxLevel = parseInt(levelsSlider.value);
    
    try {
        const levels = cv.buildPyramid(mat, maxLevel, cv.BORDER_REFLECT_101());

        levels.forEach((lvl, index) => {
            const levelBox = document.createElement('div');
            levelBox.className = 'level-box';
            
            const levelCanvas = document.createElement('canvas');
            matToCanvas(lvl, levelCanvas);
            
            const label = document.createElement('span');
            label.className = 'level-label';
            label.innerText = `L${index}: ${lvl.cols}x${lvl.rows}`;
            
            levelBox.appendChild(levelCanvas);
            levelBox.appendChild(label);
            stack.appendChild(levelBox);
            
            lvl.free();
        });
    } catch (e) {
        console.error("Pyramid error:", e);
    }

    mat.free();
}

levelsSlider.oninput = () => {
    levelsDisplay.innerText = levelsSlider.value;
    processImage();
};

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
