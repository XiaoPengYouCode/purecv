/**
 * PureCV WASM Demo Utilities
 * Shared helpers for image processing, canvas management, and drawing.
 */

/**
 * Initializes the WASM module.
 * Prefers dist-simd if available, otherwise dist-std.
 */
export async function initWasm() {
    let module;
    try {
        // Try SIMD first (most modern browsers)
        module = await import('../pkg/dist-simd/purecv_wasm.js');
    } catch (e) {
        console.warn("SIMD build not found or not supported, falling back to std", e);
        module = await import('../pkg/dist-std/purecv_wasm.js');
    }
    
    await module.default(); // Initialize WASM memory
    if (module.init_purecv) module.init_purecv();
    
    return module;
}

/**
 * Loads an image from a URL or File object.
 */
export function loadImage(src) {
    return new Promise((resolve, reject) => {
        const img = new Image();
        img.crossOrigin = 'anonymous'; // Avoid CORS issues
        img.onload = () => resolve(img);
        img.onerror = () => reject(new Error(`Failed to load image: ${src}`));
        img.src = src;
    });
}

/**
 * Resizes an image to fit within maxDimension while preserving aspect ratio.
 */
export function getScaledDimensions(img, maxDimension = 1024) {
    let width = img.width;
    let height = img.height;

    if (width > height) {
        if (width > maxDimension) {
            height *= maxDimension / width;
            width = maxDimension;
        }
    } else {
        if (height > maxDimension) {
            width *= maxDimension / height;
            height = maxDimension;
        }
    }
    return { width: Math.round(width), height: Math.round(height) };
}

/**
 * Converts a Canvas context to a PureCV Mat.
 */
export function canvasToMat(cv, canvas, ctx) {
    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    // fromU8Data(rows, cols, channels, data)
    return cv.Mat.fromU8Data(canvas.height, canvas.width, 4, imageData.data);
}

/**
 * Renders a PureCV Mat to a Canvas.
 */
export function matToCanvas(mat, canvas) {
    const ctx = canvas.getContext('2d');
    const w = mat.cols;
    const h = mat.rows;
    canvas.width = w;
    canvas.height = h;
    
    let data;
    if (mat.channels === 4 && mat.depth === 'u8') {
        data = mat.dataU8();
    } else {
        // Manual conversion to RGBA for display
        const ch = mat.channels;
        const depth = mat.depth;
        const raw = (depth === 'u8') ? mat.dataU8() : mat.dataF32();
        data = new Uint8Array(w * h * 4);
        
        for (let i = 0; i < w * h; i++) {
            if (ch === 1) {
                const v = raw[i];
                data[i * 4] = v;
                data[i * 4 + 1] = v;
                data[i * 4 + 2] = v;
                data[i * 4 + 3] = 255;
            } else if (ch === 3) {
                data[i * 4] = raw[i * 3];
                data[i * 4 + 1] = raw[i * 3 + 1];
                data[i * 4 + 2] = raw[i * 3 + 2];
                data[i * 4 + 3] = 255;
            } else if (ch === 4) {
                data[i * 4] = raw[i * 4];
                data[i * 4 + 1] = raw[i * 4 + 1];
                data[i * 4 + 2] = raw[i * 4 + 2];
                data[i * 4 + 3] = raw[i * 4 + 3];
            }
        }
    }
    
    const imageData = new ImageData(new Uint8ClampedArray(data), w, h);
    ctx.putImageData(imageData, 0, 0);
}

/**
 * Drawing helper: Points (Corners)
 */
export function drawPoints(ctx, points, color = '#00FF00', radius = 3) {
    ctx.fillStyle = color;
    for (let i = 0; i < points.length; i += 2) {
        ctx.beginPath();
        ctx.arc(points[i], points[i+1], radius, 0, 2 * Math.PI);
        ctx.fill();
    }
}

/**
 * Drawing helper: Lines (Standard Hough: rho, theta)
 */
export function drawHoughLines(ctx, lines, width, height, color = '#FF0000', thickness = 2) {
    ctx.strokeStyle = color;
    ctx.lineWidth = thickness;
    for (let i = 0; i < lines.length; i += 2) {
        const rho = lines[i];
        const theta = lines[i+1];
        const a = Math.cos(theta);
        const b = Math.sin(theta);
        const x0 = a * rho;
        const y0 = b * rho;
        const x1 = Math.round(x0 + 2000 * (-b));
        const y1 = Math.round(y0 + 2000 * (a));
        const x2 = Math.round(x0 - 2000 * (-b));
        const y2 = Math.round(y0 - 2000 * (a));

        ctx.beginPath();
        ctx.moveTo(x1, y1);
        ctx.lineTo(x2, y2);
        ctx.stroke();
    }
}

/**
 * Drawing helper: Line Segments (Probabilistic Hough: x1, y1, x2, y2)
 */
export function drawLineSegments(ctx, segments, color = '#FF0000', thickness = 2) {
    ctx.strokeStyle = color;
    ctx.lineWidth = thickness;
    for (let i = 0; i < segments.length; i += 4) {
        ctx.beginPath();
        ctx.moveTo(segments[i], segments[i+1]);
        ctx.lineTo(segments[i+2], segments[i+3]);
        ctx.stroke();
    }
}

/**
 * Drawing helper: Circles (cx, cy, r)
 */
export function drawCircles(ctx, circles, color = '#0000FF', thickness = 2) {
    ctx.strokeStyle = color;
    ctx.lineWidth = thickness;
    for (let i = 0; i < circles.length; i += 3) {
        ctx.beginPath();
        ctx.arc(circles[i], circles[i+1], circles[i+2], 0, 2 * Math.PI);
        ctx.stroke();
    }
}
