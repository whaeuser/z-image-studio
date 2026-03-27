/**
 * Canvas-based mask editor for inpainting.
 * Renders a source image with a semi-transparent red mask overlay.
 * Supports brush and eraser tools with adjustable size.
 */
class MaskEditor {
    /**
     * @param {HTMLCanvasElement} canvas - The canvas element to draw on.
     * @param {HTMLImageElement} sourceImage - The source image to paint over.
     */
    constructor(canvas, sourceImage) {
        this.canvas = canvas;
        this.ctx = canvas.getContext('2d');
        this.sourceImage = sourceImage;

        // Scale to fit container while maintaining aspect ratio
        const container = canvas.parentElement;
        const maxWidth = container.clientWidth || 800;
        const scale = Math.min(1, maxWidth / sourceImage.width);
        this.displayWidth = Math.round(sourceImage.width * scale);
        this.displayHeight = Math.round(sourceImage.height * scale);

        canvas.width = this.displayWidth;
        canvas.height = this.displayHeight;

        // Create offscreen mask canvas (same display dimensions)
        this.maskCanvas = document.createElement('canvas');
        this.maskCanvas.width = this.displayWidth;
        this.maskCanvas.height = this.displayHeight;
        this.maskCtx = this.maskCanvas.getContext('2d');
        this.maskCtx.fillStyle = 'black';
        this.maskCtx.fillRect(0, 0, this.displayWidth, this.displayHeight);

        this.tool = 'brush'; // 'brush' or 'eraser'
        this.brushSize = 30;
        this.isDrawing = false;
        this.lastX = null;
        this.lastY = null;

        this._bindEvents();
        this._render();
    }

    /** Set the active tool ('brush' or 'eraser'). */
    setTool(tool) {
        this.tool = tool;
    }

    /** Set the brush/eraser radius in pixels. */
    setBrushSize(size) {
        this.brushSize = size;
    }

    /** Clear the entire mask. */
    clearMask() {
        this.maskCtx.fillStyle = 'black';
        this.maskCtx.fillRect(0, 0, this.displayWidth, this.displayHeight);
        this._render();
    }

    /**
     * Export the mask as a base64-encoded PNG string (without data URL prefix).
     * White = areas to regenerate, black = areas to keep.
     * Exported at full source image resolution.
     */
    getMaskBase64() {
        // Scale mask up to original image resolution for export
        const exportCanvas = document.createElement('canvas');
        exportCanvas.width = this.sourceImage.naturalWidth;
        exportCanvas.height = this.sourceImage.naturalHeight;
        const exportCtx = exportCanvas.getContext('2d');
        exportCtx.drawImage(this.maskCanvas, 0, 0, exportCanvas.width, exportCanvas.height);
        return exportCanvas.toDataURL('image/png').split(',')[1];
    }

    /** Clean up event listeners. */
    destroy() {
        this.canvas.removeEventListener('mousedown', this._onMouseDown);
        this.canvas.removeEventListener('mousemove', this._onMouseMove);
        this.canvas.removeEventListener('mouseup', this._onMouseUp);
        this.canvas.removeEventListener('mouseleave', this._onMouseUp);
        this.canvas.removeEventListener('touchstart', this._onTouchStart);
        this.canvas.removeEventListener('touchmove', this._onTouchMove);
        this.canvas.removeEventListener('touchend', this._onTouchEnd);
    }

    // --- Private methods ---

    _bindEvents() {
        // Mouse events
        this._onMouseDown = (e) => {
            this.isDrawing = true;
            const pos = this._getPos(e);
            this.lastX = pos.x;
            this.lastY = pos.y;
            this._drawAt(pos.x, pos.y);
        };
        this._onMouseMove = (e) => {
            if (!this.isDrawing) return;
            const pos = this._getPos(e);
            this._drawLine(this.lastX, this.lastY, pos.x, pos.y);
            this.lastX = pos.x;
            this.lastY = pos.y;
        };
        this._onMouseUp = () => {
            this.isDrawing = false;
            this.lastX = null;
            this.lastY = null;
        };

        this.canvas.addEventListener('mousedown', this._onMouseDown);
        this.canvas.addEventListener('mousemove', this._onMouseMove);
        this.canvas.addEventListener('mouseup', this._onMouseUp);
        this.canvas.addEventListener('mouseleave', this._onMouseUp);

        // Touch events
        this._onTouchStart = (e) => {
            e.preventDefault();
            const touch = e.touches[0];
            const pos = this._getTouchPos(touch);
            this.isDrawing = true;
            this.lastX = pos.x;
            this.lastY = pos.y;
            this._drawAt(pos.x, pos.y);
        };
        this._onTouchMove = (e) => {
            e.preventDefault();
            if (!this.isDrawing) return;
            const touch = e.touches[0];
            const pos = this._getTouchPos(touch);
            this._drawLine(this.lastX, this.lastY, pos.x, pos.y);
            this.lastX = pos.x;
            this.lastY = pos.y;
        };
        this._onTouchEnd = (e) => {
            e.preventDefault();
            this.isDrawing = false;
            this.lastX = null;
            this.lastY = null;
        };

        this.canvas.addEventListener('touchstart', this._onTouchStart, { passive: false });
        this.canvas.addEventListener('touchmove', this._onTouchMove, { passive: false });
        this.canvas.addEventListener('touchend', this._onTouchEnd, { passive: false });
    }

    _getPos(e) {
        const rect = this.canvas.getBoundingClientRect();
        return {
            x: e.clientX - rect.left,
            y: e.clientY - rect.top,
        };
    }

    _getTouchPos(touch) {
        const rect = this.canvas.getBoundingClientRect();
        return {
            x: touch.clientX - rect.left,
            y: touch.clientY - rect.top,
        };
    }

    _drawAt(x, y) {
        const color = this.tool === 'brush' ? 'white' : 'black';
        this.maskCtx.fillStyle = color;
        this.maskCtx.beginPath();
        this.maskCtx.arc(x, y, this.brushSize / 2, 0, Math.PI * 2);
        this.maskCtx.fill();
        this._render();
    }

    _drawLine(x1, y1, x2, y2) {
        const color = this.tool === 'brush' ? 'white' : 'black';
        this.maskCtx.strokeStyle = color;
        this.maskCtx.lineWidth = this.brushSize;
        this.maskCtx.lineCap = 'round';
        this.maskCtx.lineJoin = 'round';
        this.maskCtx.beginPath();
        this.maskCtx.moveTo(x1, y1);
        this.maskCtx.lineTo(x2, y2);
        this.maskCtx.stroke();

        // Also draw circle at endpoint for smooth coverage
        this.maskCtx.fillStyle = color;
        this.maskCtx.beginPath();
        this.maskCtx.arc(x2, y2, this.brushSize / 2, 0, Math.PI * 2);
        this.maskCtx.fill();

        this._render();
    }

    /** Composite source image + semi-transparent red mask overlay onto the visible canvas. */
    _render() {
        const ctx = this.ctx;
        // Draw source image
        ctx.drawImage(this.sourceImage, 0, 0, this.displayWidth, this.displayHeight);

        // Overlay mask with semi-transparent red where mask is white
        ctx.save();
        ctx.globalAlpha = 0.4;
        ctx.globalCompositeOperation = 'source-atop';

        // Create a temporary canvas to colorize the mask
        const tempCanvas = document.createElement('canvas');
        tempCanvas.width = this.displayWidth;
        tempCanvas.height = this.displayHeight;
        const tempCtx = tempCanvas.getContext('2d');

        // Draw mask
        tempCtx.drawImage(this.maskCanvas, 0, 0);

        // Get pixel data and colorize white -> red
        const imageData = tempCtx.getImageData(0, 0, this.displayWidth, this.displayHeight);
        const data = imageData.data;
        for (let i = 0; i < data.length; i += 4) {
            if (data[i] > 128) { // White mask pixel
                data[i] = 255;     // R
                data[i + 1] = 0;   // G
                data[i + 2] = 0;   // B
                data[i + 3] = 255; // A
            } else {
                data[i + 3] = 0; // Transparent for non-mask areas
            }
        }
        tempCtx.putImageData(imageData, 0, 0);

        ctx.restore();
        // Draw the red overlay on top
        ctx.globalAlpha = 0.4;
        ctx.drawImage(tempCanvas, 0, 0);
        ctx.globalAlpha = 1.0;
    }
}

// Export for use in main.js
window.MaskEditor = MaskEditor;
