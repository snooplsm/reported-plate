/**
 * Render prediction boxes
 * @param {HTMLCanvasElement} canvas canvas tag reference
 * @param {Array[Object]} boxes boxes array
 */
export const renderBoxes = (ctx, boxes) => {
  // font configs
  const font = `${Math.max(
    Math.round(Math.max(ctx.canvas.width, ctx.canvas.height) / 45),
    12
  )}px Arial`;
  ctx.font = font;
  ctx.textBaseline = "top";

  boxes.forEach((box) => {
    const klass = box.label;
    const color = box.color;
    const score = (box.probability * 100).toFixed(1);
    const [x1, y1, width, height] = box.bounding;

    // draw border box
    ctx.strokeStyle = color;
    ctx.lineWidth = Math.max(Math.min(ctx.canvas.width, ctx.canvas.height) / 200, 2.5);
    ctx.strokeRect(x1, y1, width, height);

    // draw the label background.
    ctx.fillStyle = color;
    const textWidth = ctx.measureText(klass + " - " + score + "%").width;
    const textHeight = parseInt(font, 10); // base 10
    const yText = y1 - (textHeight + ctx.lineWidth);
    ctx.fillRect(
      x1 - 1,
      yText < 0 ? 0 : yText,
      textWidth + ctx.lineWidth,
      textHeight + ctx.lineWidth
    );

    // Draw labels
    ctx.fillStyle = "#ffffff";
    ctx.fillText(klass + " - " + score + "%", x1 - 1, yText < 0 ? 1 : yText + 1);
  });
};

export class Colors {
  palette: string[];
  n: number;
  // ultralytics color palette https://ultralytics.com/
  constructor() {
    this.palette = [
      "#3f1651",
      "#f89f5b",
      "#ff701f",
      "#ffb21d",
      "#cfd231",
      "#48f90a",
      "#92cc17",
      "#3ddb86",
      "#1a9334",
      "#00d4bb",
      "#2c99a8",
      "#00c2ff",
      "#344593",
      "#6473ff",
      "#0018ec",
      "#8438ff",
      "#520085",
      "#cb38ff",
      "#ff95c8",
      "#ff37c7",
    ];
    
    this.n = this.palette.length;
  }

  get = (i) => this.palette[Math.floor(i) % this.n];

  hexToRgba = (hex, alpha):[number,number,number,number] => {
    var result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex)!;
    return [parseInt(result[1], 16), parseInt(result[2], 16), parseInt(result[3], 16), alpha]
  };

  toHex = (rgba) => {
    const [r,g,b] = rgba
    const hex = "#" + [r, g, b].map(value => value.toString(16).padStart(2, "0")).join("")
    return hex
  }

  toIndex = (hex) => {    
    return this.palette.indexOf(hex)
  };
}
