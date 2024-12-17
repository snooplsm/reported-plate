import cv from "@techstark/opencv-js";
import { Tensor } from "onnxruntime-web";
import { renderBoxes, Colors } from "./renderBox";
// import labels from "./labels.json";
import labels from "./labels-v13.json";

const colors = new Colors();
const numClass = labels.length;

// Create Maps using the utility function
const infractions = ['crosswalk', 'bike lane']
const plates = ['ny', 'pa', 'nj', 'ct', 'md', 'nc', 'tlc']
const taxi = ['tlc', 'medallion'];
const bonus = ['bullbars'];

console.log(infractions)
/**
 * Detect Image
 * @param {HTMLImageElement} image Image to detect
 * @param {HTMLCanvasElement} canvas canvas to draw boxes
 * @param {ort.InferenceSession} session YOLOv8 onnxruntime session
 * @param {Number} topk Integer representing the maximum number of boxes to be selected per class
 * @param {Number} iouThreshold Float representing the threshold for deciding whether boxes overlap too much with respect to IOU
 * @param {Number} scoreThreshold Float representing the threshold for deciding when to remove boxes based on score
 * @param {Number[]} inputShape model input shape. Normally in YOLO model [batch, channels, width, height]
 */
export const detectImage = async (
  image,
  canvas,
  session,
  topk,
  iouThreshold,
  scoreThreshold,
  inputShape
) => {
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height); // clean canvas

  const [modelWidth, modelHeight] = inputShape.slice(2);
  const maxSize = Math.max(modelWidth, modelHeight); // max size in input model
  const [input, xRatio, yRatio] = preprocessing(image, modelWidth, modelHeight); // preprocess frame

  const tensor = new Tensor("float32", input.data32F, inputShape); // to ort.Tensor
  const config = new Tensor(
    "float32",
    new Float32Array([
      numClass, // num class
      topk, // topk per class
      iouThreshold, // iou threshold
      scoreThreshold, // score threshold
    ])
  ); // nms config tensor
  const { output0, output1 } = await session.net.run({ images: tensor }); // run session and get output layer. out1: detect layer, out2: seg layer
  const { selected } = await session.nms.run({ detection: output0, config: config }); // perform nms and filter boxes

  const boxes = []; // ready to draw boxes
  let overlay = new Tensor("uint8", new Uint8Array(modelHeight * modelWidth * 4), [
    modelHeight,
    modelWidth,
    4,
  ]); // create overlay to draw segmentation object

  // looping through output
  for (let idx = 0; idx < selected.dims[1]; idx++) {
    const data = selected.data.slice(idx * selected.dims[2], (idx + 1) * selected.dims[2]); // get rows
    let box = data.slice(0, 4); // det boxes
    const scores = data.slice(4, 4 + numClass); // det classes probability scores
    const score = Math.max(...scores); // maximum probability scores
    const label = scores.indexOf(score); // class id of maximum probability scores
    const color = colors.get(label); // get color
    // const infraction = infractions.get(label)
    // const plate = plates.get(label)
    // console.log(label)
    // const medallion = taxi.get(label)
    // const bullbars = bonus.get(label)
    // console.log("infraction", infraction, "plate", plate, "medallion", medallion, "bullbars", bullbars)

    box = overflowBoxes(
      [
        box[0] - 0.5 * box[2], // before upscale x
        box[1] - 0.5 * box[3], // before upscale y
        box[2], // before upscale w
        box[3], // before upscale h
      ],
      maxSize
    ); // keep boxes in maxSize range

    const [x, y, w, h] = overflowBoxes(
      [
        Math.floor(box[0] * xRatio), // upscale left
        Math.floor(box[1] * yRatio), // upscale top
        Math.floor(box[2] * xRatio), // upscale width
        Math.floor(box[3] * yRatio), // upscale height
      ],
      maxSize
    ); // upscale boxes

    boxes.push({
      label: labels[label],
      probability: score,
      color: color,
      bounding: [x, y, w, h], // upscale box
    }); // update boxes to draw later

    boxes.sort((a, b) => {
      const areaA = a.bounding[2] * a.bounding[3]; // Width * Height of box A
      const areaB = b.bounding[2] * b.bounding[3]; // Width * Height of box B
      
      const scoreA = a.probability * areaA; // Combined score for box A
      const scoreB = b.probability * areaB; // Combined score for box B
    
      return scoreB - scoreA; // Sort in descending order
    });

    console.log(boxes)
    

    const mask = new Tensor(
      "float32",
      new Float32Array([
        ...box, // original scale box
        ...data.slice(4 + numClass), // mask data
      ])
    ); // mask input
    const maskConfig = new Tensor(
      "float32",
      new Float32Array([
        maxSize,
        x, // upscale x
        y, // upscale y
        w, // upscale width
        h, // upscale height
        ...Colors.hexToRgba(color, 120), // color in RGBA
      ])
    ); // mask config
    const { mask_filter } = await session.mask.run({
      detection: mask,
      mask: output1,
      config: maskConfig,
      overlay: overlay,
    }); // perform post-process to get mask

    overlay = mask_filter; // update overlay with the new one
  }
  
  const mask_img = new ImageData(new Uint8ClampedArray(overlay.data), modelHeight, modelWidth); // create image data from mask overlay
  
  // saveSegmentedImage(ctx, image,overlay,modelHeight,modelWidth, xRatio, yRatio)

  ctx.putImageData(mask_img, 0, 0); // put overlay to canvas
  // TODO RESTORE
  // const link = document.createElement("a");
  // link.download = "segmented_mask_pixels2.png";
  // link.href = canvas.toDataURL("image/png");
  // link.click();

  console.log(image.width, image.height, modelWidth, modelHeight)

  // const img = new Image()
  // img.src = image.src
  // img.onload = function() {
  //   console.log(img.width,img.height)
    // copyPixelsFromMask(img, overlay, modelWidth, modelHeight)
    renderBoxes(ctx, boxes); // draw boxes after overlay added to canvas

    input.delete(); // delete unused Mat
  // }

 

  
};

function copyPixelsFromMask(image, overlay, modelWidth, modelHeight) {
  // Create a canvas for the scaled mask
  const maskCanvas = document.createElement("canvas");
  const maskCtx = maskCanvas.getContext("2d");
  maskCanvas.width = image.width;
  maskCanvas.height = image.height;

  // Create a canvas for the original mask (model size)
  const originalMaskCanvas = document.createElement("canvas");
  const originalMaskCtx = originalMaskCanvas.getContext("2d");
  originalMaskCanvas.width = modelWidth;
  originalMaskCanvas.height = modelHeight;

  // Create ImageData from the overlay tensor
  const maskImgData = new ImageData(
    new Uint8ClampedArray(overlay.data),
    modelWidth,
    modelHeight
  );

  // Draw the mask on the original mask canvas
  originalMaskCtx.putImageData(maskImgData, 0, 0);

  // Scale the mask to match the original image dimensions
  maskCtx.drawImage(
    originalMaskCanvas,
    0, 0, modelWidth, modelHeight, // Source dimensions
    0, 0, image.width, image.height // Destination dimensions
  );

  // Get the scaled mask data
  const scaledMaskData = maskCtx.getImageData(0, 0, image.width, image.height);

  // Create a canvas for the result
  const resultCanvas = document.createElement("canvas");
  const resultCtx = resultCanvas.getContext("2d");
  resultCanvas.width = image.width;
  resultCanvas.height = image.height;

  // Get the original image data
  const imageCtx = document.createElement("canvas").getContext("2d");
  imageCtx.canvas.width = image.width;
  imageCtx.canvas.height = image.height;
  imageCtx.drawImage(image, 0, 0);
  const originalImageData = imageCtx.getImageData(0, 0, image.width, image.height);

  // Create a new ImageData object for the result
  const resultImageData = resultCtx.createImageData(image.width, image.height);

  // Copy only the pixels within the mask
  for (let i = 0; i < scaledMaskData.data.length; i += 4) {
    if (scaledMaskData.data[i + 3] > 0) { // Check if mask pixel is not transparent
      resultImageData.data[i] = originalImageData.data[i];     // Red
      resultImageData.data[i + 1] = originalImageData.data[i + 1]; // Green
      resultImageData.data[i + 2] = originalImageData.data[i + 2]; // Blue
      resultImageData.data[i + 3] = originalImageData.data[i + 3]; // Alpha
    } else {
      // Set transparent for pixels outside the mask
      resultImageData.data[i] = 0;
      resultImageData.data[i + 1] = 0;
      resultImageData.data[i + 2] = 0;
      resultImageData.data[i + 3] = 0;
    }
  }

  // Put the result data back onto the canvas
  resultCtx.putImageData(resultImageData, 0, 0);

  // Save the final result as a PNG
  const link = document.createElement("a");
  link.download = "masked_pixels.png";
  link.href = resultCanvas.toDataURL("image/png");
  link.click();
}

// Helper function to calculate bounding box of the mask
function calculateBoundingBox(maskData, width, height) {
  let minX = width, minY = height, maxX = 0, maxY = 0;
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const alpha = maskData[(y * width + x) * 4 + 3];
      if (alpha > 0) {
        minX = Math.min(minX, x);
        minY = Math.min(minY, y);
        maxX = Math.max(maxX, x);
        maxY = Math.max(maxY, y);
      }
    }
  }

  return {
    x: minX,
    y: minY,
    width: maxX - minX + 1,
    height: maxY - minY + 1,
  };
}

/**
 * Preprocessing image
 * @param {HTMLImageElement} source image source
 * @param {Number} modelWidth model input width
 * @param {Number} modelHeight model input height
 * @param {Number} stride model stride
 * @return preprocessed image and configs
 */
const preprocessing = (source, modelWidth, modelHeight, stride = 32) => {
  const mat = cv.imread(source); // read from img tag
  const matC3 = new cv.Mat(mat.rows, mat.cols, cv.CV_8UC3); // new image matrix
  cv.cvtColor(mat, matC3, cv.COLOR_RGBA2BGR); // RGBA to BGR

  const [w, h] = divStride(stride, matC3.cols, matC3.rows);
  cv.resize(matC3, matC3, new cv.Size(w, h));

  // padding image to [n x n] dim
  const maxSize = Math.max(matC3.rows, matC3.cols); // get max size from width and height
  const xPad = maxSize - matC3.cols, // set xPadding
    xRatio = maxSize / matC3.cols; // set xRatio
  const yPad = maxSize - matC3.rows, // set yPadding
    yRatio = maxSize / matC3.rows; // set yRatio
  const matPad = new cv.Mat(); // new mat for padded image
  cv.copyMakeBorder(matC3, matPad, 0, yPad, 0, xPad, cv.BORDER_CONSTANT); // padding black

  const input = cv.blobFromImage(
    matPad,
    1 / 255.0, // normalize
    new cv.Size(modelWidth, modelHeight), // resize to model input size
    new cv.Scalar(0, 0, 0),
    true, // swapRB
    false // crop
  ); // preprocessing image matrix

  // release mat opencv
  mat.delete();
  matC3.delete();
  matPad.delete();

  return [input, xRatio, yRatio];
};

/**
 * Get divisible image size by stride
 * @param {Number} stride
 * @param {Number} width
 * @param {Number} height
 * @returns {Number[2]} image size [w, h]
 */
const divStride = (stride, width, height) => {
  if (width % stride !== 0) {
    if (width % stride >= stride / 2) width = (Math.floor(width / stride) + 1) * stride;
    else width = Math.floor(width / stride) * stride;
  }
  if (height % stride !== 0) {
    if (height % stride >= stride / 2) height = (Math.floor(height / stride) + 1) * stride;
    else height = Math.floor(height / stride) * stride;
  }
  return [width, height];
};

/**
 * Handle overflow boxes based on maxSize
 * @param {Number[4]} box box in [x, y, w, h] format
 * @param {Number} maxSize
 * @returns non overflow boxes
 */
const overflowBoxes = (box, maxSize) => {
  box[0] = box[0] >= 0 ? box[0] : 0;
  box[1] = box[1] >= 0 ? box[1] : 0;
  box[2] = box[0] + box[2] <= maxSize ? box[2] : maxSize - box[0];
  box[3] = box[1] + box[3] <= maxSize ? box[3] : maxSize - box[1];
  return box;
};
