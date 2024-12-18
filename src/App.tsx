import React, { useState, useRef } from "react";
import cv from "@techstark/opencv-js";
import { Tensor, InferenceSession } from "onnxruntime-web";
import Loader from "./components/loader";
import { detectImage } from "./utils/detect.tsx";
import { download } from "./utils/download";
import "./style/App.css";

const App = () => {
  const [session, setSession] = useState(null);
  const [loading, setLoading] = useState({ text: "Loading...", progress: null });
  const [image, setImage] = useState(null);
  const inputImage = useRef(null);
  const imageRef = useRef(null);
  const canvasRef = useRef(null);
    
  // configs
  // const modelName = "yolov8n-seg.onnx";
  // const modelName = "reported-best.onnx"
  const modelName = "reported-v13-optimized.onnx"
  const modelInputShape = [1, 3, 512, 512];
  const topk = 100;
  const iouThreshold = 0.4;
  const scoreThreshold = 0.2;

  const handleClick = (e) => {
    e.preventDefault();
  
    // Get the img element inside the clicked element
    const img = e.target.closest('a').querySelector('img');

    
    if (img) {
      // Extract the filename from the src (after the last '/')
      const filename = img.src.split('/').pop();
      
      // Split by "_" and rebuild the filename
      const baseName = filename.split('_')[0]; // Get the first part before the underscore
      const newFileName = `${baseName}.jpg`;  // Rebuild the new filename with ".jpg"
      
      // Construct the new image URL using the updated file name
      const newImageUrl = process.env.PUBLIC_URL + '/' + newFileName;
  
      // Call changeImage with the new image URL
      changeImage(newImageUrl);
    }
  };

  const changeImage = (url) => {
    if (image) {
      // Revoke the previous object URL if it was set
      URL.revokeObjectURL(image);
      setImage(null);
    }
  
    let u;
  
    if (url instanceof URL || typeof url === 'string') {
      // If the url is a valid URL string or URL object, use it directly
      u = url;
    } else if (url instanceof File) {
      // If the url is a File object (from an input field), create an object URL
      u = URL.createObjectURL(url);
    } else {
      // If the type is unknown, handle accordingly (this should not happen)
      console.error('Invalid image source');
      return;
    }
  
    // Set the image source
    imageRef.current.src = u;
    setImage(u); // Update the image state
  };

  // wait until opencv.js initialized
  cv["onRuntimeInitialized"] = async () => {
    const baseModelURL = `${process.env.PUBLIC_URL}/model`;

    // create session
    const arrBufNet = await download(
      `${baseModelURL}/${modelName}`, // url
      ["Loading Segmentation model", setLoading] // logger
    );
    const yolov8 = await InferenceSession.create(arrBufNet);
    const arrBufNMS = await download(
      `${baseModelURL}/nms-yolov8.onnx`, // url
      ["Loading NMS model", setLoading] // logger
    );
    const nms = await InferenceSession.create(arrBufNMS);
    const arrBufMask = await download(
      `${baseModelURL}/mask-yolov8-seg.onnx`, // url
      ["Loading Mask model", setLoading] // logger
    );
    const mask = await InferenceSession.create(arrBufMask);

    // warmup main model
    setLoading({ text: "Warming up model...", progress: null });
    const tensor = new Tensor(
      "float32",
      new Float32Array(modelInputShape.reduce((a, b) => a * b)),
      modelInputShape
    );
    await yolov8.run({ images: tensor });

    setSession({ net: yolov8, nms: nms, mask: mask });
    setLoading(null);
  };

  return (
    <div className="App">
      <div className="header">
        <h1>Reported.nyc v12 Segmentation Model for tlc identification and blocked bike lane and crosswalks</h1>
        <p>
          Trained on data submitted to 311 nyc{" "}
          <code>onnxruntime-web</code>
        </p>
        <p>
          Serving : <code className="code">{modelName}</code>
        </p>
      </div>

      <div>
      <a href="#" onClick={handleClick}><img src={`${process.env.PUBLIC_URL}/a_thmb.jpg`} className="thumb"/></a>
      <a href="#" onClick={handleClick}><img src ={`${process.env.PUBLIC_URL}/b_thmb.jpg`} className="thumb"/></a>
      </div>

      <div className="content">        
        <img
          className="subject"
          ref={imageRef}
          src="#"
          alt=""
          style={{ display: image ? "block" : "none" }}
          onLoad={() => {
            detectImage(
              imageRef.current,
              canvasRef.current,
              session,
              topk,
              iouThreshold,
              scoreThreshold,
              modelInputShape
            );
          }}
        />
        <canvas
          id="canvas"
          width={modelInputShape[2]}
          height={modelInputShape[3]}
          ref={canvasRef}
        />
      </div>

      <input
        type="file"
        ref={inputImage}
        accept="image/*"
        style={{ display: "none" }}
        onChange={(e) => {
          // handle next image to detect
          changeImage(e.target.files[0])          
        }}
      />
      <div className="btn-container">
        <button
          onClick={() => {
            inputImage.current.click();
          }}
        >
          Open local image
        </button>
        {image && (
          /* show close btn when there is image */
          <button
            onClick={() => {
              inputImage.current.value = "";
              imageRef.current.src = "#";
              URL.revokeObjectURL(image);
              setImage(null);
            }}
          >
            Close image
          </button>
        )}
      </div>
      {loading && (
        <Loader>
          {loading.progress ? `${loading.text} - ${loading.progress}%` : loading.text}
        </Loader>
      )}
    </div>
  );
};

export default App;
