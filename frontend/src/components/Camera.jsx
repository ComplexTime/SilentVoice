// src/components/Camera.jsx
import {
  forwardRef,
  useCallback,
  useEffect,
  useImperativeHandle,
  useRef,
  useState,
} from "react";
import { CAPTURE_HEIGHT, CAPTURE_WIDTH } from "../config";

// status: "off" | "starting" | "on" | "denied" | "unavailable" | "error"
const STATUS_TEXT = {
  off: "Camera is off",
  starting: "Starting camera...",
  on: "Camera connected",
  denied: "Camera permission denied",
  unavailable: "No camera found",
  error: "Camera error",
};

const Camera = forwardRef(function Camera({ onStatusChange }, ref) {
  const videoRef = useRef(null);
  const streamRef = useRef(null);
  const [status, setStatus] = useState("off");
  const [errorMessage, setErrorMessage] = useState("");

  const updateStatus = useCallback(
    (next) => {
      setStatus(next);
      if (onStatusChange) onStatusChange(next);
    },
    [onStatusChange]
  );

  const stopCamera = useCallback(() => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
  }, []);

  const startCamera = useCallback(async () => {
    setErrorMessage("");

    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      updateStatus("unavailable");
      setErrorMessage(
        "This browser does not support camera access. Use Chrome or Edge on localhost or HTTPS."
      );
      return;
    }

    updateStatus("starting");
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: CAPTURE_WIDTH },
          height: { ideal: CAPTURE_HEIGHT },
          facingMode: "user",
        },
        audio: false,
      });
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play().catch(() => {});
      }
      updateStatus("on");
    } catch (err) {
      stopCamera();
      if (err.name === "NotAllowedError" || err.name === "SecurityError") {
        updateStatus("denied");
        setErrorMessage(
          "Camera access was blocked. Click the camera icon in the address bar, allow access, then press Start camera again."
        );
      } else if (
        err.name === "NotFoundError" ||
        err.name === "OverconstrainedError"
      ) {
        updateStatus("unavailable");
        setErrorMessage("No camera was found on this device.");
      } else if (err.name === "NotReadableError") {
        updateStatus("error");
        setErrorMessage(
          "The camera is in use by another app. Close it and try again."
        );
      } else {
        updateStatus("error");
        setErrorMessage("Could not start the camera: " + err.message);
      }
    }
  }, [stopCamera, updateStatus]);

  const disableCamera = useCallback(() => {
    stopCamera();
    setErrorMessage("");
    updateStatus("off");
  }, [stopCamera, updateStatus]);

  // Release the camera when the component is removed
  useEffect(() => {
    return () => {
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((track) => track.stop());
        streamRef.current = null;
      }
    };
  }, []);

  // Expose the video element to the parent (used for frame capture in step 6)
  useImperativeHandle(
    ref,
    () => ({
      getVideoElement: () => videoRef.current,
      isReady: () =>
        status === "on" &&
        videoRef.current !== null &&
        videoRef.current.readyState >= 2,
    }),
    [status]
  );

  const isOn = status === "on";
  const isStarting = status === "starting";

  return (
    <div className="camera">
      <div className="camera-frame">
        {/* Mirrored with CSS only. Captured frames stay unmirrored. */}
        <video
          ref={videoRef}
          className={"camera-video" + (isOn ? " visible" : "")}
          autoPlay
          playsInline
          muted
          aria-label="Live camera preview"
        />
        {!isOn && (
          <div className="camera-placeholder">
            {isStarting ? (
              <>
                <div className="spinner" aria-hidden="true" />
                <p>Starting camera...</p>
              </>
            ) : (
              <p>Press "Start camera" to begin</p>
            )}
          </div>
        )}
      </div>

      <div className="camera-bar">
        <div className={"status-pill status-" + status} role="status">
          <span className="dot" aria-hidden="true" />
          {STATUS_TEXT[status]}
        </div>

        <div className="camera-buttons">
          {!isOn ? (
            <button
              type="button"
              className="btn btn-primary"
              onClick={startCamera}
              disabled={isStarting}
            >
              Start camera
            </button>
          ) : (
            <button
              type="button"
              className="btn btn-danger"
              onClick={disableCamera}
            >
              Stop camera
            </button>
          )}
        </div>
      </div>

      {errorMessage && (
        <p className="alert alert-error" role="alert">
          {errorMessage}
        </p>
      )}
    </div>
  );
});

export default Camera;