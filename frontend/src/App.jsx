import { useEffect, useRef, useState } from "react";
import DailyIframe from "@daily-co/daily-js";

function App() {
  const callFrameRef = useRef(null);
  const videoRefs = useRef({});
  const socketsRef = useRef({});
  const canvasRef = useRef(document.createElement("canvas"));

  const [participants, setParticipants] = useState({});



  // ----------------------------
  // 1. Join Daily call
  // ----------------------------
  useEffect(() => {
    const callFrame = DailyIframe.createFrame({
      iframeStyle: {
        width: "100%",
        height: "100vh",
        border: "0",
      },
    });

    callFrame.join({ url: "https://emotion-recognition.daily.co/emotion-recognition" });
    callFrameRef.current = callFrame;

    callFrame.on("participant-joined", updateParticipants);
    callFrame.on("participant-updated", updateParticipants);
    callFrame.on("participant-left", updateParticipants);

    function updateParticipants() {
      const p = callFrame.participants();
      setParticipants(p);
    }

    return () => callFrame.destroy();
  }, []);

  // ----------------------------
  // 2. Attach video elements
  // ----------------------------
  const setVideoRef = (id, el) => {
    if (el) {
      videoRefs.current[id] = el;
    }
  };

  // ----------------------------
  // 3. WebSocket per participant
  // ----------------------------
  useEffect(() => {
    Object.values(participants).forEach((p) => {
      const id = p.session_id;
      if (!id || socketsRef.current[id]) return;

      const socket = new WebSocket(
        `ws://localhost:8000/ws/${id}`
      );

      socketsRef.current[id] = socket;
    });
  }, [participants]);

  // ----------------------------
  // 4. Capture frame helper
  // ----------------------------
  const captureFrame = (video) => {
    if (!video) return null;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    return new Promise((resolve) => {
      canvas.toBlob((blob) => resolve(blob), "image/jpeg", 0.7);
    });
  };

  // ----------------------------
  // 5. Streaming loop
  // ----------------------------
  useEffect(() => {
    const interval = setInterval(async () => {
      const list = Object.values(participants);
      if (!list.length) return;

      for (const p of list) {
        const id = p.session_id;
        const video = videoRefs.current[id];
        const socket = socketsRef.current[id];

        if (!video || !socket || socket.readyState !== 1) continue;

        const blob = await captureFrame(video);
        if (!blob) continue;

        socket.send(blob);
      }
    }, 300); // adjust FPS here

    return () => clearInterval(interval);
  }, [participants]);

  // ----------------------------
  // 6. UI (Daily renders video elements itself)
  // ----------------------------
  return (
    <div style={{ position: "relative" }}>
      <div id="call-container" />

      {/* Hidden video refs (attach via DOM query later if needed) */}
      {Object.values(participants).map((p) => (
        <video
          key={p.session_id}
          ref={(el) => setVideoRef(p.session_id, el)}
          autoPlay
          playsInline
          style={{ display: "none" }}
        />
      ))}

      {/* Debug overlay */}
      <div style={{ position: "absolute", top: 10, left: 10 }}>
        Active participants: {Object.keys(participants).length}
      </div>
    </div>
  );
}

export default App;