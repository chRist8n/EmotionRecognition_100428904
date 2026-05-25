import { useEffect, useRef, useState } from "react";
import DailyIframe from "@daily-co/daily-js";

function App() {
  const callFrameRef = useRef(null);
  const videoRefs = useRef({});
  const socketsRef = useRef({});
  const canvasRef = useRef(document.createElement("canvas"));

  const [participants, setParticipants] = useState({});
  const [inCall, setInCall] = useState(false);

  // ----------------------------
  // 1. Create Daily frame
  // ----------------------------
  useEffect(() => {
    const callFrame = DailyIframe.createFrame({
      iframeStyle: {
        width: "100%",
        height: "100vh",
        border: "0",
      },
    });

    callFrameRef.current = callFrame;

    // participant tracking
    callFrame.on("participant-joined", updateParticipants);
    callFrame.on("participant-updated", updateParticipants);
    callFrame.on("participant-left", updateParticipants);

    function updateParticipants() {
      setParticipants(callFrame.participants());
    }

    // call state tracking (better than manual flags)
    callFrame.on("joined-meeting", () => setInCall(true));
    callFrame.on("left-meeting", () => setInCall(false));

    // AUTO-ENTER PREJOIN SCREEN / CALL FLOW
    callFrame.join({
      url: "https://emotion-recognition.daily.co/emotion-recognition",
    });

    return () => callFrame.destroy();
  }, []);

  // ----------------------------
  // 2. Leave call
  // ----------------------------
  const leaveCall = async () => {
    if (!callFrameRef.current) return;

    await callFrameRef.current.leave();
  };

  // ----------------------------
  // 3. Attach video refs
  // ----------------------------
  const setVideoRef = (id, el) => {
    if (el) {
      videoRefs.current[id] = el;
    }
  };

  // ----------------------------
  // 4. WebSockets per participant
  // ----------------------------
  useEffect(() => {
    Object.values(participants).forEach((p) => {
      const id = p.session_id;
      if (!id || socketsRef.current[id]) return;

      const socket = new WebSocket(`ws://localhost:8000/ws/${id}`);
      socketsRef.current[id] = socket;
    });
  }, [participants]);

  // ----------------------------
  // 5. Frame capture
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
  // 6. Streaming loop
  // ----------------------------
  useEffect(() => {
    if (!inCall) return;

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
    }, 300);

    return () => clearInterval(interval);
  }, [participants, inCall]);

  // ----------------------------
  // 7. UI
  // ----------------------------
  return (
    <div style={{ position: "relative" }}>

      {/* Daily iframe */}
      <div id="call-container" />

      {/* Leave button (only when in call) */}
      {inCall && (
        <button
          onClick={leaveCall}
          style={{
            position: "absolute",
            bottom: 20,
            right: 20,
            zIndex: 10,
            padding: "12px 18px",
            backgroundColor: "red",
            color: "white",
            border: "none",
            borderRadius: "6px",
            cursor: "pointer",
          }}
        >
          Leave Call
        </button>
      )}

      {/* Hidden video refs */}
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