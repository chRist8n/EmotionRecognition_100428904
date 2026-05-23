import { useEffect, useRef, useState } from "react";
import DailyIframe from "@daily-co/daily-js";

function App() {
  const callObjectRef = useRef(null);
  const videoRefs = useRef({});

  const [participants, setParticipants] = useState({});
  const [joined, setJoined] = useState(false);
  const [micOn, setMicOn] = useState(true);
  const [camOn, setCamOn] = useState(true);

  // ----------------------------
  // Setup Daily
  // ----------------------------
  useEffect(() => {
    if (callObjectRef.current) return;

    const callObject = DailyIframe.createCallObject();
    callObjectRef.current = callObject;

    const updateParticipants = () => {
      setParticipants({ ...callObject.participants() });
    };

    callObject.on("joined-meeting", updateParticipants);
    callObject.on("participant-joined", updateParticipants);
    callObject.on("participant-updated", updateParticipants);
    callObject.on("participant-left", updateParticipants);

    return () => {
      callObject.destroy();
      callObjectRef.current = null;
    };
  }, []);

  // ----------------------------
  // Attach video tracks
  // ----------------------------
  useEffect(() => {
    Object.values(participants).forEach((p) => {
      const track = p?.tracks?.video?.persistentTrack;
      const videoEl = videoRefs.current[p.session_id];

      if (!track || !videoEl) return;

      const stream = new MediaStream([track]);
      videoEl.srcObject = stream;
    });
  }, [participants]);

  // ----------------------------
  // Frame capture loop
  // ----------------------------
  useEffect(() => {
    const interval = setInterval(() => {
      Object.values(participants).forEach((p) => {
        const video = videoRefs.current[p.session_id];
        if (!video || video.readyState < 2) return;

        const canvas = document.createElement("canvas");
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;

        const ctx = canvas.getContext("2d");
        ctx.drawImage(video, 0, 0);

        canvas.toBlob(async (blob) => {
          if (!blob) return;

          const formData = new FormData();
          formData.append("file", blob, "frame.jpg");

          try {
            const res = await fetch("http://localhost:8000/predict", {
              method: "POST",
              body: formData,
            });

            const data = await res.json();

            console.log(p.user_name || p.session_id, data);
          } catch (err) {
            console.error(err);
          }
        }, "image/jpeg");
      });
    }, 1000); // 1 FPS

    return () => clearInterval(interval);
  }, [participants]);

  // ----------------------------
  // Controls
  // ----------------------------
  const joinCall = async () => {
    await callObjectRef.current.join({
      url: "https://emotion-recognition.daily.co/emotion-recognition",
    });
    setJoined(true);
  };

  const leaveCall = async () => {
    await callObjectRef.current.leave();
    setJoined(false);
  };

  const toggleMic = async () => {
    const next = !micOn;
    await callObjectRef.current.setLocalAudio(next);
    setMicOn(next);
  };

  const toggleCamera = async () => {
    const next = !camOn;
    await callObjectRef.current.setLocalVideo(next);
    setCamOn(next);
  };

  // ----------------------------
  // UI
  // ----------------------------
  return (
    <div style={{ padding: "10px" }}>
      {/* Controls */}
      <div style={{ display: "flex", gap: "10px", marginBottom: "10px" }}>
        {!joined ? (
          <button onClick={joinCall}>Join</button>
        ) : (
          <>
            <button onClick={leaveCall}>Leave</button>
            <button onClick={toggleMic}>
              {micOn ? "Mute Mic" : "Unmute Mic"}
            </button>
            <button onClick={toggleCamera}>
              {camOn ? "Turn Camera Off" : "Turn Camera On"}
            </button>
          </>
        )}
      </div>

      {/* Video grid */}
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fit, minmax(300px, 1fr))",
          gap: "10px",
        }}
      >
        {Object.values(participants).map((p) => (
          <video
            key={p.session_id}
            ref={(el) => {
              videoRefs.current[p.session_id] = el;
            }}
            autoPlay
            playsInline
            muted={p.local}
            style={{
              width: "100%",
              background: "black",
              borderRadius: "10px",
            }}
          />
        ))}
      </div>
    </div>
  );
}

export default App;