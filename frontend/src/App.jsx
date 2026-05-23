import { useEffect, useRef, useState, useMemo } from "react";
import DailyIframe from "@daily-co/daily-js";

function App() {
  const callObjectRef = useRef(null);
  const videoRefs = useRef({});

  const [participants, setParticipants] = useState({});
  const [joined, setJoined] = useState(false);
  const [micOn, setMicOn] = useState(true);
  const [camOn, setCamOn] = useState(true);

  // -----------------------------
  // Setup Daily
  // -----------------------------
  useEffect(() => {
    if (callObjectRef.current) return;

    const callObject = DailyIframe.createCallObject();
    callObjectRef.current = callObject;

    const update = () => {
      setParticipants({ ...callObject.participants() });
    };

    callObject.on("joined-meeting", update);
    callObject.on("participant-joined", update);
    callObject.on("participant-updated", update);
    callObject.on("participant-left", update);

    return () => callObject.destroy();
  }, []);

  // -----------------------------
  // Attach video streams
  // -----------------------------
  useEffect(() => {
    Object.values(participants).forEach((p) => {
      const track = p?.tracks?.video?.persistentTrack;
      const el = videoRefs.current[p.session_id];

      if (!track || !el) return;

      el.srcObject = new MediaStream([track]);
    });
  }, [participants]);

  // -----------------------------
  // Derived layout state
  // -----------------------------
  const remoteParticipants = useMemo(() => {
    return Object.values(participants).filter((p) => !p.local);
  }, [participants]);

  const localParticipant = useMemo(() => {
    return Object.values(participants).find((p) => p.local);
  }, [participants]);

  const activeParticipant = useMemo(() => {
    const all = Object.values(participants);

    const remote = all.filter(p => !p.local);
    const local = all.find(p => p.local);

    // Prefer remote with video on
    const activeRemote = remote.find(
      p => p.tracks?.video?.state === "playable"
    );

    return activeRemote || local || null;
  }, [participants]);

  // -----------------------------
  // Controls
  // -----------------------------
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

  const toggleCam = async () => {
    const next = !camOn;
    await callObjectRef.current.setLocalVideo(next);
    setCamOn(next);
  };

  // -----------------------------
  // UI
  // -----------------------------
  return (
    <div style={styles.app}>
      {/* Top Bar */}
      <div style={styles.topBar}>
        <div style={styles.title}>Emotion Call</div>

        <div style={styles.controls}>
          {!joined ? (
            <button onClick={joinCall} style={styles.btnPrimary}>
              Join
            </button>
          ) : (
            <>
              <button onClick={toggleMic} style={styles.btn}>
                {micOn ? "Mute" : "Unmute"}
              </button>

              <button onClick={toggleCam} style={styles.btn}>
                {camOn ? "Cam Off" : "Cam On"}
              </button>

              <button onClick={leaveCall} style={styles.btnDanger}>
                Leave
              </button>
            </>
          )}
        </div>
      </div>

      {/* Main Stage */}
      <div style={styles.stage}>
        {activeParticipant && (
          <div style={styles.activeTile}>
            <video
              ref={(el) => {
                if (el)
                  videoRefs.current[activeParticipant.session_id] = el;
              }}
              autoPlay
              playsInline
              muted={activeParticipant.local}
              style={styles.activeVideo}
            />

            <div style={styles.label}>
              {activeParticipant.local ? "You" : activeParticipant.user_name}
            </div>
          </div>
        )}
      </div>

      {/* Bottom Strip */}
      <div style={styles.strip}>
        {Object.values(participants).map((p) => (
          <div key={p.session_id} style={styles.thumb}>
            <video
              ref={(el) => {
                if (el) videoRefs.current[p.session_id] = el;
              }}
              autoPlay
              playsInline
              muted={p.local}
              style={styles.thumbVideo}
            />
            <div style={styles.thumbLabel}>
              {p.local ? "You" : p.user_name}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

export default App;

// -----------------------------
// Styles (Zoom-like layout)
// -----------------------------
const styles = {
  app: {
    height: "100vh",
    display: "flex",
    flexDirection: "column",
    background: "#0b0f14",
    color: "white",
    fontFamily: "sans-serif",
  },

  topBar: {
    height: "60px",
    display: "flex",
    justifyContent: "space-between",
    alignItems: "center",
    padding: "0 16px",
    background: "#121826",
    borderBottom: "1px solid #1f2937",
  },

  title: {
    fontWeight: "bold",
  },

  controls: {
    display: "flex",
    gap: "8px",
  },

  btn: {
    padding: "8px 12px",
    background: "#1f2937",
    color: "white",
    border: "none",
    borderRadius: "6px",
    cursor: "pointer",
  },

  btnPrimary: {
    padding: "8px 12px",
    background: "#2563eb",
    color: "white",
    border: "none",
    borderRadius: "6px",
    cursor: "pointer",
  },

  btnDanger: {
    padding: "8px 12px",
    background: "#dc2626",
    color: "white",
    border: "none",
    borderRadius: "6px",
    cursor: "pointer",
  },

  stage: {
    flex: 1,
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    padding: "20px",
  },

  activeTile: {
    width: "70%",
    maxWidth: "900px",
    position: "relative",
    borderRadius: "12px",
    overflow: "hidden",
    background: "black",
  },

  activeVideo: {
    width: "100%",
    height: "100%",
    objectFit: "cover",
  },

  label: {
    position: "absolute",
    bottom: "10px",
    left: "10px",
    background: "rgba(0,0,0,0.5)",
    padding: "4px 8px",
    borderRadius: "6px",
  },

  strip: {
    height: "140px",
    display: "flex",
    gap: "10px",
    padding: "10px",
    overflowX: "auto",
    background: "#0f172a",
    borderTop: "1px solid #1f2937",
  },

  thumb: {
    width: "160px",
    flexShrink: 0,
    position: "relative",
    borderRadius: "10px",
    overflow: "hidden",
    background: "black",
  },

  thumbVideo: {
    width: "100%",
    height: "100%",
    objectFit: "cover",
  },

  thumbLabel: {
    position: "absolute",
    bottom: "4px",
    left: "4px",
    fontSize: "12px",
    background: "rgba(0,0,0,0.5)",
    padding: "2px 6px",
    borderRadius: "4px",
  },
};