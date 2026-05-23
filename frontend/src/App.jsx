import { useEffect, useMemo, useRef, useState } from "react";
import DailyIframe from "@daily-co/daily-js";

function App() {
  const callRef = useRef(null);

  const stageVideoRef = useRef(null);
  const thumbRefs = useRef({});

  const [participants, setParticipants] = useState({});
  const [joined, setJoined] = useState(false);

  const [micOn, setMicOn] = useState(true);
  const [camOn, setCamOn] = useState(true);

  // -------------------------
  // Init Daily call object
  // -------------------------
  useEffect(() => {
    if (callRef.current) return;

    const call = DailyIframe.createCallObject();
    callRef.current = call;

    const update = () => {
      setParticipants({ ...call.participants() });
    };

    call.on("joined-meeting", update);
    call.on("participant-joined", update);
    call.on("participant-updated", update);
    call.on("participant-left", update);

    return () => call.destroy();
  }, []);

  // -------------------------
  // Join / Leave
  // -------------------------
  const join = async () => {
    await callRef.current.join({
      url: "https://emotion-recognition.daily.co/emotion-recognition",
    });
    setJoined(true);
  };

  const leave = async () => {
    await callRef.current.leave();
    setJoined(false);
  };

  const toggleMic = async () => {
    const next = !micOn;
    await callRef.current.setLocalAudio(next);
    setMicOn(next);
  };

  const toggleCam = async () => {
    const next = !camOn;
    await callRef.current.setLocalVideo(next);
    setCamOn(next);
  };

  // -------------------------
  // Participants derived state
  // -------------------------
  const allParticipants = useMemo(
    () => Object.values(participants),
    [participants]
  );

  const remoteParticipants = useMemo(
    () => allParticipants.filter((p) => !p.local),
    [allParticipants]
  );

  const localParticipant = useMemo(
    () => allParticipants.find((p) => p.local),
    [allParticipants]
  );

  // pick best "active" participant
  const activeParticipant = useMemo(() => {
    return (
      remoteParticipants.find(
        (p) => p.tracks?.video?.state === "playable"
      ) ||
      remoteParticipants[0] ||
      localParticipant ||
      null
    );
  }, [remoteParticipants, localParticipant]);

  // -------------------------
  // Attach MAIN video stream
  // -------------------------
  useEffect(() => {
    if (!activeParticipant || !stageVideoRef.current) return;

    const track =
      activeParticipant?.tracks?.video?.persistentTrack;

    if (!track) return;

    stageVideoRef.current.srcObject = new MediaStream([track]);
  }, [activeParticipant]);

  // -------------------------
  // Attach thumbnail streams
  // -------------------------
  useEffect(() => {
    allParticipants.forEach((p) => {
      const el = thumbRefs.current[p.session_id];
      const track = p?.tracks?.video?.persistentTrack;

      if (!el || !track) return;

      el.srcObject = new MediaStream([track]);
    });
  }, [allParticipants]);

  // -------------------------
  // UI
  // -------------------------
  return (
    <div style={styles.app}>
      {/* TOP BAR */}
      <div style={styles.topBar}>
        <div style={styles.title}>Emotion Call</div>

        <div style={styles.controls}>
          {!joined ? (
            <button onClick={join} style={styles.primary}>
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

              <button onClick={leave} style={styles.danger}>
                Leave
              </button>
            </>
          )}
        </div>
      </div>

      {/* MAIN STAGE */}
      <div style={styles.stage}>
        {activeParticipant ? (
          <div style={styles.stageBox}>
            <video
              ref={stageVideoRef}
              autoPlay
              playsInline
              muted={activeParticipant.local}
              style={styles.stageVideo}
            />

            <div style={styles.label}>
              {activeParticipant.local
                ? "You"
                : activeParticipant.user_name}
            </div>
          </div>
        ) : (
          <div style={styles.empty}>Waiting for participants...</div>
        )}
      </div>

      {/* THUMB STRIP */}
      <div style={styles.strip}>
        {allParticipants.map((p) => (
          <div key={p.session_id} style={styles.thumb}>
            <video
              ref={(el) => {
                if (el) thumbRefs.current[p.session_id] = el;
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

// -------------------------
// Styles
// -------------------------
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
    background: "#111827",
    borderBottom: "1px solid #1f2937",
  },

  title: {
    fontWeight: 600,
  },

  controls: {
    display: "flex",
    gap: "8px",
  },

  btn: {
    padding: "8px 12px",
    background: "#1f2937",
    border: "none",
    borderRadius: "6px",
    color: "white",
    cursor: "pointer",
  },

  primary: {
    padding: "8px 12px",
    background: "#2563eb",
    border: "none",
    borderRadius: "6px",
    color: "white",
    cursor: "pointer",
  },

  danger: {
    padding: "8px 12px",
    background: "#dc2626",
    border: "none",
    borderRadius: "6px",
    color: "white",
    cursor: "pointer",
  },

  stage: {
    flex: 1,
    display: "flex",
    justifyContent: "center",
    alignItems: "center",
    padding: "20px",
  },

  stageBox: {
    width: "75%",
    maxWidth: "1000px",
    position: "relative",
    borderRadius: "12px",
    overflow: "hidden",
    background: "black",
  },

  stageVideo: {
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
    fontSize: "14px",
  },

  empty: {
    opacity: 0.6,
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
    borderRadius: "10px",
    overflow: "hidden",
    background: "black",
    position: "relative",
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