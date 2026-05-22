import { useEffect, useRef } from "react";
import DailyIframe from "@daily-co/daily-js";

function App() {
  const containerRef = useRef(null);
  const frameRef = useRef(null);
  const joinedRef = useRef(false);

  useEffect(() => {
    // prevents StrictMode double-mount + re-entry
    if (joinedRef.current) return;
    joinedRef.current = true;

    const frame = DailyIframe.createFrame(containerRef.current, {
      showLeaveButton: true,
      iframeStyle: {
        width: "100%",
        height: "100vh",
        border: "0",
      },
    });

    frameRef.current = frame;

    frame.join({ url: "https://emotion-recognition.daily.co/emotion-recognition" });

    return () => {
      frame.destroy();
      frameRef.current = null;
      joinedRef.current = false;
    };
  }, []);

  return (
    <div>
      <div ref={containerRef} />
      <div style={{ position: "absolute", top: 20, left: 20 }}>
        Emotion Output
      </div>
    </div>
  );
}

export default App;