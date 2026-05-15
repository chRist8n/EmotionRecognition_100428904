import { useEffect, useRef } from "react";
import DailyIframe from "@daily-co/daily-js";

export default function App() {
  const callFrameRef = useRef(null);

  useEffect(() => {
    const callFrame = DailyIframe.createFrame(
      callFrameRef.current,
      {
        showLeaveButton: true,
      }
    );


    callFrame.join({
      url: "https://emotion-recognition.daily.co/emotion-recognition",
    });

    return () => {
      //when quitting, close everything
      const shutdown = () => {
        const tracks = callFrame.localParticipant?.tracks;

        if (tracks) {
          Object.values(tracks).forEach(t => {
            if (t?.track?.stop) t.track.stop();
          });
        }

        callFrame.leave();
        callFrame.destroy();
      }

      shutdown();
    };
  }, []);

  return (
    <div
      ref={callFrameRef}
      style={{
        position: "fixed",
        inset: 0,
        width: "100vw",
        height: "100vh",
        overflow: "hidden"
      }}
    />
  );
}