# main.py

import cv2
from video_loader import load_and_sample_video
from object_detection import YOLODetector
from tracker import CentroidTracker
from anomaly_detection import detect_anomalies_with_tracking
from utils import draw_detections
from vlm_reasoning import GeminiReasoner
from rag_index import AnomalyRAG
from PIL import Image
import numpy as np
import gradio as gr

rag = AnomalyRAG()
reasoner = GeminiReasoner(api_key="AIzaSyCV077Q-1nwxMbPM_3f4Z1h1b7ZHjIBE78")  # Replace with your key


def get_frame_image(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def main(video_path="sample.mp4", frame_interval=5, max_frames=30, save_path="output.mp4"):
    print(f"▶️ Loading video: {video_path}")
    frames = load_and_sample_video(video_path, frame_interval=frame_interval, max_frames=max_frames)
    print(f"🖼️ {len(frames)} frames sampled.")

    print("🔍 Running object detection...")
    detector = YOLODetector(model_path="yolov8n.pt", conf=0.5)
    detections_per_frame = detector.detect_objects(frames)

    print("🛰️ Tracking objects across frames...")
    tracker = CentroidTracker(max_distance=100, max_lost=2)
    tracks_per_frame = [tracker.update(d) for d in detections_per_frame]

    print("⚠️ Analyzing for anomalies...")
    anomalies = detect_anomalies_with_tracking(
        frames_detections=detections_per_frame,
        frames_tracks=tracks_per_frame,
        movement_thresh=10,
        max_stationary_frames=5,
        min_traffic_threshold=1
    )

    print("📚 Building RAG base...")
    for i, text in enumerate(anomalies):
        if text:
            rag.add_anomaly(text, metadata={'frame': i + 1})

    print("🎞️ Generating annotated video...")
    h, w, _ = frames[0].shape
    out = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), 5, (w, h))
    frame_annotated = []
    for i in range(len(frames)):
        frame = frames[i].copy()
        detections = detections_per_frame[i]
        tracks = tracks_per_frame[i]
        anomaly_text = anomalies[i]
        annotated = draw_detections(frame, detections, tracks, anomalies_text=anomaly_text)
        frame_annotated.append(annotated)
        out.write(annotated)
    out.release()
    print(f"✅ Saved annotated video to: {save_path}")
    return anomalies, rag, frame_annotated


# Launch the processing pipeline
anomalies, rag, frames = main()


# Gradio chatbot handler
def chat_with_rag(messages, history):
    try:
        # breakpoint()
        query = messages
        related = rag.retrieve(query, top_k=3)

        if not related:
            response = "❌ No relevant anomalies found."
            image = None
        else:
            context = rag.format_context(related)
            frame_idx = related[0]["meta"]["frame"] - 1
            image = get_frame_image(frames[frame_idx])
            prompt = f"Context from past frames:\n{context}\n\nQ: {query}"
            response = reasoner.ask(prompt)

    except Exception as e:
        response = f"⚠️ Error: {e}"
        image = None
    return response, image  # ✅ 2 outputs: (chatbot response, image)




# Gradio UI setup
def launch_ui():
    with gr.Blocks() as demo:
        image = gr.Image(label="Related Frame", type="pil", render=False)
        gr.Markdown(
            """
            # 🧠 Traffic Insight Chat (Gemini + RAG)

            Ask natural questions about traffic activity in the video.
            This system combines object detection, tracking, anomaly detection, and reasoning from visual context.
            """
        )
        with gr.Row():
            with gr.Column():
                gr.Markdown(
                    "### Traffic Insight Chatbot\n"
                )
                gr.ChatInterface(
                    fn=chat_with_rag,
                    additional_outputs=[image],
                    chatbot=gr.Chatbot(label="Gemini RAG Assistant", height=400),
                    type="messages"
                )
            with gr.Column():
                gr.Markdown('Related Frame:')
                image.render()

    demo.launch()




# CLI or UI entry
if __name__ == "__main__":
    mode = input("🟢 Type 'cli' for command line or 'ui' to launch web app: ").strip().lower()
    if mode == "ui":
        launch_ui()
    else:
        print("\n💬 Interactive Traffic Reasoning (type 'exit' to quit)\n")
        while True:
            user_query = input("🔎 Ask about traffic behavior (or 'exit'): ").strip()
            if user_query.lower() == "exit":
                print("👋 Exiting interactive mode.")
                break
            related_docs = rag.retrieve(user_query, top_k=3)
            context = rag.format_context(related_docs)
            full_prompt = f"Context from past frames:\n{context}\n\nQ: {user_query}"
            response = reasoner.ask(full_prompt)
            print(f"\n🧠 Gemini RAG:\n{response}\n")
