import gradio as gr
import cv2
import numpy as np
import tempfile
import subprocess
import os
import shutil
import sys
import commentjson
import zipfile
from seem_extraction import SEEMPipeline, SEEMPreview

HEADER_TEXT = """
# 🍻 Hold My NeRF

## Instructions
1. Provide a video of you turning the object to be captured in your hand.
2. Provide a text prompt describing the object with a single word.
3. Preview the segmentation to ensure that the object is shown correctly
4. Start processing by pressing "Submit"
5. A 3D model and the Instant-NGP checkpoint will be available for download once completed
6. If desired, a video orbit of the object can be rendered from the NeRF directly
"""
ROOT_DIR = os.path.dirname(os.path.realpath(__file__))

def get_video_duration(video):
    vid = cv2.VideoCapture(video)

    fps = vid.get(cv2.CAP_PROP_FPS)
    frames = vid.get(cv2.CAP_PROP_FRAME_COUNT)

    return frames / fps

def get_first_frame(video: str):
    if video != "":
        cap = cv2.VideoCapture(video)
        is_read, img = cap.read()
        if is_read:
            cv2.imwrite(os.path.join(os.path.dirname(video), "first_frame.png"),img)

def preview_segmentation(params):
    video_file = params[video]
    gradio_dir = os.path.dirname(video_file)

    get_first_frame(video_file)

    img = os.path.join(gradio_dir, "first_frame.png")
    mask = SEEMPreview(img, params[text_prompt])

    mask = mask.astype(np.float32)
    mask /= 255.
    mask *= 0.5

    return (img, [(mask, params[text_prompt])])

def run_nerf(params, progress=gr.Progress()):
    video_file = params[video]
    video_name = os.path.basename(video_file)
    video_length = get_video_duration(video_file)
    gradio_dir = os.path.dirname(video_file)

    with tempfile.TemporaryDirectory() as tempdir:
        shutil.copy2(video_file, tempdir)
        progress((0,4), desc="Extracting Frames")
        subprocess.run([sys.executable,
                        os.path.join(ROOT_DIR,"dependencies/instant_ngp/scripts/colmap2nerf.py"), 
                        "--video_in", os.path.join(tempdir, video_name),
                        "--video_fps", str(int(100 / video_length)),
                        "--overwrite"], cwd=tempdir)

        progress((1,4), desc="Removing Background")
        masked_dir = os.path.join(tempdir, "masked")
        SEEMPipeline(os.path.join(tempdir, "images"), masked_dir, params[text_prompt])

        progress((2,4), desc="Running COLMAP")
        subprocess.run([sys.executable,
                        os.path.join(ROOT_DIR,"dependencies/instant_ngp/scripts/colmap2nerf.py"), 
                        "--images", masked_dir,
                        "--run_colmap",
                        "--aabb_scale", "1",
                        "--overwrite"], cwd=tempdir)

        if params[use_per_image]:
            with open(os.path.join(tempdir, "transforms.json"), "r") as transforms:
                data = commentjson.load(transforms)
            data["n_extra_learnable_dims"] = 16
            with open(os.path.join(tempdir, "transforms.json"), "w") as transforms:
                commentjson.dump(data, transforms)
            shutil.copy2(os.path.join(tempdir, "transforms.json"), gradio_dir)

        progress((3,4), desc="Training NeRF")
        subprocess.run([sys.executable,
                        os.path.join(ROOT_DIR, "dependencies/instant_ngp/scripts/run.py"),
                        "--n_steps", f"{params[n_steps]}",
                        "--save_snapshot", "snapshot.ingp",
                        "--save_mesh", "model.obj",
                        os.path.join(tempdir, "transforms.json")], cwd=tempdir)

        progress((4,4), desc="Completed")

        shutil.copy2(os.path.join(tempdir, "snapshot.ingp"), gradio_dir)
        shutil.copy2(os.path.join(tempdir, "model.obj"), gradio_dir)

        if params[debug_intermediate]:
            with zipfile.ZipFile(os.path.join(gradio_dir, "intermediates.zip"), "w") as zipf:
                for root, _, filenames in os.walk(tempdir):
                    for f in filenames:
                        zipf.write(os.path.join(root, f))

            return {checkpoint_file: os.path.join(gradio_dir, "snapshot.ingp"), 
                    model:os.path.join(gradio_dir, "model.obj"),
                    intermediates:zipf.filename}

    return {checkpoint_file: os.path.join(gradio_dir, "snapshot.ingp"), 
            model:os.path.join(gradio_dir, "model.obj")}

def create_video(params):
    gradio_dir = os.path.dirname(params[checkpoint_file].name)

    subprocess.run([
        sys.executable,
        os.path.join(ROOT_DIR, "dependencies/instant_ngp/scripts/run.py"),
        "--load_snapshot", params[checkpoint_file].name,
        "--width", f"{params[video_width]}",
        "--height", f"{params[video_height]}",
        "--video_camera_path", os.path.join(ROOT_DIR, "config/camera_path.json"),
        "--video_fps", f"{params[fps]}",
        "--video_n_seconds", f"{params[seconds]}",
        "--video_spp", f"{params[spp]}",
    ], cwd=gradio_dir)

    return os.path.join(gradio_dir, "video.mp4")

if __name__ == "__main__":
    #inputs
    video = gr.Video(format="mp4", source="upload", label="Video", interactive=True)
    text_prompt = gr.Textbox(label="Object Label", info="Provide a label for the object for segmentation", interactive=True)

    #segmentation preview
    segmentation = gr.AnnotatedImage(label="Segmentation")

    #outputs
    model = gr.Model3D(label="3D Model", interactive=False, clear_color=[0,0,0])
    checkpoint_file = gr.File(label="Instant-NPG Checkpoint", interactive=False)
    orbit_video = gr.Video(label="Orbit Video", interactive=False)
    intermediates = gr.Files(label="Intermediate Files", interactive=False, visible=False)

    with gr.Blocks() as demo:
        gr.Markdown(HEADER_TEXT)

        with gr.Row():
            with gr.Column():
                with gr.Box():
                    video.render()
                    text_prompt.render()

                    with gr.Accordion("NeRF Parameters", open=False):
                        use_per_image = gr.Checkbox(value=True, label="Per Image Latents")
                        n_steps = gr.Number(value=10000, label="#Steps", precision=0)
                        debug_intermediate = gr.Checkbox(value=False, label="Save Intermediates")
                        debug_intermediate.change(fn=lambda dbg: gr.update(visible=dbg), inputs=[debug_intermediate], outputs=[intermediates])

                    with gr.Row():
                        preview = gr.Button("Preview Segmentation")
                        preview.click(fn=preview_segmentation, inputs={video, text_prompt}, outputs=[segmentation], api_name="preview")
                        run = gr.Button("Submit")
                        run.click(fn=run_nerf, inputs={video, text_prompt, n_steps, use_per_image, debug_intermediate}, outputs=[model, checkpoint_file, intermediates], api_name="nerf")


            with gr.Column():
                with gr.Box():
                    with gr.Tab("Preview"):
                        segmentation.render()
                    with gr.Tab("Results"):
                        model.render()
                        checkpoint_file.render()
                        intermediates.render()
                        with gr.Box():
                            orbit_video.render()
                            with gr.Accordion("Video Parameters", open=True):
                                with gr.Row():
                                    video_width = gr.Number(value=720, label="Width", precision=0)
                                    video_height = gr.Number(value=480, label="Height", precision=0)
                                fps = gr.Slider(minimum=10, maximum=60, value=30, label="FPS", step=10)
                                seconds = gr.Number(value=5, label="Video Length", precision=1)
                                spp = gr.Slider(1,16,8, label="Samples per Pixel")
                                render_vid = gr.Button("Render Video")
                                render_vid.click(fn=create_video, 
                                                 inputs={checkpoint_file, video_width, video_height,
                                                         fps, seconds, spp}, outputs=[orbit_video], api_name="get_video")

        gr.Examples([["examples/cube_clean.mp4", "cube"]], inputs=[video, text_prompt])

    demo.queue()
    demo.launch()