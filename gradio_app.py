import gradio as gr
import cv2
import numpy as np
import tempfile
import subprocess
import os
import shutil
import sys
import commentjson

HEADER_TEXT = """
# Hold My NeRF

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

def preview_segmentation(params):
    img = np.random.random(size=(300,300,3))
    mask = np.zeros(shape=(300,300))
    mask[30:40, 30:40] = 1

    return (img, [(mask, "Mask")])

def run_nerf(params):

    video_file = params[video]
    video_name = os.path.basename(video_file)
    video_length = get_video_duration(video_file)
    gradio_dir = os.path.dirname(video_file)

    with tempfile.TemporaryDirectory() as tempdir:
        shutil.copy2(video_file, tempdir)
        print("Running COLMAP")
        subprocess.run([sys.executable,
                        os.path.join(ROOT_DIR,"dependencies/instant-ngp/scripts/colmap2nerf.py"), 
                        "--video_in", os.path.join(tempdir, video_name),
                        "--run_colmap",
                        "--aabb_scale", "1",
                        "--video_fps", str(int(100 / video_length)),
                        "--overwrite"], cwd=tempdir)

        if params[use_per_image]:
            with open(os.path.join(tempdir, "transforms.json"), "r") as transforms:
                data = commentjson.load(transforms)
            data["n_extra_learnable_dims"] = 16
            with open(os.path.join(tempdir, "transforms.json"), "w") as transforms:
                commentjson.dump(data, transforms)
            shutil.copy2(os.path.join(tempdir, "transforms.json"), gradio_dir)

        print("Training NeRF")
        subprocess.run([sys.executable,
                        os.path.join(ROOT_DIR, "dependencies/instant-ngp/scripts/run.py"),
                        "--n_steps", f"{params[n_steps]}",
                        "--save_snapshot", "snapshot.ingp",
                        "--save_mesh", "model.obj",
                        os.path.join(tempdir, "transforms.json")], cwd=tempdir)

        shutil.copy2(os.path.join(tempdir, "snapshot.ingp"), gradio_dir)
        shutil.copy2(os.path.join(tempdir, "model.obj"), gradio_dir)

    return {checkpoint_file: os.path.join(gradio_dir, "snapshot.ingp"), 
            model:os.path.join(gradio_dir, "model.obj")}

def create_video(params):
    return None

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

                    with gr.Row():
                        preview = gr.Button("Preview Segmentation")
                        preview.click(fn=preview_segmentation, inputs={video, text_prompt}, outputs=[segmentation], api_name="preview")
                        run = gr.Button("Submit")
                        run.click(fn=run_nerf, inputs={video, text_prompt, n_steps, use_per_image}, outputs=[model, checkpoint_file], api_name="nerf")


            with gr.Column():
                with gr.Box():
                    with gr.Tab("Preview"):
                        segmentation.render()
                    with gr.Tab("Results"):
                        model.render()
                        checkpoint_file.render()
                        with gr.Box():
                            orbit_video.render()
                            with gr.Accordion("Video Parameters", open=True):
                                render_vid = gr.Button("Render Video")
                                render_vid.click(fn=create_video, inputs={checkpoint_file}, outputs=[orbit_video], api_name="get_video")

        gr.Examples([["examples/cube_clean.mp4", "cube"]], inputs=[video, text_prompt])

    demo.queue()
    demo.launch()