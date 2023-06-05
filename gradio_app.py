import gradio as gr
import cv2
import numpy as np

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

def get_first_frame(video: str):
    if video != "":
        cap = cv2.VideoCapture(video)
        is_read, img = cap.read()
        if is_read:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return gr.update(value=img, interactive=True)
    return gr.update(value=None, interactive=False)

def preview_segmentation(params):
    img = np.random.random(size=(300,300,3))
    mask = np.zeros(shape=(300,300))
    mask[30:40, 30:40] = 1

    return (img, [(mask, "Mask")])

def run_nerf(params):
    return {checkpoint_file: None, model:None}

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
                        pass

                    with gr.Row():
                        preview = gr.Button("Preview Segmentation")
                        preview.click(fn=preview_segmentation, inputs={video, text_prompt}, outputs=[segmentation], api_name="preview")
                        run = gr.Button("Submit")
                        run.click(fn=run_nerf, inputs={video, text_prompt}, outputs=[model, checkpoint_file], api_name="nerf")


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