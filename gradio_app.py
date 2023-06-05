import gradio as gr
import cv2

HEADER_TEXT = """
# Hold My NeRF

---

Reconstructing a 3d model of an object from a video of it being held in the hand with a static camera using NeRF.
"""

def get_first_frame(video: str):
    if video != "":
        cap = cv2.VideoCapture(video)
        is_read, img = cap.read()
        if is_read:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return gr.update(value=img, interactive=True)
    return gr.update(value=None, interactive=False)

if __name__ == "__main__":
    #inputs
    video = gr.Video(format="mp4", source="upload", label="Video", interactive=True)
    mask_draw = gr.Image(label="Mask", source="upload", tool="sketch", interactive=False)
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
                    mask_draw.render()
                    text_prompt.render()

                    with gr.Accordion("NeRF Parameters", open=False):
                        pass

                    with gr.Row():
                        preview = gr.Button("Preview Segmentation")
                        run = gr.Button("Submit")

                    video.change(fn=get_first_frame, inputs=[video], outputs=[mask_draw])

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

        gr.Examples([["examples/cube_clean.mp4", "cube"]], inputs=[video, text_prompt])

    demo.launch()