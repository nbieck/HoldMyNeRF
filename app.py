import gradio as gr
import cv2
import numpy as np
import tempfile
import subprocess
import os
import shutil
import sys
import commentjson
import math
from dependencies.hloc.hloc import extract_features, pairs_from_exhaustive, match_features, reconstruction, triangulation
import pycolmap
import argparse
import logging
from pathlib import Path
from scipts.seem_extraction import SEEMPipeline, SEEMPreview
from dependencies.instant_ngp.scripts.colmap2nerf import run_ffmpeg

HEADER_TEXT = """
# 🍻 Hold My NeRF
[Code](https://github.com/nbieck/HoldMyNeRF)

## Instructions
1. Provide a video of you turning the object to be captured in your hand.
2. Provide a text prompt describing the object with a single word.
3. Preview the segmentation to ensure that the object is shown correctly
4. Start processing by pressing "Submit"
5. A 3D model and the Instant-NGP checkpoint will be available for download once completed
6. If desired, a video orbit of the object can be rendered from the NeRF directly
"""
ROOT_DIR = os.path.dirname(os.path.realpath(__file__))

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--port", "-p", type=int, default=7860)

    public = parser.add_argument_group('public server')
    public.add_argument("--public", action="store_true", help="Make the server publically accessible. Consider setting authentication if doung so.")
    public.add_argument("--auth_user", help="Username to authenticate with when running publically accessible. Please provide both username and password, if using authentication.")
    public.add_argument("--auth_pwd", help="Password to use for authentication. Please provide both username and password, if using authentication.")
    public.add_argument("--auth_message", help="Message to display on the authentication screen.")
    public.add_argument("--server_name", help="0.0.0.0 to access from outside Docker containers.")

    return parser.parse_args()

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

def check_input_present_or_raise(params):
    if not params[video] or not params[text_prompt]:
        raise gr.Error("Please provide both a video and a text prompt.")

def preview_segmentation(params):
    print("a")

    check_input_present_or_raise(params)

    print("b")

    video_file = params[video]
    gradio_dir = os.path.dirname(video_file)

    get_first_frame(video_file)

    print("c")

    img = os.path.join(gradio_dir, "first_frame.png")
    mask = SEEMPreview(img, params[text_prompt], params[use_rembg])

    mask = mask.astype(np.float32)
    mask /= 255.
    mask *= 0.5

    return (img, [(mask, params[text_prompt])])

#used to convert a dict into something of the format a.key = value (needed to invoke run_ffmpeg)
Object = lambda **kwargs: type("Object", (), kwargs)()

def mask_frames(params, progress=gr.Progress()):
    check_input_present_or_raise(params)

    video_file = params[video]
    video_name = os.path.basename(video_file)
    video_length = get_video_duration(video_file)


    tempdir = tempfile.mkdtemp(prefix="HMN")

    shutil.copy2(video_file, tempdir)
    progress((0,5), desc="Extracting Frames")
    run_ffmpeg(Object(
        overwrite=True, 
        images=os.path.join(tempdir, "frames"),
        video_in=os.path.join(tempdir, video_name),
        video_fps=(params[num_frames] / video_length),
        time_slice=None))

    progress((1,5), desc="Removing Background")
    masked_dir = os.path.join(tempdir, "masked")
    SEEMPipeline(os.path.join(tempdir, "frames"), masked_dir, params[text_prompt], params[use_rembg])


    zipf = shutil.make_archive(os.path.join(tempdir, "intermediates"), "zip", masked_dir)
    with os.scandir(os.path.join(tempdir, "masked")) as it:
        images = [f.path for f in it if f.is_file()]

    print("Masked")

    return {intermediates: [zipf],
            masked_images: images}

FEATURE_ALGORITHMS = [
    "COLMAP",
    "COLMAP_Exhaustive",
    "SuperPoint+LightGlue"
]

def run_nerf(params, progress=gr.Progress()):
    print("RUNNING NERF")
    print(params)


    tempdir = tempfile.mkdtemp(prefix="HMN", suffix="nerf")

    masked_dir = os.path.join(tempdir, "masked")
    if intermediates in params:
        intermediates_zip = params[intermediates][0]
        shutil.unpack_archive(intermediates_zip.name, masked_dir)
    else:
        os.mkdir(masked_dir)
        imgs = [img for (img, _) in params[images]]
        for img in imgs:
            shutil.copy2(img, masked_dir)

    progress((2,5), desc="Reconstruction")
    if params[features] == "SuperPoint+LightGlue":
        pairs = Path(os.path.join(tempdir, "pairs"))
        feats = Path(os.path.join(tempdir, "features.h5"))
        matches = Path(os.path.join(tempdir, "matches.h5"))
        db = Path(os.path.join(tempdir, "colmap.db"))

        extract_features.main(extract_features.confs["superpoint_aachen"],
                              Path(masked_dir),
                              Path(tempdir),
                              feature_path=feats)
        pairs_from_exhaustive.main(pairs,
                                        features=feats)
        match_features.main(match_features.confs["superpoint+lightglue"],
                                 pairs, feats, matches=matches)
        
        reconstruction.create_empty_db(db)
        reconstruction.import_images(Path(masked_dir), db, pycolmap.CameraMode.SINGLE)
        image_ids = reconstruction.get_image_ids(db)
        triangulation.import_features(image_ids, db, feats)
        triangulation.import_matches(image_ids, db, pairs, matches, skip_geometric_verification=True)
    else:
        subprocess.run(["colmap",
                        "feature_extractor",
                        "--ImageReader.camera_model", "OPENCV",
                        "--SiftExtraction.estimate_affine_shape=true",
                        "--SiftExtraction.domain_size_pooling=true",
                        "--ImageReader.single_camera", "1",
                        "--ImageReader.camera_params", "",
                        "--database_path", "colmap.db",
                        "--image_path", masked_dir], cwd=tempdir, check=True)
        matcher = "sequential"
        if "Exhaustive" in params[features]:
            matcher = "exhaustive"
        subprocess.run(["colmap",
                        f"{matcher}_matcher",
                        "SiftMatching.guided_matching=true",
                        "--database_path", "colmap.db"], cwd=tempdir, check=True)


    os.mkdir(os.path.join(tempdir, "sparse"))
    if (params[glomap]):
        subprocess.run(["glomap",
                        "mapper",
                        "--database_path", "colmap.db",
                        "--image_path", masked_dir,
                        "--output_path", "sparse"], cwd=tempdir, check=True)
    else:
        subprocess.run(["colmap",
                        "mapper",
                        "--database_path", "colmap.db",
                        "--image_path", masked_dir,
                        "--output_path", "sparse",
                        "--Mapper.init_num_trials", str(params[num_colmap_trials]),
                        "--Mapper.max_reg_trials", str(params[num_reg_trials])], cwd=tempdir, check=True);
        subprocess.run(["colmap",
                        "bundle_adjuster",
                        "--input_path", "sparse/0",
                        "--output_path", "sparse/0",
                        "--BundleAdjustment.refine_principal_point", "1"], cwd=tempdir, check=True)

    os.mkdir(os.path.join(tempdir, "text"))
    subprocess.run(["colmap",
                    "model_converter",
                    "--input_path", "sparse/0",
                    "--output_path", "text",
                    "--output_type", "TXT"], cwd=tempdir, check=True)

    progress((3,5), desc="Extracting camera data")
    subprocess.run([sys.executable,
                    os.path.join(ROOT_DIR,"dependencies/instant_ngp/scripts/colmap2nerf.py"), 
                    "--images", masked_dir,
                    "--text", "text",
                    "--aabb_scale", "1",
                    "--overwrite"], cwd=tempdir)

    if params[use_per_image]:
        with open(os.path.join(tempdir, "transforms.json"), "r") as transforms:
            data = commentjson.load(transforms)
        data["n_extra_learnable_dims"] = 16
        with open(os.path.join(tempdir, "transforms.json"), "w") as transforms:
            commentjson.dump(data, transforms)

    progress((4,5), desc="Training NeRF")
    subprocess.run([sys.executable,
                    os.path.join(ROOT_DIR, "dependencies/instant_ngp/scripts/run.py"),
                    "--n_steps", f"{params[n_steps]}",
                    "--save_snapshot", "snapshot.ingp",
                    "--save_mesh", "model.ply",
                    "--marching_cubes_res", "128",
                    os.path.join(tempdir, "transforms.json")], cwd=tempdir)

    progress((5,5), desc="Completed")

    zipdir = os.path.join(tempdir, "zip")
    os.mkdir(zipdir)
    shutil.copytree(os.path.join(tempdir, "sparse"), os.path.join(zipdir, "sparse"))
    shutil.copytree(os.path.join(tempdir, "text"), os.path.join(zipdir, "text"))
    shutil.copy2(os.path.join(tempdir, "colmap.db"), zipdir)
    colmap_data = shutil.make_archive(os.path.join(tempdir, "colmap"), "zip", zipdir)
    shutil.rmtree(zipdir)

    print("NERF DONE")

    return {nerf_files: [os.path.join(tempdir, "snapshot.ingp"), colmap_data],
            model: os.path.join(tempdir, "model.ply")}

def create_video_defaults(params):
    params[video_width] = 720
    params[video_height] = 480
    params[fps] = 30
    params[seconds] = 5
    params[spp] = 8
    return create_video(params)

def create_video(params):
    checkpoint_file = [f.name for f in params[nerf_files] if f.name.endswith(".ingp")][0]
    gradio_dir = os.path.dirname(checkpoint_file)
    videofile = os.path.join(gradio_dir, "video.mp4")

    if os.path.isfile(videofile):
        os.remove(videofile)

    subprocess.run([
        sys.executable,
        os.path.join(ROOT_DIR, "dependencies/instant_ngp/scripts/run.py"),
        "--load_snapshot", checkpoint_file,
        "--width", f"{params[video_width]}",
        "--height", f"{params[video_height]}",
        "--video_camera_path", os.path.join(ROOT_DIR, "config/camera_path.json"),
        "--video_fps", f"{params[fps]}",
        "--video_n_seconds", f"{params[seconds]}",
        "--video_spp", f"{params[spp]}",
    ], cwd=gradio_dir)

    return videofile

def regen_model_fn(files, resolution):
    snapshot = [f.name for f in files if f.name.endswith(".ingp")][0]
    gradio_dir = os.path.dirname(snapshot)

    subprocess.run([
        sys.executable,
        os.path.join(ROOT_DIR, "dependencies/instant_ngp/scripts/run.py"),
        "--load_snapshot", snapshot,
        "--save_mesh", "model.ply",
        "--marching_cubes_res", f"{resolution}"
    ], cwd=gradio_dir)

    return os.path.join(gradio_dir, "model.ply")

if __name__ == "__main__":
    logging.basicConfig(stream = sys.stdout)
    #inputs
    video = gr.Video(format="mp4", sources=["upload"], label="Video", interactive=True)
    text_prompt = gr.Textbox(label="Object Label", info="Provide a label for the object for segmentation", interactive=True)
    images = gr.Gallery(label="Frames", interactive=True)

    #segmentation preview
    segmentation = gr.AnnotatedImage(label="Segmentation")

    #outputs
    nerf_files = gr.File(label="Instant-NPG output", interactive=False, file_count="multiple")
    orbit_video = gr.Video(label="Orbit Video", interactive=False)
    model = gr.Model3D(label="Model", interactive=False, display_mode="solid", clear_color=(0,0,0,0))
    masked_images = gr.Gallery(label="Masked Frames", interactive=False, preview=True)
    intermediates = gr.Files(label="Intermediate Files", interactive=False)
    settings = gr.Markdown(label="Run Settings")

    with gr.Blocks() as demo:
        gr.Markdown(HEADER_TEXT)

        with gr.Row():
            with gr.Column():

                with gr.Accordion("Parameters", open=False):
                    with gr.Tab("BG Parameters"):
                        use_rembg = gr.Checkbox(value=True, label="Use rembg", info="Remove background before segmenting. Can improve or worsen performance.")
                        num_frames = gr.Slider(minimum=20, maximum=200, step=1, value=100, label="Number of frames")
                    with gr.Tab("COLMAP Params"):
                        features = gr.Dropdown(choices=FEATURE_ALGORITHMS, label="Feature Extractor+Matcher", multiselect=False)
                        glomap = gr.Checkbox(value=False, label="Use GLOMAP")
                        num_colmap_trials = gr.Slider(minimum=200, maximum=500, step=10, label="COLMAP trials")
                        num_reg_trials = gr.Slider(minimum=3, maximum=20, step=1, label="Max registration trials")
                    with gr.Tab("NERF Params"):
                        use_per_image = gr.Checkbox(value=True, label="Per Image Latents", info="Associates a trainable embedding with input images. Can accomodate changes in lighting.")
                        n_steps = gr.Number(value=1000, label="#Steps", precision=0, info="Number of steps to train NeRF.")

                with gr.Tab("Video Input"):
                    video.render()
                    text_prompt.render()
                    with gr.Row():
                        preview = gr.Button("Preview Segmentation")
                        preview.click(fn=preview_segmentation, inputs={video, text_prompt, use_rembg}, outputs=[segmentation], api_name="preview")
                        run = gr.Button("Submit")
                        run.click(fn=lambda: [None]*4,
                                outputs=[masked_images, intermediates, nerf_files, orbit_video]
                            ).success(
                                inputs = {use_rembg, num_frames, features, glomap, num_colmap_trials, num_reg_trials, use_per_image, n_steps},
                                outputs=settings,
                                api_name="save_settings",
                                fn = lambda par: "```\n" + "\n".join([comp.label + ": " + f"{value}" for (comp, value) in par.items()]) + "\n```"
                            ).success(fn=mask_frames, 
                                inputs={video, text_prompt, use_rembg, num_frames}, 
                                outputs=[masked_images, intermediates, nerf_files], 
                                api_name="mask_frames"
                            ).success(
                                fn=run_nerf,
                                inputs={intermediates, use_per_image, n_steps, features, glomap, num_colmap_trials, num_reg_trials},
                                outputs=[nerf_files, model],
                                api_name="run_nerf"
                            ).success(
                                fn=create_video_defaults,
                                inputs={nerf_files},
                                outputs=[orbit_video],
                                api_name="default_video"
                            )
                with gr.Tab("Image Input"):
                    images.render()
                    run_img = gr.Button("Run")
                    run_img.click(fn=lambda: [None]*4,
                                  outputs=[masked_images, intermediates, nerf_files, orbit_video]
                                ).success(
                                    fn=run_nerf,
                                    inputs={images, use_per_image, n_steps, features, glomap, num_colmap_trials, num_reg_trials},
                                    outputs=[nerf_files, model],
                                    api_name="run_nerf_imgs"
                                ).success(
                                    fn=create_video_defaults,
                                    inputs={nerf_files},
                                    outputs=[orbit_video],
                                    api_name="default_video"
                                )


            with gr.Column():
                with gr.Tab("Preview"):
                    segmentation.render()
                with gr.Tab("Results"):
                    with gr.Row():
                        model_res = gr.Number(value=128, label="Marching cubes resolution", precision=0, info="Spatial resolution of the grid used for marching cubes.")
                        regen_model = gr.Button("Regenerate Model")
                        regen_model.click(fn=regen_model_fn, inputs=[nerf_files, model_res], outputs=[model], api_name="regen_model")

                    with gr.Accordion("Parameters"):
                        settings.render()

                    with gr.Accordion("Frames", open=False):
                        intermediates.render()
                        masked_images.render()

                    nerf_files.render()
                    model.render()
                    orbit_video.render()

                    with gr.Accordion("Video Parameters", open=False):
                        with gr.Row():
                            video_width = gr.Number(value=720, label="Width", precision=0)
                            video_height = gr.Number(value=480, label="Height", precision=0)
                        fps = gr.Slider(minimum=10, maximum=60, value=30, label="FPS", step=10)
                        seconds = gr.Number(value=5, label="Video Length (s)", precision=1)
                        spp = gr.Slider(1,16,8, label="Samples per Pixel", info="Improves visual result at the cost of longer rending time.")
                        render_vid = gr.Button("Render Video")
                        render_vid.click(fn=create_video, 
                                            inputs={nerf_files, video_width, video_height,
                                                    fps, seconds, spp}, outputs=[orbit_video], api_name="get_video")

        gr.Examples([["examples/cube_clean.mp4", "cube"],
                     ["examples/flower_handheld.mp4", "flower"]], inputs=[video, text_prompt])

    args = parse_args()
    demo.queue()

    if args.public:
        demo.launch(server_port=args.port, share=True, 
                    auth=(args.auth_user, args.auth_pwd) if (args.auth_user and args.auth_pwd) else None,
                    auth_message=args.auth_message,
                    server_name=args.server_name)
    else:
        demo.launch(server_port=args.port, server_name=args.server_name)