# AniPortrait
AniPortrait: Audio-Driven Synthesis of Photorealistic Portrait Animations

Huawei Wei, Zejun Yang, Zhisheng Wang

Tencent Games Zhiji, Tencent

<!-- <a href='https://humanaigc.github.io/emote-portrait-alive/'><img src='https://img.shields.io/badge/Project-Page-Green'></a> -->
<a href='https://arxiv.org'><img src='https://img.shields.io/badge/Paper-Arxiv-red'></a>

## Pipeline
![pipeline](content/intro.png)

TODO

## Various Generated Videos

### Self driven

<!-- [<img src="asset/mn.gif" width="500">](https://github.com/hpcaitech/Open-Sora/assets/99191637/de1963d3-b43b-4e68-a670-bb821ebb6f80)
[<img src="asset/jinji.gif" width="500">](https://github.com/hpcaitech/Open-Sora/assets/99191637/de1963d3-b43b-4e68-a670-bb821ebb6f80) -->

<div class="row">
<video id="video" controls="" preload="none" width="500">
      <source id="mp4" src="asset/cxk_cxk_pose_video_512x512_3_1631.mp4" type="video/mp4">
</video>
<video id="video" controls="" preload="none" width="500">
      <source id="mp4" src="asset/solo_song_solo1_kps_512x512_3_1017.mp4" type="video/mp4">
</video>
</div>

<!-- <div class="row">
<video id="video" controls="" preload="none" width="500">
      <source id="mp4" src="asset/jijin_driving_audio_2-jijin_512x512_3_1631.mp4" type="video/mp4">
</video>
<video id="video" controls="" preload="none" width="500">
      <source id="mp4" src="asset/mn_driving_audio_2-mn_512x512_3_1631.mp4" type="video/mp4">
</video>
</div> -->

<div class="row">
<video id="video" controls="" preload="none" width="500">
      <source id="mp4" src="asset/陈墨瞳_3D_陈墨瞳合成-陈墨瞳_3D_512x512_3_1631.mp4" type="video/mp4">
</video>
<video id="video" controls="" preload="none" width="500">
      <source id="mp4" src="asset/楚子航_3D_楚子航合成-楚子航_3D_512x512_3_1631.mp4" type="video/mp4">
</video>
</div>

<div class="row">
<video id="video" controls="" preload="none" width="500">
      <source id="mp4" src="asset/李云龙_李云龙合成-李云龙_512x512_3_1631.mp4" type="video/mp4">
</video>
<video id="video" controls="" preload="none" width="500">
      <source id="mp4" src="asset/张亮_张亮合成-张亮_512x512_3_1631.mp4" type="video/mp4">
</video>
</div>

### Face reenacment
<div class="row">
<video id="video" controls="" preload="none" width="500">
      <source id="mp4" src="/apdcephfs_qy3/share_1474453/huaweiwei_tmp/workspace/AniPortrait/asset/Aragaki_pain_512x512_3_1811.mp4" type="video/mp4">
</video>
<video id="video" controls="" preload="none" width="500">
      <source id="mp4" src="/apdcephfs_qy3/share_1474453/huaweiwei_tmp/workspace/AniPortrait/asset/time_is_up_butterfly_512x512_3_1738.mp4" type="video/mp4">
</video>
</div>

### Audio driven
<div class="row">
<video id="video" controls="" preload="none" width="500">
      <source id="mp4" src="/apdcephfs_qy3/share_1474453/huaweiwei_tmp/workspace/AniPortrait/asset/jijin_talk_hb_512x512_3_1624.mp4" type="video/mp4">
</video>
<video id="video" controls="" preload="none" width="500">
      <source id="mp4" src="/apdcephfs_qy3/share_1474453/huaweiwei_tmp/workspace/AniPortrait/asset/kara_talk_kara_512x512_3_1603.mp4" type="video/mp4">
</video>
</div>

## Installation

### Build environment
We Recommend a python version >=3.10 and cuda version =11.7. Then build environment as follows:
```
pip install -r requirements.txt
```

### Download weights
We will upload them to huggingface soon!

All the weights should be placed under the `./pretrained_weights` direcotry. You can download weights manually as follows:

1. Download our trained [weights](https://huggingface.co/patrolli/AnimateAnyone/tree/main), which include four parts: `denoising_unet.pth`, `reference_unet.pth`, `pose_guider.pth`, `motion_module.pth`, `audio2mesh.pt` and `audio2pose.pt`.

2. Download pretrained weight of based models and other components: 
    - [StableDiffusion V1.5](https://huggingface.co/runwayml/stable-diffusion-v1-5)
    - [sd-vae-ft-mse](https://huggingface.co/stabilityai/sd-vae-ft-mse)
    - [image_encoder](https://huggingface.co/lambdalabs/sd-image-variations-diffusers/tree/main/image_encoder)
    - [wav2vec2-base-960h](?)

3. Download dwpose weights (`dw-ll_ucoco_384.onnx`, `yolox_l.onnx`) following [this](https://github.com/IDEA-Research/DWPose?tab=readme-ov-file#-dwpose-for-controlnet).

Finally, these weights should be orgnized as follows:

```text
./pretrained_weights/
|-- DWPose
|   |-- dw-ll_ucoco_384.onnx
|   `-- yolox_l.onnx
|-- image_encoder
|   |-- config.json
|   `-- pytorch_model.bin
|-- audio2mesh.pt
|-- audio2pose.pt
|-- denoising_unet.pth
|-- motion_module.pth
|-- pose_guider.pth
|-- reference_unet.pth
|-- sd-vae-ft-mse
|   |-- config.json
|   |-- diffusion_pytorch_model.bin
|   `-- diffusion_pytorch_model.safetensors
|-- stable-diffusion-v1-5
|   |-- feature_extractor
|   |   `-- preprocessor_config.json
|   |-- model_index.json
|   |-- unet
|   |   |-- config.json
|   |   `-- diffusion_pytorch_model.bin
|   `-- v1-inference.yaml
`-- wav2vec2-base-960h
```

Note: If you have installed some of the pretrained models, such as `StableDiffusion V1.5`, you can specify their paths in the config file (e.g. `./config/prompts/animation.yaml`).

## Inference
Here is the cli command for running inference scripts:

### Self driven
```
python -m scripts.pose2vid --config ./configs/prompts/animation.yaml -W 512 -H 512 -L 64
```

You can refer the format of animation.yaml to add your own reference images or pose videos. To convert the raw video into a pose video (keypoint sequence), you can run with the following command:
```
python -m scripts.vid2pose --video_path pose_video_path.mp4
```

### Face reenacment
```
python -m scripts.vid2vid --config ./configs/prompts/animation_facereenac.yaml -W 512 -H 512 -L 64
```
Add source face videos and reference images in the animation_facereenac.yaml.

### Audio driven
```
python -m scripts.audio2vid --config ./configs/prompts/animation_audio.yaml -W 512 -H 512 -L 64
```
Add audios and reference images in the animation_audio.yaml.

## Training
Comming soon!
 
## Citation
```
@misc{wei2024aniportrait,
      title={AniPortrait: Audio-Driven Synthesis of Photorealistic Portrait Animations}, 
      author={Huawei Wei and Zejun Yang and Zhisheng Wang},
      year={2024},
      eprint={*},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```