"""FFmpeg command construction for every backend (no encoder or GPU needed)."""

import pytest

from av1sfm.encode import EncodeParams, _first_error, ffmpeg_command

SRC = ["-framerate", "10", "-i", "seq/%06d.png"]


def cmd_for(**kw):
    return ffmpeg_command(SRC, "out.ivf", EncodeParams(**kw), num_frames=117, ffmpeg="ffmpeg")


def opt(cmd, name):
    return cmd[cmd.index(name) + 1]


def test_libaom_streaming_configuration():
    c = cmd_for(encoder="libaom")
    assert opt(c, "-c:v") == "libaom-av1"
    assert opt(c, "-usage") == "realtime" and opt(c, "-cpu-used") == "6"
    assert opt(c, "-lag-in-frames") == "0" and opt(c, "-crf") == "32"
    assert opt(c, "-g") == "118" and opt(c, "-keyint_min") == "118"  # one keyframe
    assert opt(c, "-vf") == "format=yuv420p" and c[-3:] == ["-f", "ivf", "out.ivf"]


def test_svtav1_low_delay_rtc():
    c = cmd_for(encoder="svtav1", svt_preset=12, threads=4, svt_params="enable-tf=0")
    assert opt(c, "-c:v") == "libsvtav1" and opt(c, "-preset") == "12"
    params = opt(c, "-svtav1-params").split(":")
    assert params == ["pred-struct=1", "rtc=1", "keyint=-1", "lp=4", "enable-tf=0"]


def test_svtav1_pads_odd_sizes_only():
    def vf(size, encoder="svtav1"):
        c = ffmpeg_command(
            SRC, "o.ivf", EncodeParams(encoder=encoder), num_frames=3, size=size, ffmpeg="ffmpeg"
        )
        return opt(c, "-vf")

    assert vf((1241, 376)) == "pad=1242:376,fillborders=right=1:bottom=0:mode=smear,format=yuv420p"
    assert vf((321, 181)) == "pad=322:182,fillborders=right=1:bottom=1:mode=smear,format=yuv420p"
    assert vf((1920, 1280)) == "format=yuv420p"
    assert vf((1241, 376), encoder="libaom") == "format=yuv420p"  # libaom codes odd sizes


def test_vulkan_uploads_to_device_and_disables_b_frames():
    c = cmd_for(encoder="vulkan", qp=100, hw_device="1")
    assert c[c.index("-init_hw_device") + 1] == "vulkan=vk:1"
    assert opt(c, "-filter_hw_device") == "vk"
    assert opt(c, "-vf") == "format=nv12,hwupload"
    assert opt(c, "-c:v") == "av1_vulkan" and opt(c, "-rc_mode") == "cqp"
    assert opt(c, "-qp") == "100" and opt(c, "-bf") == "0" and opt(c, "-g") == "118"
    # global device options must precede the input
    assert c.index("-init_hw_device") < c.index("-i")


def test_qsv_render_node_via_vaapi():
    c = cmd_for(encoder="qsv", hw_device="/dev/dri/renderD129")
    inits = [c[i + 1] for i, x in enumerate(c) if x == "-init_hw_device"]
    assert inits == ["vaapi=va:/dev/dri/renderD129", "qsv=qs@va"]
    assert opt(c, "-vf") == "format=nv12,hwupload=extra_hw_frames=64"
    assert opt(c, "-c:v") == "av1_qsv" and opt(c, "-q:v") == "128" and opt(c, "-bf") == "0"
    assert opt(cmd_for(encoder="qsv"), "-init_hw_device") == "qsv=qs"


def test_vaapi_constant_qindex_on_render_node():
    c = cmd_for(encoder="vaapi", qp=110, hw_device="/dev/dri/renderD128")
    assert opt(c, "-init_hw_device") == "vaapi=va:/dev/dri/renderD128"
    assert opt(c, "-filter_hw_device") == "va" and opt(c, "-vf") == "format=nv12,hwupload"
    assert opt(c, "-c:v") == "av1_vaapi" and opt(c, "-rc_mode") == "CQP"
    # -global_quality (not -q:v, which would set QSCALE and divide by FF_QP2LAMBDA)
    assert opt(c, "-global_quality") == "110" and "-q:v" not in c
    assert opt(c, "-bf") == "0" and opt(c, "-g") == "118"
    assert c.index("-init_hw_device") < c.index("-i")
    assert opt(cmd_for(encoder="vaapi"), "-init_hw_device") == "vaapi=va"


def test_scale_is_applied_before_upload():
    c = ffmpeg_command(
        SRC,
        "o.ivf",
        EncodeParams(encoder="vulkan"),
        num_frames=3,
        scale=(640, 360),
        ffmpeg="ffmpeg",
    )
    assert opt(c, "-vf") == "scale=640:360:flags=area,format=nv12,hwupload"


def test_unknown_encoder():
    with pytest.raises(ValueError):
        cmd_for(encoder="av1_nvenc")


def test_first_error_prefers_specific_line():
    log = "[Vulkan] GPU listing\n[av1_vulkan] Device does not support the VK_KHR_video_encode_queue extension!\n[out#0] Nothing was written"
    assert "VK_KHR_video_encode_queue" in _first_error(log)
    assert _first_error("") == "failed"
