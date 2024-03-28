from __future__ import annotations

import os
import tempfile

from typing import TYPE_CHECKING, Optional, Iterator, Callable, Iterable, Literal, List, Union, Tuple, Dict

from aniportrait.utils.utils import reiterator

if TYPE_CHECKING:
    from numpy import ndarray as NDArray
    from PIL.Image import Image
    from moviepy.editor import (
        VideoFileClip,
        AudioFileClip,
        CompositeaudioClip
    )

__all__ = ["Audio", "Video"]

class Audio:
    def __init__(
        self,
        frames: Iterable[Tuple[float]],
        rate: Optional[int]=None
    ) -> None:
        self.frames = reiterator(frames)
        self.rate = rate

    def get_clip(
        self,
        rate: Optional[int]=None,
        maximum_seconds: Optional[float]=None
    ) -> AudioClip:
        """
        Gets the moviepy audioclip
        """
        if not rate:
            rate = self.rate
        if not rate:
            rate = 44100

        from moviepy.editor import AudioClip

        all_frames = [frame for frame in self.frames] # type: ignore
        if maximum_seconds is not None:
            all_frames = all_frames[:int(maximum_seconds*rate)]

        total_frames = len(all_frames)
        duration = total_frames / rate

        def get_frame(time: float) -> Tuple[float]:
            if isinstance(time, int) or isinstance(time, float):
                return all_frames[int(total_frames*time)]
            return [ # type: ignore[unreachable]
                all_frames[int(t*rate)]
                for t in time
            ]

        return AudioClip(get_frame, duration=duration, fps=rate)

    def get_composite_clip(
        self,
        rate: Optional[int]=None,
        maximum_seconds: Optional[float]=None
    ) -> CompositeAudioClip:
        """
        Gets the moviepy composite audioclip
        """
        from moviepy.editor import CompositeAudioClip
        return CompositeAudioClip([
            self.get_clip(rate=rate, maximum_seconds=maximum_seconds)
        ])

    def save(
        self,
        path: str,
        rate: Optional[int]=None,
        maximum_seconds: Optional[float]=None,
    ) -> int:
        """
        Saves the audio frames to file
        """
        if not rate:
            rate = self.rate
        if not rate:
            rate = 44100
        if path.startswith("~"):
            path = os.path.expanduser(path)
        clip = self.get_clip(rate=rate, maximum_seconds=maximum_seconds)
        clip.write_audiofile(path)
        if not os.path.exists(path):
            raise IOError(f"Nothing was written to {path}.")
        size = os.path.getsize(path)
        if size == 0:
            raise IOError(f"Nothing was written to {path}.")
        return size

    @classmethod
    def file_to_frames(
        cls,
        path: str,
        skip_frames: Optional[int] = None,
        maximum_frames: Optional[int] = None,
        on_open: Optional[Callable[[AudioFileClip], None]] = None
    ) -> Iterable[Tuple[float]]:
        """
        Starts an audio capture and yields tuples for each frame.
        """
        from moviepy.editor import AudioFileClip
        if path.startswith("~"):
            path = os.path.expanduser(path)
        if not os.path.exists(path):
            raise IOError(f"Audio at path {path} not found or inaccessible")

        clip = AudioFileClip(path)
        if on_open is not None:
            on_open(clip)

        total_frames = 0
        for i, frame in enumerate(clip.iter_frames()):
            if skip_frames is not None and i < skip_frames:
                continue
            if maximum_frames is not None and total_frames + 1 > maximum_frames:
                break
            yield frame
            total_frames += 1

        if total_frames == 0:
            raise IOError(f"No frames were read from audio at path {path}")

    @classmethod
    def from_file(
        cls,
        path: str,
        skip_frames: Optional[int] = None,
        maximum_frames: Optional[int] = None,
        on_open: Optional[Callable[[AudioFileClip], None]] = None
    ) -> Audio:
        """
        Uses Audio.frames_from_file and instantiates an Audio object.
        """
        rate: Optional[int] = None
        audio = None
        def get_rate_on_open(clip: AudioFileClip) -> None:
            nonlocal audio
            audio.rate = clip.fps
            if on_open is not None:
                on_open(clip)

        audio = cls(
            frames=cls.file_to_frames(
                path=path,
                skip_frames=skip_frames,
                maximum_frames=maximum_frames,
                on_open=get_rate_on_open
            )
        )
        return audio

    @classmethod
    def combine(
        cls,
        *audios: Union[str, List[Tuple[float]]],
        **kwargs: Any
    ) -> Audio:
        """
        Combines multiple audio chunks
        """
        num_audios = len(audios)
        rate: Optional[float] = kwargs.get("rate", None)
        silence: Optional[float] = kwargs.get("silence", None)

        def maybe_get_rate_on_open(clip: AudioFileClip) -> None:
            nonlocal rate
            if rate is None:
                rate = clip.fps

        def iterate_audio() -> Iterable[Tuple[float]]:
            nonlocal rate
            channels: Optional[int] = None
            for i, audio in enumerate(audios):
                if isinstance(audio, str):
                    frames = cls.file_to_frames(
                        path=audio,
                        on_open=maybe_get_rate_on_open
                    )
                else:
                    frames = audio
                for frame in frames:
                    if channels is None:
                        channels = len(frame)
                    yield frame
                if i < num_audios - 1 and silence:
                    silence_frames = (rate if rate else 44100) * silence
                    silence_channels = (channels if channels else 1)
                    for j in range(int(silence_frames)):
                        yield (0,) * silence_channels

        return cls(frames=iterate_audio(), rate=rate)

class Video:
    """
    Provides helper methods for video
    """
    audio: Optional[Audio] = None

    def __init__(
        self,
        frames: Iterable[Image],
        frame_rate: Optional[float]=None,
        audio: Optional[Union[str, Audio]]=None,
        audio_frames: Optional[Iterable[Tuple[float]]]=None,
        audio_rate: Optional[int]=None,
    ) -> None:
        self.frames = reiterator(frames)
        self.frame_rate = frame_rate
        self.audio_rate = audio_rate
        if audio is not None:
            self.audio = audio
        elif audio_frames is not None:
            self.audio = Audio(frames=audio_frames, rate=audio_rate)
        else:
            self.audio = None

    def save(
        self,
        path: str,
        overwrite: bool=False,
        rate: Optional[float]=None,
        audio_rate: Optional[int]=None,
        crf: Optional[int]=18,
    ) -> int:
        """
        Saves PIL image frames to a video.
        Returns the total size of the video in bytes.
        """
        if rate is None:
            rate = self.frame_rate
        if rate is None:
            raise ValueError(f"Rate cannot be None.")
        if audio_rate is None:
            audio_rate = self.audio_rate
        if path.startswith("~"):
            path = os.path.expanduser(path)
        if os.path.exists(path):
            if not overwrite:
                raise IOError(f"File exists at path {path}, pass overwrite=True to write anyway.")
            os.unlink(path)
        basename, ext = os.path.splitext(os.path.basename(path))
        if ext in [".gif", ".png", ".tiff", ".webp"]:
            frames = [frame for frame in self.frames] # type: ignore[attr-defined]
            if rate > 50:
                logger.warning(f"Rate {rate} exceeds maximum frame rate (50), clamping.")
                rate = 50
            frames[0].save(path, loop=0, duration=1000.0/rate, save_all=True, append_images=frames[1:])
            return os.path.getsize(path)
        elif ext not in [".mp4", ".ogg", ".webm"]:
            raise IOError(f"Unknown file extension {ext}")

        from moviepy.editor import ImageSequenceClip
        from moviepy.video.io.ffmpeg_writer import ffmpeg_write_video
        import numpy as np

        clip_frames = [np.array(frame) for frame in self.frames] # type: ignore[attr-defined]
        clip = ImageSequenceClip(clip_frames, fps=rate)

        if self.audio is not None:
            audio_file = os.path.join(tempfile.mkdtemp(), "audio.mp3")
            maximum_seconds = len(clip_frames)/rate
            if isinstance(self.audio, str):
                Audio.from_file(self.audio).save(
                    audio_file,
                    rate=audio_rate,
                    maximum_seconds=maximum_seconds,
                )
            else:
                self.audio.save(
                    audio_file,
                    rate=audio_rate,
                    maximum_seconds=maximum_seconds
                )
        else:
            audio_file = None

        ffmpeg_write_video(
            clip,
            path,
            rate,
            audiofile=audio_file,
            ffmpeg_params=[] if crf is None else ["-crf", str(crf)]
        )

        if not os.path.exists(path):
            raise IOError(f"Nothing was written to {path}")
        if audio_file is not None:
            try:
                os.remove(audio_file)
            except:
                pass
        return os.path.getsize(path)

    @classmethod
    def file_to_frames(
        cls,
        path: str,
        skip_frames: Optional[int] = None,
        maximum_frames: Optional[int] = None,
        divide_frames: Optional[int] = None,
        on_open: Optional[Callable[[VideoFileClip], None]] = None,
    ) -> Iterator[Image]:
        """
        Starts a video capture and yields PIL images for each frame.
        """
        if path.startswith("~"):
            path = os.path.expanduser(path)
        if not os.path.exists(path):
            raise IOError(f"Video at path {path} not found or inaccessible")

        i = 0
        frame_start = 0 if skip_frames is None else skip_frames
        frame_end = None if maximum_frames is None else frame_start + (maximum_frames * (1 if not divide_frames else divide_frames)) - 1

        basename, ext = os.path.splitext(os.path.basename(path))
        if ext in [".gif", ".png", ".apng", ".tiff", ".webp", ".avif"]:
            from PIL import Image
            image = Image.open(path)
            for i in range(image.n_frames):
                if frame_start > i:
                    continue
                if divide_frames is not None and (i - frame_start) % divide_frames != 0:
                    continue
                image.seek(i)
                yield image.convert("RGBA")
                if frame_end is not None and i >= frame_end:
                    break
            return

        from moviepy.editor import VideoFileClip
        from PIL import Image

        clip = VideoFileClip(path)
        if on_open is not None:
            on_open(clip)

        for frame in clip.iter_frames():
            i += 1

            if frame_start > i:
                continue
            if divide_frames is not None and (i - frame_start) % divide_frames != 0:
                continue

            yield Image.fromarray(frame)

            if frame_end is not None and i >= frame_end:
                break

        if i == 0:
            raise IOError(f"No frames were read from video at {path}")

    @classmethod
    def from_file(
        cls,
        path: str,
        skip_frames: Optional[int] = None,
        maximum_frames: Optional[int] = None,
        divide_frames: Optional[int] = None,
        on_open: Optional[Callable[[VideoFileClip], None]] = None,
    ) -> Video:
        """
        Uses Video.frames_from_file and instantiates a Video object.
        """
        video = None

        def set_rate_on_open(clip: VideoFileClip) -> None:
            nonlocal video
            video.frame_rate = clip.fps
            if on_open is not None:
                on_open(clip)

        video = cls(
            frames=cls.file_to_frames(
                path=path,
                skip_frames=skip_frames,
                divide_frames=divide_frames,
                maximum_frames=maximum_frames,
                on_open=set_rate_on_open,
            )
        )
        return video
