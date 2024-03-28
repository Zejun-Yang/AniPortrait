from __future__ import annotations

import os

from aniportrait.utils.utils import reiterator
from typing import TYPE_CHECKING, Iterable, Tuple, Optional, List, Dict, Callable

if TYPE_CHECKING:
    from moviepy.editor import (
        AudioClip,
        AudioFileClip,
        CompositeAudioClip
    )

__all__ = ["Audio"]

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
        rate: Optional[int]=None
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
        clip = self.get_clip(rate=rate)
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

        def get_rate_on_open(clip: AudioFileClip) -> None:
            nonlocal rate
            rate = clip.fps
            if on_open is not None:
                on_open(clip)

        frames=cls.file_to_frames(
            path=path,
            skip_frames=skip_frames,
            maximum_frames=maximum_frames,
            on_open=get_rate_on_open
        )
        return cls(frames=frames, rate=rate)

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
