from urllib.parse import parse_qs, urlparse

from youtube_transcript_api import YouTubeTranscriptApi

from Config.Config import Config


class YouTubeTranscript:

    def __init__(self):

        self.config = Config()
        self.model = self.config.get_model()

    def _video_id(self, url):

        if "youtu.be" in url:
            return url.split("/")[-1].split("?")[0]

        p = urlparse(url)

        if p.path == "/watch":
            return parse_qs(p.query)["v"][0]

        if p.path.startswith("/shorts/"):
            return p.path.split("/")[2]

        if p.path.startswith("/embed/"):
            return p.path.split("/")[2]

        raise Exception("Invalid YouTube URL")

    def transcribe(self, url):

        vid = self._video_id(url)

        api = YouTubeTranscriptApi()

        transcript = api.fetch(vid)

        segments = []
        full_text = []

        for i, seg in enumerate(transcript):

            text = seg.text.strip()

            full_text.append(text)

            segments.append({
                "id": i + 1,
                "start": round(seg.start, 2),
                "end": round(seg.start + seg.duration, 2),
                "duration": round(seg.duration, 2),
                "text": text
            })

        return {
            "video_id": vid,
            "language": transcript.language,
            "language_code": transcript.language_code,
            "is_generated": transcript.is_generated,
            "segment_count": len(segments),
            "full_text": " ".join(full_text),
            "segments": segments
        }
    
"""
Returned JSON Structure
=======================

{
    "video_id": str,
        # YouTube video's unique ID.

    "language": str,
        # Human-readable transcript language.
        # Example: "English"

    "language_code": str,
        # ISO language code.
        # Example: "en"

    "is_generated": bool,
        # True if YouTube auto-generated the captions.
        # False if the captions were manually created.

    "segment_count": int,
        # Total number of transcript segments.

    "full_text": str,
        # Complete transcript as a single string.

    "segments": [
        {
            "id": int,
                # Segment number (starts from 1).

            "start": float,
                # Segment start time in seconds.

            "end": float,
                # Segment end time in seconds.

            "duration": float,
                # Length of this segment in seconds.

            "text": str
                # Transcript text for this segment.
        },
        ...
    ]
}

Example
-------

{
    "video_id": "7VHH1QNErDE",
    "language": "English",
    "language_code": "en",
    "is_generated": True,
    "segment_count": 152,
    "full_text": "Hello everyone...",

    "segments": [
        {
            "id": 1,
            "start": 0.0,
            "end": 3.12,
            "duration": 3.12,
            "text": "Hello everyone."
        },
        {
            "id": 2,
            "start": 3.12,
            "end": 7.48,
            "duration": 4.36,
            "text": "Welcome back to our channel."
        }
    ]
}
"""