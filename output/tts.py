import asyncio
import edge_tts
import os


class TextToSpeech:
    def __init__(self):
        self.voices = {
            "male": "en-US-AndrewMultilingualNeural",
            "female": "en-US-AvaMultilingualNeural"
        }

    async def _generate(self, text, voice, output):
        communicate = edge_tts.Communicate(
            text=text,
            voice=voice,
            rate="-5%",      # Slightly slower for teaching
            pitch="+2Hz",    # Slightly brighter
            volume="+5%"
        )

        await communicate.save(output)

    def _prepare_text(self, text: str):
        text = text.strip()

        # Encourage natural pauses
        text = text.replace(". ", ".\n")
        text = text.replace("? ", "?\n")
        text = text.replace("! ", "!\n")
        text = text.replace(", ", ", ")

        return text

    def speak(self, txt, gender="female", output="speech.mp3"):
        gender = gender.lower()

        if gender not in self.voices:
            raise ValueError("Gender must be either 'male' or 'female'.")

        txt = self._prepare_text(txt)

        try:
            asyncio.run(
                self._generate(
                    txt,
                    self.voices[gender],
                    output
                )
            )
        except RuntimeError:
            loop = asyncio.get_event_loop()
            loop.run_until_complete(
                self._generate(
                    txt,
                    self.voices[gender],
                    output
                )
            )

        return os.path.abspath(output)

    async def speak_async(self, txt, gender="female", output="speech.mp3"):
        """Async version of speak - safe to call from within an event loop."""
        gender = gender.lower()

        if gender not in self.voices:
            raise ValueError("Gender must be either 'male' or 'female'.")

        txt = self._prepare_text(txt)

        await self._generate(
            txt,
            self.voices[gender],
            output
        )

        return os.path.abspath(output)

