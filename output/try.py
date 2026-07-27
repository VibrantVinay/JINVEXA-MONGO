from output.tts import TextToSpeech

tts = TextToSpeech()

tts.speak(
    txt="Hello everyone. Welcome!",
    gender="female",
    output="female.mp3"
)

tts.speak(
    txt="Hello everyone. Welcome!",
    gender="male",
    output="male.mp3"
)

print("Done!")