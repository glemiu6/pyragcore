#pyragcore/utils_io/voice.py
from embeddings.embedder import Embedder
class Voice:
    def __init__(self):
        self.language=None


    def listen(self):
        import speech_recognition as sr
        recognize = sr.Recognizer()
        recognize.pause_threshold = 2.0  # if there is no sound for 2 seconds , it will stop
        try:
            print("Available microphones:")
            print(sr.Microphone.list_microphone_names())

            with sr.Microphone() as source:
                print("Adjusting noise...")
                recognize.adjust_for_ambient_noise(source, duration=1.5)
                print("recording for 4 seconds")
                recorded_audio = recognize.listen(source, timeout=4,
                                                  phrase_time_limit=10)  # timeout =the time it wait for the user to speak ; phrase_time_limit= time to speak is there is no silence detected
                print("done")
                text:str = recognize.recognize_google(recorded_audio)
                lang, _ = Embedder.detect_language(texts=text)
                self.language = lang
                return text
        except sr.UnknownValueError:
            print("Could not understand audio")

    def speak(self, text: str, language: str = "en") -> None:
        """
        Convert text to speech using gTTS and play it.
        """
        from gtts import gTTS
        import tempfile
        import os
        import platform

        tts = gTTS(text=text, lang=language)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
            tts.save(fp.name)
            temp_file = fp.name

        # Play the audio depending on OS
        if platform.system() == "Darwin":  # macOS
            os.system(f"afplay '{temp_file}'")
        elif platform.system() == "Windows":
            os.system(f"start /min wmplayer '{temp_file}'")
        else:  # Linux
            os.system(f"mpg123 '{temp_file}'")

        # Delete temporary file after playing
        os.remove(temp_file)