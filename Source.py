import numpy as np
# for text-to-speech
from gtts import gTTS
# for language model
import transformers
import os
import datetime
import time

# Beginning of the AI
class ChatBot():
    def __init__(self, name):
        print("----- starting up", name, "-----")
        self.name = name

    def text_input(self):
        self.text = input("me --> ")

    def wake_up(self, text):
        return True if self.name.lower() in text.lower() else False

    @staticmethod
    def text_to_speech(name, text):
        print(name + "--> ", text)
        speaker = gTTS(text=text, lang="en", slow=False)
        speaker.save("res.mp3")
        statbuf = os.stat("res.mp3")
        mbytes = statbuf.st_size / 1024
        duration = mbytes / 200
        os.system('start res.mp3')  # if you are using mac->afplay or else for windows->start
        # os.system("close res.mp3")
        time.sleep(int(50 * duration))
        os.remove("res.mp3")

    @staticmethod
    def action_time():
        return datetime.datetime.now().time().strftime('%H:%M')

# Execute the AI
if __name__ == "__main__":
     ai = ChatBot(name="Boty")
     nlp = transformers.pipeline("conversational", model="microsoft/DialoGPT-medium")
     os.environ["TOKENIZERS_PARALLELISM"] = "true"
     ex = True
     while ex:
         # ai.speech_to_text()
         ai.text_input()
         ## wake up
         if ai.wake_up(ai.text) is True:
             res = "Hello I am " + ai.name +" the AI, what can I do for you?"
             ## do any action
         elif "time" in ai.text:
             res = ai.action_time()
         #model info
         elif "your model" in ai.text:
             res = "My model is microsoft/DialoGPT-medium. The model is trained on 147M multi-turn dialogue from Reddit discussion thread."
         ## respond politely
         elif any(i in ai.text for i in ["thank", "thanks"]):
             res = np.random.choice(
                 ["you're welcome!", "anytime!", "no problem!", "cool!", "I'm here if you need me!", "peace out!"])
         elif any(i in ai.text for i in ["exit", "close"]):
             res = np.random.choice(["Tata", "Have a good day", "Bye", "Goodbye", "Hope to meet soon", "peace out!", "I will be back"])
             ex = False
         else:
             if ai.text == "ERROR":
                 res = "Sorry, come again?"
             ## conversation
             else:
                 chat = nlp(transformers.Conversation(ai.text), pad_token_id=50256)
                 res = str(chat)
                 res = res[res.find("bot >> ") + 6:].strip()
         ai.text_to_speech(ai.name, res)
     print("----- Closing down "+ ai.name +" -----")